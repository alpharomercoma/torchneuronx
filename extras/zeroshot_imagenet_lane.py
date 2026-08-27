"""ImageNet-1k zero-shot top-1 for CLIP / SigLIP -- the field-standard
image-text benchmark, run as a validity check on the Neuron compile.

WHAT THIS MEASURES AND WHY
--------------------------
`clip_lane.py` classifies ONE picture of two cats against FIVE labels and
reports 188.48 images/s. That is a smoke test wearing a benchmark's clothes:
five labels is a 20%-by-chance task, and the throughput number is unaffected
by whether the answer is right.

It also already produced the failure this lane exists to catch. On trn1 the
same traced graph compiled, loaded, ran at full speed and returned NaN
probabilities. Speed said "success". Only an accuracy number can say
otherwise -- and 1000-way zero-shot is 0.1%-by-chance, so a degraded graph
has nowhere to hide.

  benchmark   ImageNet-1k zero-shot classification (CLIP paper protocol;
              MLPerf has no CLIP task, this is the community standard)
  dataset     clip-benchmark/wds_imagenet1k -- LAION's mirror of the ILSVRC
              2012 validation set, and the exact artefact the published
              zero-shot numbers are computed on. Ungated.
  prompts     the repository's own classnames.txt (1000) and
              zeroshot_classification_templates.txt (80) -- the OpenAI prompt
              ensemble. Using the dataset's shipped strings rather than our
              own removes prompt wording as a free variable.
  metric      top-1 / top-5 accuracy
  gate        MLPerf's >=99%-of-reference rule, borrowed and re-anchored

THE COMPARISON THAT CARRIES THE CLAIM
-------------------------------------
Paired, same box, same images, same prompts:

    engine=cpu     stock float32 eager HF model on the instance's vCPUs
    engine=neuron  torch_neuronx-traced towers on a NeuronCore

Published top-1 (63.2% for CLIP ViT-B/32) is the SECONDARY anchor -- it
catches a broken evaluation loop, but it cannot adjudicate hardware because
it carries someone else's preprocessing.

WHAT RUNS WHERE -- stated exactly, because "on the accelerator" is a claim
-------------------------------------------------------------------------
  on the engine   the image tower, AND the text tower: the 1000-way classifier
                  is built by encoding 1000 x 80 prompts on the same engine
                  that encodes the images. Production systems cache text
                  embeddings, so running the text tower on host would be
                  defensible practice -- but it would hide half the compile
                  from the measurement, which is the thing this lane exists to
                  prevent.
  on the host     the final projection `image_features @ classifier.T`, the
                  L2 normalisations, and the top-k sort -- in float32, eager,
                  for BOTH arms.

That last line matters and cuts against us: a real deployment would run the
1000-way projection on-chip in bf16, where adjacent-class decisions are made
with an 8-bit mantissa instead of 24. Doing it on host in fp32 is IDENTICAL
across the two arms, so it cannot bias the paired delta -- but it does mean
the measured top-1 is a slight upper bound on a fully on-chip pipeline.

WHY THE TEXT TOWER TAKES NO ATTENTION MASK
-----------------------------------------
Because passing one returns NaN on a NeuronCore-v2, and this lane is what
found it. Bisected on inf2.xlarge, 2026-08-20, one variable per cell
(extras/clip_nan_bisect.py, receipt in results/clip_nan_bisect.json):

    cpu    text +mask    CLEAN     0/2048 non-finite
    cpu    text -mask    CLEAN     0/2048
    cpu    image         CLEAN     0/2048
    neuron text +mask    NaN       2048/2048     <-- and "Compiler status PASS"
    neuron text -mask    CLEAN     0/2048
    neuron image         CLEAN     0/2048

CLIP's text encoder builds a CAUSAL mask filled with torch.finfo(dtype).min
(-3.4e38 in float32) and then ADDS the padding mask, filled with the same
constant. -3.4e38 + -3.4e38 overflows to -inf, and softmax(-inf - -inf) is
NaN. On CPU the addition saturates harmlessly; through the Neuron compiler it
does not.

This is the SAME failure the joint CLIP trace hit on trn1 on 2026-07-31 and
did NOT hit on trn2 -- consistent with it being specific to NeuronCore-v2
(trn1, inf2) and absent on v3 (trn2). The graph compiles clean and runs at
1,165 images/s while returning NaN, which is precisely why a throughput
number cannot stand alone as evidence that a compile worked.

Dropping the mask is not a workaround, it is the REFERENCE RECIPE: open_clip
and LAION's clip_benchmark pad every prompt to 77 and pass no attention mask,
relying on the causal mask plus argmax-EOS pooling. Positions after the EOS
token cannot influence it, so the padding is inert by construction.
`--text-attention-mask on` reproduces the NaN as a scored number.

WHY --auto-cast DEFAULTS TO none
--------------------------------
Measured 2026-07-31: with the default matmul->bf16 cast, the traced CLIP
graph returns NaN logits. `--auto-cast matmult` is kept as a rung precisely so
that failure can be reported as a NUMBER (top-1 ~ 0.1%, chance) rather than
as an anecdote.

    python3 extras/zeroshot_imagenet_lane.py --engine cpu    --out .../zs_clip_cpu.json
    python3 extras/zeroshot_imagenet_lane.py --engine neuron --out .../zs_clip_neuron.json
    python3 extras/zeroshot_imagenet_lane.py --compare A.json B.json --out delta.json
"""
import argparse
import json
import os
import random
import sys
import tarfile
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "shared", "eval"))
import accuracy as A  # noqa: E402

REPO = "clip-benchmark/wds_imagenet1k"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
N_SHARDS = 7          # test/0.tar .. test/6.tar == the 50,000 ILSVRC val images
N_CLASSES = 1000

# Published zero-shot top-1, for the secondary anchor only. Sources are the
# CLIP paper (Radford et al. 2021) and the SigLIP paper (Zhai et al. 2023);
# both use the 80-template OpenAI ensemble on the same validation set.
PUBLISHED = {
    "openai/clip-vit-base-patch32": {
        "imagenet_zeroshot_top1": 0.632,
        "source": "CLIP paper / LAION clip_benchmark leaderboard"},
    "google/siglip-base-patch16-224": {
        "imagenet_zeroshot_top1": 0.760,
        "source": "SigLIP paper (Zhai et al. 2023)"},
}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", choices=["cpu", "neuron"], default="cpu")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--per-class", type=int, default=10,
                    help="images per class, seeded draw; 50 = the whole val set")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--image-batch", type=int, default=16)
    ap.add_argument("--text-attention-mask", choices=["off", "on"],
                    default="off",
                    help="off = the open_clip/clip_benchmark reference recipe, "
                         "and the only setting that does not return NaN on a "
                         "NeuronCore-v2. Applied to BOTH arms.")
    ap.add_argument("--cpu-dtype", choices=["float32", "bfloat16"],
                    default="float32",
                    help="CPU arm dtype. bfloat16 is the CONTROL for the "
                         "--auto-cast matmul rung: without it, a bf16 NaN on "
                         "Neuron cannot be told apart from a bf16 NaN in the "
                         "model itself.")
    # "matmult", not "matmul". neuronx-cc's CLI spells it with the trailing t
    # (`--auto-cast {none,matmult,all}`) while optimum-neuron's PYTHON api takes
    # auto_cast="matmul". This lane drives the compiler directly through
    # torch_neuronx.trace(compiler_args=...), so it needs the CLI spelling.
    # Measured 2026-08-20: the wrong one exits 2 with "invalid choice", which
    # surfaces as a bare `RuntimeError: neuronx-cc failed with 2` several
    # frames up -- loud, but it names the wrong layer.
    ap.add_argument("--auto-cast", choices=["none", "matmult", "all"],
                    default="none",
                    help="Neuron compiler cast. 'none' (fp32) is the honest "
                         "default; 'matmult' is the rung that reproduces the "
                         "measured bf16 NaN as a scored number.")
    ap.add_argument("--data-dir", default="/opt/np/models/imagenet1k")
    ap.add_argument("--compiled-dir", default=None)
    ap.add_argument("--max-templates", type=int, default=0,
                    help="0 = all 80 (the published recipe)")
    ap.add_argument("--compare", nargs=2, metavar=("CANDIDATE", "REFERENCE"))
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--keep-tars", action="store_true")
    return ap


# ----------------------------------------------------------------- dataset

def _fetch(url, dest, token):
    from urllib.request import Request, urlopen
    req = Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urlopen(req) as r, open(dest + ".part", "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(dest + ".part", dest)


def ensure_dataset(data_dir, keep_tars):
    """Download + explode all 7 shards into {data_dir}/images/{cls:04d}/{key}.jpg.

    ALL shards, not a prefix: the shards are CLASS-ORDERED (verified -- shard 0
    opens with 50 consecutive class-0 images), so taking the first k shards
    would score a few hundred classes and call it ImageNet. That is the kind
    of quiet subsetting this lane exists to rule out.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    img_root = os.path.join(data_dir, "images")
    done = os.path.join(data_dir, ".extracted")
    if os.path.exists(done):
        return img_root
    os.makedirs(img_root, exist_ok=True)
    for i in range(N_SHARDS):
        tar_path = os.path.join(data_dir, f"{i}.tar")
        if not os.path.exists(tar_path):
            print(f"downloading shard {i}/{N_SHARDS-1}", flush=True)
            _fetch(f"{BASE}/test/{i}.tar", tar_path, token)
        print(f"extracting shard {i}", flush=True)
        pending = {}
        with tarfile.open(tar_path, "r|") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                key, ext = os.path.splitext(m.name)
                # `key` becomes a filename below; a member named "../x" would
                # write outside img_root. Webdataset keys are flat, so anything
                # with a separator in it is malformed, not merely unusual.
                if os.sep in key or (os.altsep and os.altsep in key) \
                        or key in ("", ".", "..") or os.path.isabs(key):
                    raise RuntimeError(f"unsafe tar member name: {m.name!r}")
                data = tf.extractfile(m).read()
                if ext == ".cls":
                    pending.setdefault(key, {})["cls"] = int(data.decode().strip())
                else:
                    pending.setdefault(key, {})["img"] = data
                    pending[key]["ext"] = ext
                rec = pending.get(key, {})
                if "cls" in rec and "img" in rec:
                    d = os.path.join(img_root, f"{rec['cls']:04d}")
                    os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, key + rec["ext"]), "wb") as fh:
                        fh.write(rec["img"])
                    pending.pop(key)
        if not keep_tars:
            os.remove(tar_path)
    open(done, "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return img_root


def load_prompts(data_dir, max_templates):
    """classnames.txt + zeroshot_classification_templates.txt, from the dataset
    repo itself -- the same strings the published numbers were computed with."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    out = {}
    for name in ("classnames.txt", "zeroshot_classification_templates.txt"):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            _fetch(f"{BASE}/{name}", path, token)
        out[name] = [l.rstrip("\n") for l in open(path) if l.strip()]
    classnames = out["classnames.txt"]
    templates = out["zeroshot_classification_templates.txt"]
    if len(classnames) != N_CLASSES:
        raise SystemExit(f"expected {N_CLASSES} classnames, got {len(classnames)}")
    if max_templates:
        templates = templates[:max_templates]
    return classnames, templates


def select_images(img_root, per_class, seed):
    """Seeded, class-stratified draw. Sorted inputs, seeded per class, so the
    list is a pure function of (per_class, seed) and reproduces on any box.

    The sample key is CLASS-QUALIFIED ("0007/s0000342"). WebDataset basenames
    are only unique within a shard, and compare() pairs the two engines'
    per-image records through a dict keyed by this string -- an unqualified
    basename shared by two classes would silently pair a shark against a
    volcano and the bootstrap would run over cross-class comparisons.
    """
    chosen, short = [], []
    for c in range(N_CLASSES):
        d = os.path.join(img_root, f"{c:04d}")
        if not os.path.isdir(d):
            raise SystemExit(f"missing class dir {d} -- dataset incomplete")
        files = sorted(os.listdir(d))
        if len(files) < per_class:
            # A HALF-EXTRACTED dataset used to pass silently: the old
            # `take = files if per_class >= len(files)` clause took whatever
            # was there, so an extraction that died after shard 2 yielded
            # ~2,400 images across 1000 classes, both arms scored the same
            # 2,400, the digest matched, assert_paired passed, and the study
            # would have published "ImageNet-1k top-1" from a 24% slice.
            short.append((c, len(files)))
            continue
        rng = random.Random(f"{seed}:{c}")
        take = rng.sample(files, per_class)
        chosen.extend({"key": f"{c:04d}/{os.path.splitext(f)[0]}", "label": c,
                       "path": os.path.join(d, f)} for f in sorted(take))
    if short:
        raise SystemExit(
            f"dataset incomplete: {len(short)} of {N_CLASSES} classes hold "
            f"fewer than {per_class} images (e.g. {short[:5]}). Delete "
            f"{img_root}/../.extracted and re-stage rather than scoring a "
            "partial ImageNet.")
    chosen.sort(key=lambda r: (r["label"], r["key"]))
    return chosen


# ------------------------------------------------------------------ engines

def _wrappers(model_id):
    """(text_forward, image_forward, needs_attention_mask) for the model family.

    CLIP consumes an attention mask; SigLIP pads every prompt to a fixed 64
    and takes none. Both expose get_*_features, so one code path covers both
    and neither gets a bespoke, unaudited branch.
    """
    is_siglip = "siglip" in model_id.lower()
    return is_siglip


def load_cpu(args, n_templates):
    import torch
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.cpu_dtype]
    model = AutoModel.from_pretrained(args.model, torch_dtype=dt).eval()

    def encode_text(inputs):
        with torch.no_grad():
            return model.get_text_features(**inputs)   # caller drops the mask

    def encode_image(pixel_values):
        with torch.no_grad():
            return model.get_image_features(pixel_values=pixel_values.to(dt))

    return processor, encode_text, encode_image, {
        "dtype": args.cpu_dtype, "framework": "transformers eager",
        "auto_cast": args.auto_cast if args.cpu_dtype != "float32" else "none",
        "threads": torch.get_num_threads()}


def load_neuron(args, n_templates):
    """Trace the text and image towers separately, at fixed batch.

    Separate traces, not one joint graph, because the compiled text batch in
    the joint export is len(labels) -- fine for a 5-label demo, useless for a
    1000-class benchmark. Two towers is also how every real zero-shot
    deployment is built.
    """
    import torch
    import torch_neuronx
    from transformers import AutoModel, AutoProcessor

    is_siglip = _wrappers(args.model)
    compiled = os.path.abspath(args.compiled_dir or os.path.join(
        "/opt/np/models/neuron-compiled",
        "zs-" + args.model.split("/")[-1] + f"-{args.auto_cast}"
        f"-t{n_templates}-b{args.image_batch}-m{args.text_attention_mask}"))
    text_pt = os.path.join(compiled, "text_tower.pt")
    image_pt = os.path.join(compiled, "image_tower.pt")
    processor = AutoProcessor.from_pretrained(args.model)
    seq = 64 if is_siglip else 77
    px = getattr(processor, "image_processor", processor).size
    px = px.get("height") or px.get("shortest_edge") or 224

    compile_s = 0.0
    cached = os.path.exists(text_pt) and os.path.exists(image_pt)
    if not cached:
        print(f"--- tracing {args.model} towers (text_batch={n_templates}, "
              f"image_batch={args.image_batch}, auto_cast={args.auto_cast}) ---",
              flush=True)
        t0 = time.perf_counter()
        base = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).eval()

        use_mask = args.text_attention_mask == "on" and not is_siglip

        class TextTower(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            if use_mask:
                def forward(self, input_ids, attention_mask):
                    return self.m.get_text_features(input_ids=input_ids,
                                                    attention_mask=attention_mask)
            else:
                def forward(self, input_ids):
                    return self.m.get_text_features(input_ids=input_ids)

        class ImageTower(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pixel_values):
                return self.m.get_image_features(pixel_values=pixel_values)

        cargs = ["--auto-cast", args.auto_cast]
        if args.auto_cast != "none":
            cargs += ["--auto-cast-type", "bf16"]
        t_dummy = ((torch.ones((n_templates, seq), dtype=torch.int64),
                    torch.ones((n_templates, seq), dtype=torch.int64))
                   if use_mask else
                   (torch.ones((n_templates, seq), dtype=torch.int64),))
        os.makedirs(compiled, exist_ok=True)
        torch.jit.save(torch_neuronx.trace(TextTower(base), t_dummy,
                                           compiler_args=cargs), text_pt)
        torch.jit.save(torch_neuronx.trace(
            ImageTower(base),
            (torch.zeros((args.image_batch, 3, px, px), dtype=torch.float32),),
            compiler_args=cargs), image_pt)
        processor.save_pretrained(compiled)
        del base
        compile_s = time.perf_counter() - t0

    # A trace is only reusable for the EXACT protocol it was built for. The
    # default compiled path encodes cast/templates/batch, but an explicit
    # --compiled-dir does not, so a NaN-producing bf16 trace could be reloaded
    # by a run whose receipt then claimed auto_cast=none. The sidecar makes
    # that a loud refusal instead of a false record.
    want = {"model": args.model, "auto_cast": args.auto_cast,
            "text_batch": n_templates, "image_batch": args.image_batch,
            "seq": seq, "px": px, "text_attention_mask": args.text_attention_mask}
    meta_path = os.path.join(compiled, "trace_meta.json")
    if cached:
        got = json.load(open(meta_path)) if os.path.exists(meta_path) else None
        if got != want:
            raise SystemExit(
                f"compiled artefact at {compiled} was traced for {got}, this "
                f"run needs {want} -- refusing to reload a mismatched trace")
    else:
        A.atomic_json(meta_path, want)

    text_traced = torch.jit.load(text_pt)
    image_traced = torch.jit.load(image_pt)
    # Processor from the HUB SNAPSHOT, identical to the CPU arm. A processor
    # pickled at compile time under a different transformers could resize with
    # a different resample filter (bicubic vs bilinear is worth ~0.3pp top-1 on
    # ImageNet) and the delta would absorb a library skew as if it were silicon.
    processor = AutoProcessor.from_pretrained(args.model)

    def encode_text(inputs):
        if args.text_attention_mask == "on" and not is_siglip:
            return text_traced(inputs["input_ids"], inputs["attention_mask"])
        return text_traced(inputs["input_ids"])

    def encode_image(pixel_values):
        return image_traced(pixel_values)

    return processor, encode_text, encode_image, {
        "dtype": {"none": "float32", "matmult": "bf16 matmuls",
                  "all": "bf16"}[args.auto_cast],
        "framework": "torch_neuronx.trace (two towers)",
        "auto_cast": args.auto_cast, "text_attention_mask": args.text_attention_mask,
        "text_batch": n_templates,
        "image_batch": args.image_batch, "seq_len": seq, "image_px": px,
        "compile_s": round(compile_s, 1), "compiled_from_cache": cached,
        "compiled_dir": compiled}


# --------------------------------------------------------------------- run

def run_engine(args):
    import torch
    img_root = ensure_dataset(args.data_dir, args.keep_tars)
    classnames, templates = load_prompts(args.data_dir, args.max_templates)
    samples = select_images(img_root, args.per_class, args.seed)
    keys = [s["key"] for s in samples]
    digest = A.sample_digest(keys)
    print(f"# zeroshot engine={args.engine} model={args.model} "
          f"n_images={len(samples)} per_class={args.per_class} "
          f"templates={len(templates)} seed={args.seed} "
          f"auto_cast={args.auto_cast} digest={digest[:16]}", flush=True)

    processor, encode_text, encode_image, engine_meta = (
        load_cpu(args, len(templates)) if args.engine == "cpu"
        else load_neuron(args, len(templates)))
    is_siglip = _wrappers(args.model)
    wants_mask = args.text_attention_mask == "on" and not is_siglip
    tok_kw = dict(padding="max_length", max_length=64, truncation=True) \
        if is_siglip else dict(padding="max_length", max_length=77, truncation=True)

    # ---- classifier: one traced call per class (exactly n_templates prompts)
    t0 = time.perf_counter()
    rows = []
    for c, name in enumerate(classnames):
        prompts = [t.replace("{c}", name) for t in templates]
        inputs = processor(text=prompts, return_tensors="pt", **tok_kw)
        # The mask is dropped for BOTH engines or neither. Dropping it only on
        # the arm that NaNs without it would be exactly the kind of quiet
        # per-arm accommodation this lane exists to rule out.
        keep = ("input_ids", "attention_mask") if wants_mask else ("input_ids",)
        inputs = {k: v for k, v in inputs.items() if k in keep}
        emb = encode_text(inputs).float()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        emb = emb.mean(dim=0)
        rows.append(emb / emb.norm().clamp_min(1e-12))
        if args.progress_every and (c + 1) % max(1, args.progress_every // 5) == 0:
            print(f"  classifier {c+1}/{N_CLASSES}", flush=True)
    classifier = torch.stack(rows)          # (1000, D)
    text_s = time.perf_counter() - t0
    nonfinite_classifier = int((~torch.isfinite(classifier)).sum())

    # ---- images
    from PIL import Image
    B = args.image_batch
    records, logit_rows, labels = [], [], []
    img_s = 0.0
    t_start = time.perf_counter()
    for start in range(0, len(samples), B):
        chunk = samples[start:start + B]
        pil = [Image.open(s["path"]).convert("RGB") for s in chunk]
        pv = processor(images=pil, return_tensors="pt").pixel_values
        pad = 0
        if args.engine == "neuron" and pv.shape[0] < B:
            # The traced image tower has a FIXED batch. Pad, then discard the
            # padded rows before scoring -- they must never reach the metric.
            pad = B - pv.shape[0]
            pv = torch.cat([pv, pv[:1].repeat(pad, 1, 1, 1)], dim=0)
        t0 = time.perf_counter()
        feats = encode_image(pv).float()
        img_s += time.perf_counter() - t0
        if pad:
            feats = feats[:-pad]
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        logits = feats @ classifier.T
        for s, row in zip(chunk, logits):
            logit_rows.append(row.tolist())
            labels.append(s["label"])
        if args.progress_every and (start + B) % args.progress_every < B:
            print(f"  images {min(start+B, len(samples))}/{len(samples)}",
                  flush=True)
    total_wall = time.perf_counter() - t_start

    scored = A.topk_accuracy(logit_rows, labels, ks=(1, 5))
    # MARGIN, per image: the gap between the winning class and the runner-up.
    # Adversarial review's sharpest statistical point: bf16 rounding near a
    # decision boundary FLIPS a near-tie, and the delta then reports that as
    # an accuracy loss even though nothing was "answered wrong" in any
    # meaningful sense. Publishing the margin distribution lets a reader see
    # how much of any delta could be tie-flipping rather than degradation.
    margins = []
    for s, row, label in zip(samples, logit_rows, labels):
        finite = all(v == v and abs(v) != float("inf") for v in row)
        order = sorted(range(len(row)), key=lambda i: (-row[i], i)) if finite else []
        margin = (row[order[0]] - row[order[1]]) if len(order) > 1 else None
        if margin is not None:
            margins.append(margin)
        records.append({"key": s["key"], "label": label,
                        "pred": order[0] if finite else None,
                        "margin": margin,
                        "top1_hit": int(finite and order[0] == label),
                        "top5_hit": int(finite and label in order[:5]),
                        "finite": finite})
    margins.sort()
    tie_fragility = {
        "median_margin": margins[len(margins) // 2] if margins else None,
        "frac_margin_lt_1e-3": sum(1 for m in margins if m < 1e-3) / len(margins)
        if margins else None,
        "frac_margin_lt_1e-2": sum(1 for m in margins if m < 1e-2) / len(margins)
        if margins else None,
        "why": "images whose top-2 cosine similarities are this close can flip "
               "on rounding alone; they bound how much of a top-1 delta is "
               "tie-flipping rather than degradation",
    }

    pub = PUBLISHED.get(args.model)
    payload = {
        "lane": "zeroshot_imagenet", "engine": args.engine, "model": args.model,
        "benchmark": "ImageNet-1k zero-shot classification (CLIP protocol)",
        "dataset": {"repo": REPO, "shards": N_SHARDS, "per_class": args.per_class,
                    "seed": args.seed, "classes": N_CLASSES,
                    "templates": len(templates),
                    "prompt_source": "dataset repo classnames.txt + "
                                     "zeroshot_classification_templates.txt",
                    "text_attention_mask": args.text_attention_mask},
        "n_samples": len(samples), "sample_digest": digest,
        "engine_meta": engine_meta,
        "top1": scored["top1"], "top5": scored["top5"],
        "top1_hits": scored["top1_hits"], "top5_hits": scored["top5_hits"],
        "nonfinite_rows": scored["nonfinite_rows"],
        "degenerate_rows": scored["degenerate_rows"],
        "nonfinite_classifier_entries": nonfinite_classifier,
        "tie_fragility": tie_fragility,
        "timing": {
            "text_tower_s": round(text_s, 1),
            "image_forward_s": round(img_s, 2),
            "total_wall_s": round(total_wall, 1),
            # NAMED PRECISELY. This is the image tower alone. It excludes the
            # 1000x80 text-tower pass that builds the classifier, JPEG decode,
            # resize, and the host-side 1000-way projection. Quoting it as
            # "zero-shot images/s" would overstate the deployable rate; the
            # end-to-end figure below is the one to put on a slide.
            "image_tower_images_per_s": round(len(samples) / img_s, 2)
            if img_s else None,
            "end_to_end_images_per_s": round(len(samples) / total_wall, 2)
            if total_wall else None,
            "end_to_end_excludes": "classifier construction (text tower), "
                                   "counted separately as text_tower_s",
        },
        "published_anchor": pub,
        "chance_top1": 1.0 / N_CLASSES,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    A.atomic_json(args.out, payload)
    A.atomic_json(args.out.replace(".json", ".images.json"),
                  {"sample_digest": digest, "records": records})
    print(f"\ntop1 {scored['top1']*100:.2f}%  top5 {scored['top5']*100:.2f}%  "
          f"(nonfinite rows {scored['nonfinite_rows']}/{scored['n']}, "
          f"chance 0.10%)  ->  {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------- compare

def compare(args):
    cand_p, ref_p = args.compare
    cand = json.load(open(cand_p))
    ref = json.load(open(ref_p))
    A.assert_paired(cand, ref)
    # Without this, compare() would subtract an fp32 run from a bf16 run --
    # two different experiments -- and print -63pp as though the accelerator
    # had lost accuracy.
    A.assert_same_protocol(cand, ref)
    ci_recs = json.load(open(cand_p.replace(".json", ".images.json")))["records"]
    ri_recs = json.load(open(ref_p.replace(".json", ".images.json")))["records"]
    by_key = {r["key"]: r for r in ri_recs}
    units = [{"a_hit": c["top1_hit"], "b_hit": by_key[c["key"]]["top1_hit"]}
             for c in ci_recs if c["key"] in by_key]
    if len(units) != len(ci_recs):
        raise ValueError(f"image files disagree: {len(units)} of {len(ci_recs)} matched")
    ci = A.paired_bootstrap_ci(units, A.topk_delta_statistic)
    gate = A.accuracy_gate(cand["top1"], ref["top1"])
    flips = [c for c in ci_recs if c["pred"] != by_key[c["key"]]["pred"]]
    disagree = len(flips)
    # A flip on an image whose top-2 were within 1e-3 cosine is a rounding
    # coin-toss, not a wrong answer. Separating them stops a reader reading
    # "0.4pp lost" as degradation when it is tie noise.
    near_tie_flips = sum(1 for c in flips
                         if (c.get("margin") or 1.0) < 1e-3
                         or (by_key[c["key"]].get("margin") or 1.0) < 1e-3)
    payload = {
        "lane": "zeroshot_imagenet_delta",
        "candidate": {"engine": cand["engine"], "top1": cand["top1"],
                      "top5": cand["top5"],
                      "nonfinite_rows": cand["nonfinite_rows"],
                      "auto_cast": cand["engine_meta"].get("auto_cast"),
                      "receipt": cand_p},
        "reference": {"engine": ref["engine"], "top1": ref["top1"],
                      "top5": ref["top5"],
                      "nonfinite_rows": ref["nonfinite_rows"],
                      "receipt": ref_p},
        "n_samples": cand["n_samples"], "sample_digest": cand["sample_digest"],
        "top1_delta_pp": (cand["top1"] - ref["top1"]) * 100,
        "top1_delta_ci95_pp": {"lo": ci["lo"] * 100, "hi": ci["hi"] * 100,
                               "point": ci["point"] * 100,
                               "n_boot": ci["n_boot"], "seed": ci["seed"]},
        "prediction_disagreements": disagree,
        "prediction_disagreement_rate": disagree / len(ci_recs) if ci_recs else None,
        "disagreements_on_near_ties": near_tie_flips,
        "disagreements_decisive": disagree - near_tie_flips,
        "tie_fragility": {"candidate": cand.get("tie_fragility"),
                          "reference": ref.get("tie_fragility")},
        "mlperf_gate": gate,
        "interpretation": (
            "top1_delta_pp < 0 means the candidate engine classified FEWER of "
            "the same images correctly. prediction_disagreements counts images "
            "where the two engines picked different classes at all -- it moves "
            "even when accuracy happens to cancel out, so it is the more "
            "sensitive detector of a numerically drifting graph."),
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    A.atomic_json(args.out, payload)
    print(json.dumps({k: payload[k] for k in
                      ("top1_delta_pp", "top1_delta_ci95_pp",
                       "prediction_disagreements", "mlperf_gate")}, indent=1))
    return 0


def _record_failure(out, stage):
    reason = traceback.format_exc().strip().splitlines()[-1]
    path = out[:-5] + ".failure.json" if out.endswith(".json") else out + ".failure.json"
    A.atomic_json(path, {"lane": "zeroshot_imagenet", "status": "lane_failed",
                         "stage": stage, "reason": reason,
                         "traceback": traceback.format_exc(),
                         "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime())})
    print(f"zeroshot FAILED at {stage}: {reason}", file=sys.stderr)
    return 1


def main():
    args = build_parser().parse_args()
    if args.compare:
        try:
            return compare(args)
        except Exception:
            return _record_failure(args.out, "compare")
    try:
        return run_engine(args)
    except Exception:
        return _record_failure(args.out, "run")


if __name__ == "__main__":
    sys.exit(main())
