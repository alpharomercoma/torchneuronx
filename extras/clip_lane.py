"""CLIP ViT-B/32 zero-shot classification on Inferentia2, mlx-models parity.

Mirror of github.com/alpharomercoma/mlx-models 4_clip/classify.py so the demo
is comparable across silicon (Apple M5 there, Inferentia2 here): the same COCO
demo image (two cats on a couch), the same label set, the same "a photo of
{label}" prompts, the same softmax-over-labels bar chart. What the port target
changes:

  * MLX builds CLIP from scratch and JITs dynamic shapes; here optimum-neuron
    0.4.3 traces the stock openai/clip-vit-base-patch32 into ONE static graph
    (text batch = number of labels, image batch 1, seq 77, 224px). The
    compile is a cached artifact under /opt/np/models/neuron-compiled/.
  * The traced unit is the FULL model -- every forward encodes 1 image plus
    all label texts -- so images/s @ batch 1 here is honest but conservative
    next to an image-encoder-only number.

    python3 extras/clip_lane.py --out /tmp/clip.json
    # expect (M5 reference, mlx 4_clip README): "two cats lying on a couch"
    # at ~100% softmax, every other label ~0%
"""
import argparse
import json
import os
import sys
import time
import traceback
from urllib.request import urlretrieve

MODEL_ID = "openai/clip-vit-base-patch32"
# Byte-identical demo inputs to mlx-models 4_clip/classify.py.
DEMO_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DEFAULT_LABELS = [
    "two cats lying on a couch",
    "a dog playing in a park",
    "a plate of food on a table",
    "a car driving on a road",
    "a person riding a bicycle",
]
SEQ_LEN = 77              # CLIP text context length (compiled static)
IMAGE_SIZE = 224
WARMUP_ITERS = 10
BENCH_ITERS = 100


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--compiled-dir",
                    default="/opt/np/models/neuron-compiled/clip-vit-base-patch32")
    ap.add_argument("--data-dir", default="/opt/np/models/neuron-compiled/data")
    ap.add_argument("--image", default=None)
    # NOTE: the compiled text batch is fixed at export time; a labels list
    # LONGER than the compiled batch needs a recompile (delete the compiled
    # dir or point --compiled-dir elsewhere).
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    return ap


def get_demo_image(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "cats.jpg")
    if not os.path.exists(path):
        print(f"downloading demo image -> {path}", flush=True)
        urlretrieve(DEMO_URL, path)
    return path


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, path)


def _record_failure(out, stage):
    """Structured failure receipt NEXT TO --out (launch_vllm load_failure
    pattern): the lane is retried on the next driver pass because --out
    itself stays absent, but the evidence survives."""
    reason = traceback.format_exc().strip().splitlines()[-1]
    path = out[:-5] + ".failure.json" if out.endswith(".json") \
        else out + ".failure.json"
    _atomic_json(path, {
        "lane": "clip", "status": "lane_failed", "stage": stage,
        "reason": reason,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(f"clip lane FAILED at {stage}: {reason}", file=sys.stderr)
    print(f"failure recorded: {path}", file=sys.stderr)


def main():
    args = build_parser().parse_args()

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor
        from optimum.neuron import NeuronCLIPModel
    except Exception:
        _record_failure(args.out, "import")
        return 1

    compiled = os.path.abspath(args.compiled_dir)
    cached = os.path.exists(os.path.join(compiled, "config.json"))
    compile_s = 0.0
    if not cached:
        print(f"--- compiling {args.model} -> {compiled} (one-time, minutes) ---",
              flush=True)
        t0 = time.perf_counter()
        try:
            # Export API per optimum-neuron 0.4.3 CLIP docs (task
            # feature-extraction, auto-detected from the model class):
            # static text/image batches + shapes, matmuls cast to bf16.
            model = NeuronCLIPModel.from_pretrained(
                args.model,
                export=True,
                text_batch_size=len(args.labels),
                image_batch_size=1,
                sequence_length=SEQ_LEN,
                num_channels=3,
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                auto_cast="matmul",
                auto_cast_type="bf16",
            )
            model.save_pretrained(compiled)
            AutoProcessor.from_pretrained(args.model).save_pretrained(compiled)
            del model
        except Exception:
            _record_failure(args.out, "compile")
            return 1
        compile_s = time.perf_counter() - t0

    # Always (re)load from the compiled dir so load_s means the same thing
    # on cold and warm runs.
    t0 = time.perf_counter()
    try:
        model = NeuronCLIPModel.from_pretrained(compiled)
        processor = AutoProcessor.from_pretrained(compiled)
    except Exception:
        _record_failure(args.out, "load")
        return 1
    load_s = time.perf_counter() - t0

    image_path = args.image or get_demo_image(args.data_dir)
    image = Image.open(image_path).convert("RGB")
    prompts = [f"a photo of {label}" for label in args.labels]

    print(f"# clip  model={args.model} text_batch={len(args.labels)} "
          f"image_batch=1 seq={SEQ_LEN} px={IMAGE_SIZE} auto_cast=matmul/bf16 "
          f"cached={cached} compiled_dir={compiled} "
          f"image={os.path.basename(str(image_path))} "
          f"({image.width}x{image.height})", flush=True)

    try:
        inputs = processor(
            text=prompts, images=image, return_tensors="pt",
            padding="max_length", max_length=SEQ_LEN, truncation=True,
        )
        feed = dict(input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs["attention_mask"])
        outputs = model(**feed)
        # logits_per_image: (1, num_labels). CLIP: softmax over labels.
        probs = torch.softmax(outputs.logits_per_image[0], dim=-1)
    except Exception:
        _record_failure(args.out, "inference")
        return 1

    order = sorted(range(len(args.labels)), key=lambda i: -probs[i].item())
    print("\nzero-shot probabilities (softmax over labels, sum to 1):")
    for rank, i in enumerate(order):
        bar = "#" * int(probs[i].item() * 30)
        mark = " <-- best" if rank == 0 else ""
        print(f"  {probs[i].item():6.1%}  {bar:<30} {args.labels[i]}{mark}")

    # -------- images/s at batch 1: the compiled graph is the whole model
    # (1 image + len(labels) texts per call), so this is end-to-end
    # zero-shot throughput, not image-encoder-only.
    try:
        for _ in range(WARMUP_ITERS):
            model(**feed)
        t0 = time.perf_counter()
        for _ in range(BENCH_ITERS):
            model(**feed)
        bench_wall = time.perf_counter() - t0
    except Exception:
        _record_failure(args.out, "throughput")
        return 1
    images_per_s = BENCH_ITERS / bench_wall

    top_label = args.labels[order[0]]
    payload = {
        "model": args.model,
        "probs": {args.labels[i]: round(probs[i].item(), 4)
                  for i in range(len(args.labels))},
        "top_label": top_label,
        "images_per_s_b1": round(images_per_s, 2),
        "bench_iters": BENCH_ITERS,
        "compile_s": round(compile_s, 1),
        "load_s": round(load_s, 1),
        "mlx_reference": "github.com/alpharomercoma/mlx-models 4_clip (Apple M5)",
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(args.out, payload)
    print(f"\ntop '{top_label}' ({probs[order[0]].item():.1%})  |  "
          f"{images_per_s:.1f} images/s @ b1 ({BENCH_ITERS} iters)  |  "
          f"compile {payload['compile_s']}s  load {payload['load_s']}s"
          f"  ->  {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
