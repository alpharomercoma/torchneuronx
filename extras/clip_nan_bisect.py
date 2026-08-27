"""Why does the CLIP text tower return NaN on a NeuronCore-v2?

Measured 2026-08-20 on inf2.xlarge: the zero-shot lane's traced text tower
returned NaN for all 1000 classes (512,000 non-finite classifier entries)
while the image tower ran clean at 1,165 images/s. Same failure signature as
trn1's joint CLIP trace (2026-07-31). trn2 -- NeuronCore-v3 -- does not show it.

HYPOTHESIS: mask doubling. CLIP's text encoder builds a CAUSAL mask filled
with torch.finfo(dtype).min (-3.4e38 in fp32) and then ADDS the padding mask,
also filled with finfo.min. -3.4e38 + -3.4e38 overflows to -inf, and
softmax(-inf - -inf) is NaN. The reference zero-shot implementations
(open_clip, LAION clip_benchmark) pass NO attention mask at all -- they pad to
77 and rely on the causal mask plus argmax-EOS pooling -- so the reference
recipe never triggers it.

Four cells, one variable each. Prints a table; changes nothing.

    python3 extras/clip_nan_bisect.py
"""
import json
import os
import sys
import time

MODEL = "openai/clip-vit-base-patch32"
SEQ, PX, B = 77, 224, 4


def finite_report(name, t):
    import torch
    bad = int((~torch.isfinite(t)).sum())
    return {"cell": name, "shape": list(t.shape), "nonfinite": bad,
            "total": int(t.numel()),
            "verdict": "CLEAN" if bad == 0 else "NaN/inf",
            "sample": [round(float(x), 5) for x in t.flatten()[:3]]}


def main():
    import torch
    import torch_neuronx
    from transformers import AutoModel, AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL)
    base = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    prompts = ["a photo of a tench.", "a photo of a goldfish.",
               "a photo of a shark.", "a photo of a hen."]
    enc = proc(text=prompts, return_tensors="pt", padding="max_length",
               max_length=SEQ, truncation=True)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    pv = torch.zeros((B, 3, PX, PX), dtype=torch.float32)

    class TextMasked(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, i, a): return s.m.get_text_features(input_ids=i, attention_mask=a)

    class TextNoMask(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, i): return s.m.get_text_features(input_ids=i)

    class Image(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, p): return s.m.get_image_features(pixel_values=p)

    rows = []
    # CPU controls first: if CPU is also NaN the model is the problem, not the
    # compile, and there is nothing here to attribute to the hardware.
    with torch.no_grad():
        rows.append(finite_report("cpu text  +mask", TextMasked(base)(ids, mask)))
        rows.append(finite_report("cpu text  -mask", TextNoMask(base)(ids)))
        rows.append(finite_report("cpu image", Image(base)(pv)))

    cargs = ["--auto-cast", "none"]
    for name, mod, args in (("neuron text +mask", TextMasked(base), (ids, mask)),
                            ("neuron text -mask", TextNoMask(base), (ids,)),
                            ("neuron image", Image(base), (pv,))):
        t0 = time.perf_counter()
        try:
            traced = torch_neuronx.trace(mod, args, compiler_args=cargs)
            out = traced(*args)
            r = finite_report(name, out)
        except Exception as exc:
            r = {"cell": name, "verdict": "TRACE FAILED",
                 "reason": f"{type(exc).__name__}: {exc}"[:200]}
        r["compile_s"] = round(time.perf_counter() - t0, 1)
        rows.append(r)
        print(f"  {r['cell']:<20} {r['verdict']:<12} "
              f"nonfinite={r.get('nonfinite')}/{r.get('total')} "
              f"({r['compile_s']}s)", flush=True)

    out = "/opt/np/results/clip_nan_bisect.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"model": MODEL, "auto_cast": "none", "seq": SEQ, "rows": rows,
               "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(out, "w"), indent=1)
    print("\n" + json.dumps(rows, indent=1)[:1200])
    print("->", out)


if __name__ == "__main__":
    sys.exit(main())
