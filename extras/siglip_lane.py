"""SigLIP export ATTEMPT on Inferentia2 -- expected structured exclusion.

Mirror target: github.com/alpharomercoma/mlx-models 5_siglip/classify.py runs
google/siglip-base-patch16-224 zero-shot on Apple M5 (sigmoid, independent
per-label probabilities). On Neuron the validity check said no: optimum-neuron
0.4.3 registers NO siglip exporter config (model_configs.py has clip but not
siglip), so this lane exists to record that gap as a measurement, not to be
skipped silently.

The lane ATTEMPTS the export anyway and, on ANY failure, writes a structured
receipt to --out and exits 0 -- a recorded exclusion is an outcome, exactly
like launch_vllm's load_failure.json. If a future venv unexpectedly exports
it, the lane runs the mlx-parity zero-shot with sigmoid probs and records
status "unexpected_success" instead.

    python3 extras/siglip_lane.py --out /tmp/siglip.json
    # expect: {"status": "export_unsupported", "reason": "<verbatim error>",
    #          "mirror_note": "...Apple M5 -- ecosystem gap", ...}, exit 0
"""
import argparse
import json
import os
import sys
import time
import traceback
from urllib.request import urlretrieve

MODEL_ID = "google/siglip-base-patch16-224"
# Byte-identical demo inputs to mlx-models 5_siglip/classify.py.
DEMO_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DEFAULT_LABELS = [
    "two cats lying on a couch",
    "a dog playing in a park",
    "a plate of food on a table",
    "a car driving on a road",
    "a person riding a bicycle",
]
MIRROR_NOTE = "mlx-models 5_siglip runs this on Apple M5 -- ecosystem gap"


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--compiled-dir",
                    default="/opt/np/models/neuron-compiled/siglip-base-patch16-224")
    ap.add_argument("--data-dir", default="/opt/np/models/neuron-compiled/data")
    return ap


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, path)


def build_receipt(reason, status="export_unsupported"):
    """Structured attempt receipt. Pure; unit-tested."""
    return {
        "status": status,
        "model": MODEL_ID,
        "reason": reason,
        "mirror_note": MIRROR_NOTE,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_receipt(path, reason, status="export_unsupported"):
    """Build + atomically write the receipt; returns the payload written."""
    payload = build_receipt(reason, status)
    _atomic_json(path, payload)
    return payload


def get_demo_image(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "cats.jpg")
    if not os.path.exists(path):
        print(f"downloading demo image -> {path}", flush=True)
        urlretrieve(DEMO_URL, path)
    return path


def _last_error_line():
    return traceback.format_exc().strip().splitlines()[-1]


def main():
    args = build_parser().parse_args()

    print(f"# siglip  model={MODEL_ID} attempt-only "
          f"(no siglip exporter config in optimum-neuron 0.4.3; "
          f"expected structured exclusion)", flush=True)

    try:
        # Generic traced-export attempt: there is no NeuronSiglipModel, so
        # the closest class is the generic feature-extraction one. In 0.4.3
        # the TasksManager lookup for "siglip" fails before the static
        # shapes below are even consumed -- that verbatim error IS the
        # result of this lane.
        from optimum.neuron import NeuronModelForFeatureExtraction
        model = NeuronModelForFeatureExtraction.from_pretrained(
            MODEL_ID,
            export=True,
            # CLIP-shaped static axes, for the hypothetical future venv
            # where a siglip exporter config exists.
            text_batch_size=len(DEFAULT_LABELS),
            image_batch_size=1,
            sequence_length=64,
            num_channels=3,
            width=224,
            height=224,
        )
    except Exception:
        reason = _last_error_line()
        write_receipt(args.out, reason)
        print(f"export attempt failed as expected: {reason}")
        print(f"exclusion receipt -> {args.out}", flush=True)
        return 0

    # ------------------------------------------------ unexpected success
    print("UNEXPECTED: siglip export succeeded; running mlx-parity zero-shot",
          flush=True)
    try:
        model.save_pretrained(args.compiled_dir)
        import torch
        from PIL import Image
        from transformers import AutoProcessor

        image = Image.open(get_demo_image(args.data_dir)).convert("RGB")
        # SigLIP prompts are usually phrased "This is a photo of {label}."
        prompts = [f"This is a photo of {label}." for label in DEFAULT_LABELS]
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        inputs = processor(text=prompts, images=image,
                           padding="max_length", return_tensors="pt")
        outputs = model(**inputs)
        logits = getattr(outputs, "logits_per_image", None)
        if logits is None:
            logits = outputs[0]
        # SigLIP -> sigmoid for independent per-label probabilities.
        probs = torch.sigmoid(logits)[0]

        order = sorted(range(len(DEFAULT_LABELS)),
                       key=lambda i: -probs[i].item())
        print("\nzero-shot match probabilities (independent, sigmoid):")
        for rank, i in enumerate(order):
            bar = "#" * int(probs[i].item() * 30)
            mark = " <-- best" if rank == 0 else ""
            print(f"  {probs[i].item():6.1%}  {bar:<30} "
                  f"{DEFAULT_LABELS[i]}{mark}")

        payload = build_receipt(
            reason="export succeeded on this venv", status="unexpected_success")
        payload["probs"] = {DEFAULT_LABELS[i]: round(probs[i].item(), 4)
                            for i in range(len(DEFAULT_LABELS))}
        payload["top_label"] = DEFAULT_LABELS[order[0]]
        _atomic_json(args.out, payload)
        print(f"unexpected-success record -> {args.out}", flush=True)
    except Exception:
        reason = "export succeeded but inference failed: " + _last_error_line()
        write_receipt(args.out, reason, status="unexpected_success")
        print(f"{reason}\nreceipt -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
