#!/usr/bin/env python3
"""Lane 6: consolidate the TP-sharded LoRA adapter, merge it, ship it to S3.

Usage:

    python shared/train/merge_adapter.py \
        --base meta-llama/Llama-3.1-8B-Instruct \
        --adapter /opt/np/models/adapters/llama31_lora \
        --out trn1/results/train/merge_llama31.json

    # -> banner, then:
    #      [1/5] consolidate : 2 TP shards -> adapter_model.safetensors (84.1 MiB)
    #      [2/5] load base   : meta-llama/Llama-3.1-8B-Instruct  bf16 on CPU
    #      [3/5] merge       : merge_and_unload() ... 41.2 s
    #      [4/5] save        : /opt/np/models/llama31-8b-dolly-merged (14.97 GiB)
    #      [5/5] upload      : s3://neuron-pipelines-artifacts-.../artifacts/...
    #      SUMMARY base=... merged_out=... files=6 total_gib=14.97 upload_s=612.3
    #    and trn1/results/train/merge_llama31.json with a sha256 per file.

    # merge only, no artifact upload (dry runs, or an air-gapped box):
    python shared/train/merge_adapter.py --base ... --adapter ... --out ... --no-upload

WHY THIS IS A SEPARATE SCRIPT FROM sft_lora.py
----------------------------------------------
sft_lora.py runs under torchrun with two ranks and the Neuron runtime holding
both NeuronCores. This script runs single-process on the CPU and touches no
accelerator at all -- merging LoRA into base weights is a pure host-side
matrix add. Keeping them apart means a merge failure never costs a training
run, and the merge can be re-run (or moved to a different machine entirely)
without recompiling anything.

  !! HOST RAM WARNING -- THIS IS THE HEAVIEST STEP ON trn1.2xlarge !!
  ------------------------------------------------------------------
  trn1.2xlarge has 32 GiB of host RAM. This script must hold, on the host:
    * the base model in bf16                        ~16 GiB
    * the PeftModel wrapper's adapter tensors        ~0.1 GiB
    * the merged state dict during save_pretrained  up to another ~16 GiB,
      because safetensors serialisation materialises tensors it writes
  That is a peak in the high-20s of GiB against a 32 GiB budget shared with
  the page cache. It usually fits; it does not always fit.

  Mitigations already applied here: low_cpu_mem_usage=True (weights stream in
  shard by shard instead of being built twice), torch_dtype=bfloat16 (never
  fp32 -- fp32 alone would be 32 GiB and cannot fit), and an explicit gc pass
  before save_pretrained.

  If it still OOMs, DO NOT paper over it by adding swap and waiting hours.
  Use the local-merge fallback:
      # on the box
      aws s3 sync /opt/np/models/adapters/llama31_lora \\
          s3://neuron-pipelines-artifacts-600627330911/adapters/llama31_lora/
      # on a laptop/workstation with more RAM and AWS credentials
      aws s3 sync s3://.../adapters/llama31_lora/ ./llama31_lora/
      python shared/train/merge_adapter.py --base meta-llama/Llama-3.1-8B-Instruct \\
          --adapter ./llama31_lora --out merge_llama31.json \\
          --merged-out ./llama31-8b-dolly-merged
  Note there is NO --no-upload in the fallback: the whole point is that the
  laptop uploads the merged weights to the same bucket, so the inf2 box pulls
  a byte-identical artifact either way. The metrics JSON records the same
  sha256 set from either machine, which is what makes the two paths
  interchangeable rather than merely similar.

WHY CONSOLIDATION IS NEEDED FIRST
---------------------------------
Training ran with tensor_parallel_size=2, so optimum-neuron saved the adapter
as one shard per TP rank under <adapter>/adapter_default/shards/. peft cannot
read that layout. optimum-neuron ships the inverse transform --
consolidate_model_parallel_checkpoints_to_unified_checkpoint, which is also
what `optimum-cli neuron consolidate` calls -- and it does more than
concatenate: it reverses the QKV fusion and the GQA column-parallel layout
that the Neuron training modeling applied, so the result is a stock peft
adapter again. Doing this by hand with torch.cat would silently produce
transposed weights that load fine and generate garbage.

Already-consolidated adapters are passed through untouched, so this script is
idempotent and safe to re-run.
"""

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

DEFAULT_BASE = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MERGED_OUT = "/opt/np/models/llama31-8b-dolly-merged"
DEFAULT_BUCKET = "neuron-pipelines-artifacts-600627330911"
DEFAULT_S3_PREFIX = "artifacts"

# The files that determine what the inf2 box will actually execute. Same spirit
# as shared/verify_models.sh: hash what changes the outputs, ignore the rest.
ARTIFACT_PATTERNS = ("*.safetensors", "config.json", "tokenizer.json")

# optimum-neuron's sharded-checkpoint layout.
SHARDS_DIR_NAME = "shards"
ADAPTER_DIR_PREFIX = "adapter_"
CONSOLIDATED_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")

HASH_CHUNK_BYTES = 1 << 20


# --------------------------------------------------------------------- cli
def build_parser():
    """argparse builder. Top level and import-cheap, like sft_lora.py's."""
    ap = argparse.ArgumentParser(
        description="Consolidate + merge a Trainium LoRA adapter and ship it")
    ap.add_argument("--base", default=DEFAULT_BASE, help="base model HF id")
    ap.add_argument("--adapter", required=True,
                    help="adapter dir written by sft_lora.py (--adapter-out)")
    ap.add_argument("--out", required=True, help="metrics JSON path")
    ap.add_argument("--merged-out", default=DEFAULT_MERGED_OUT)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX)
    ap.add_argument("--no-upload", action="store_true",
                    help="skip the aws s3 sync step")
    return ap


# ------------------------------------------------------------- pure helpers
def s3_destination(bucket, merged_out, prefix=DEFAULT_S3_PREFIX):
    """s3:// URI for a merged model dir. Pure, so the path is unit-testable.

    Keyed on the merged dir's basename so the inf2 box's pull path is derivable
    from the model name alone -- no lookup table between the two boxes.
    """
    name = os.path.basename(os.path.normpath(merged_out))
    return f"s3://{bucket}/{prefix}/{name}/"


def find_consolidated_adapter(adapter_dir):
    """Directory containing a peft-loadable adapter, or None. Filesystem-pure.

    Checks the adapter root itself and then any adapter_* subdirectory, because
    optimum-neuron names the default adapter's directory `adapter_default`.
    """
    candidates = [adapter_dir]
    try:
        candidates += sorted(
            os.path.join(adapter_dir, d) for d in os.listdir(adapter_dir)
            if d.startswith(ADAPTER_DIR_PREFIX)
            and os.path.isdir(os.path.join(adapter_dir, d)))
    except OSError:
        return None
    for candidate in candidates:
        for weights in CONSOLIDATED_WEIGHT_NAMES:
            if os.path.isfile(os.path.join(candidate, weights)):
                return candidate
    return None


def find_shard_dirs(adapter_dir):
    """Every `shards/` directory under an adapter dir. Filesystem-pure.

    Returns the shard directories themselves (not their parents) so the caller
    can report how many TP ranks are about to be consolidated.
    """
    found = []
    direct = os.path.join(adapter_dir, SHARDS_DIR_NAME)
    if os.path.isdir(direct):
        found.append(direct)
    found += sorted(
        p for p in glob.glob(os.path.join(adapter_dir, "*", SHARDS_DIR_NAME))
        if os.path.isdir(p))
    return found


def count_shards(shard_dir):
    """Number of per-rank checkpoint files in a shards dir. Filesystem-pure."""
    model_dir = os.path.join(shard_dir, "model")
    root = model_dir if os.path.isdir(model_dir) else shard_dir
    n = 0
    for pattern in ("dp_rank*.tensors", "dp_rank_*.pt", "dp_rank_*"):
        n = max(n, len(glob.glob(os.path.join(root, pattern))))
    return n


def sha256_file(path, chunk_bytes=HASH_CHUNK_BYTES):
    """Streaming SHA-256 -- never loads a 5 GiB shard into RAM to hash it."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def matches_any(name, patterns=ARTIFACT_PATTERNS):
    """True if a basename matches any glob pattern. Pure."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def hash_artifacts(dirpath, patterns=ARTIFACT_PATTERNS):
    """{basename: sha256} for the files that define the model, plus total bytes.

    Top level only: sharded safetensors and the configs all live there, and
    recursing would sweep up caches that differ between machines for reasons
    that have nothing to do with the weights.
    """
    files = {}
    total_bytes = 0
    for name in sorted(os.listdir(dirpath)):
        path = os.path.join(dirpath, name)
        if not os.path.isfile(path) or not matches_any(name, patterns):
            continue
        files[name] = sha256_file(path)
        total_bytes += os.path.getsize(path)
    return files, total_bytes


def dir_size_bytes(dirpath):
    """Total bytes of every file in a directory tree. Filesystem-pure."""
    total = 0
    for root, _dirs, names in os.walk(dirpath):
        for name in names:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                total += os.path.getsize(path)
    return total


def atomic_write_json(path, payload):
    """tmp+rename, same contract as sft_lora.py -- see the note there."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def resolve_versions(packages=("torch", "optimum-neuron", "transformers",
                               "peft", "safetensors", "neuronx-cc")):
    """Installed versions from importlib.metadata; None when absent. Pure."""
    import importlib.metadata
    import platform

    out = {"python": platform.python_version()}
    for name in packages:
        try:
            out[name] = importlib.metadata.version(name)
        except Exception:
            out[name] = None
    return out


# ----------------------------------------------------------------- actions
def consolidate_adapter(adapter_dir, log):
    """Turn TP shards into a peft-loadable adapter. Returns (dir, info).

    No-op when the adapter is already consolidated, so re-running lane 6 after
    a failed upload does not redo this work.
    """
    ready = find_consolidated_adapter(adapter_dir)
    if ready is not None:
        log(f"[1/5] consolidate : already consolidated at {ready} -- skipping")
        return ready, {"performed": False, "shard_dirs": [], "shards": 0}

    shard_dirs = find_shard_dirs(adapter_dir)
    if not shard_dirs:
        raise SystemExit(
            f"FATAL: {adapter_dir} has neither a consolidated adapter "
            f"({'/'.join(CONSOLIDATED_WEIGHT_NAMES)}) nor any "
            f"{SHARDS_DIR_NAME}/ directory. Did sft_lora.py finish?")

    # Point the consolidator at the adapter root: it discovers the adapter_*
    # subdirectories itself and writes the unified adapter into output_dir.
    # Output goes next to the shards so adapter_config.json (already written
    # there by the trainer) sits beside the weights, which is what peft needs.
    output_dir = os.path.dirname(shard_dirs[0])
    n_shards = count_shards(shard_dirs[0])
    log(f"[1/5] consolidate : {n_shards} TP shard(s) in {shard_dirs[0]}")

    from optimum.neuron.models.training import (
        consolidate_model_parallel_checkpoints_to_unified_checkpoint,
    )

    consolidate_model_parallel_checkpoints_to_unified_checkpoint(
        adapter_dir, output_dir, save_format="safetensors")

    ready = find_consolidated_adapter(adapter_dir)
    if ready is None:
        raise SystemExit(
            f"FATAL: consolidation reported success but no "
            f"{CONSOLIDATED_WEIGHT_NAMES[0]} appeared under {adapter_dir}")
    log(f"[1/5] consolidate : -> {ready}")
    return ready, {"performed": True, "shard_dirs": shard_dirs,
                   "shards": n_shards}


def merge(base_id, adapter_dir, merged_out, log):
    """Load base on CPU, apply the adapter, merge, save. Returns timings."""
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"[2/5] load base   : {base_id}  bf16 on CPU")
    t0 = time.perf_counter()
    # device_map is deliberately NOT set: this must stay on CPU. The Neuron
    # device is not a torch device you can .to() a transformers model onto, and
    # accelerate would happily try.
    model = AutoModelForCausalLM.from_pretrained(
        base_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    load_s = time.perf_counter() - t0

    log(f"[3/5] merge       : merge_and_unload()")
    t0 = time.perf_counter()
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    merge_s = time.perf_counter() - t0

    # The PeftModel wrapper and its adapter tensors are dead after
    # merge_and_unload(); dropping them before serialisation is worth a
    # gigabyte or so at exactly the moment the host is tightest.
    gc.collect()

    log(f"[4/5] save        : {merged_out}")
    t0 = time.perf_counter()
    os.makedirs(merged_out, exist_ok=True)
    model.save_pretrained(merged_out, safe_serialization=True)
    # The tokenizer travels with the weights: the inf2 box serves this
    # directory directly, and a merged model without its tokenizer is not a
    # servable artifact.
    AutoTokenizer.from_pretrained(base_id).save_pretrained(merged_out)
    save_s = time.perf_counter() - t0

    return {"load_s": round(load_s, 2), "merge_s": round(merge_s, 2),
            "save_s": round(save_s, 2)}


def upload(merged_out, bucket, prefix, log):
    """aws s3 sync the merged dir. Returns (destination, seconds, ok)."""
    dest = s3_destination(bucket, merged_out, prefix)
    cmd = ["aws", "s3", "sync", merged_out, dest]
    log(f"[5/5] upload      : {dest}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        log(f"      upload FAILED (exit {proc.returncode}) -- merged weights "
            f"are still on disk at {merged_out}; re-run with the same args")
    return dest, round(elapsed, 2), proc.returncode == 0


# -------------------------------------------------------------------- main
def main(argv=None):
    args = build_parser().parse_args(argv)

    def log(*parts):
        print(*parts, flush=True)

    log("")
    log("=" * 78)
    log("  merge LoRA adapter -> servable artifact")
    log("=" * 78)
    log(f"  base       : {args.base}")
    log(f"  adapter    : {args.adapter}")
    log(f"  merged out : {args.merged_out}")
    log(f"  upload     : {'disabled (--no-upload)' if args.no_upload else s3_destination(args.bucket, args.merged_out, args.s3_prefix)}")
    log("=" * 78)
    log("")

    wall0 = time.perf_counter()
    adapter_dir, consolidation = consolidate_adapter(args.adapter, log)
    timings = merge(args.base, adapter_dir, args.merged_out, log)

    files, hashed_bytes = hash_artifacts(args.merged_out)
    total_bytes = dir_size_bytes(args.merged_out)
    total_gib = total_bytes / float(1 << 30)

    dest, upload_s, upload_ok = (None, None, None)
    if not args.no_upload:
        dest, upload_s, upload_ok = upload(
            args.merged_out, args.bucket, args.s3_prefix, log)

    payload = {
        "lane": "merge",
        "base": args.base,
        "adapter": args.adapter,
        "adapter_consolidated_dir": adapter_dir,
        "consolidation": consolidation,
        "merged_out": args.merged_out,
        "files": files,
        "files_note": (
            "sha256 of every *.safetensors plus config.json and tokenizer.json "
            "in the merged directory. The inf2 box re-hashes the same set with "
            "shared/verify_models.sh after pulling from S3; matching digests "
            "are what make 'the served weights are the trained weights' a "
            "check rather than an assumption."),
        "hashed_bytes": hashed_bytes,
        "total_bytes": total_bytes,
        "total_gib": round(total_gib, 3),
        "timings_s": timings,
        "upload_dest": dest,
        "upload_s": upload_s,
        "upload_ok": upload_ok,
        "wall_s": round(time.perf_counter() - wall0, 2),
        "versions": resolve_versions(),
        "host_ram_note": (
            "Heaviest host-RAM step in the trn1 suite: base model in bf16 plus "
            "the merged state dict during safetensors serialisation, against "
            "32 GiB on trn1.2xlarge. See the module docstring for the "
            "local-merge fallback if this OOMs."),
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = atomic_write_json(args.out, payload)

    log("")
    log(f"SUMMARY base={args.base} merged_out={args.merged_out} "
        f"files={len(files)} total_gib={total_gib:.2f} "
        f"consolidated={consolidation['performed']} "
        f"merge_s={timings['merge_s']:.1f} "
        f"upload_s={'null' if upload_s is None else f'{upload_s:.1f}'} "
        f"upload_ok={upload_ok}")
    log(f"wrote {out_path}")

    # A failed upload is a lane failure: the inf2 box has nothing to pull.
    return 0 if (args.no_upload or upload_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
