#!/usr/bin/env bash
# The best-effort lane: Llama 3.1 8B LoRA run FULL LENGTH at the configuration
# Trainium2 actually wants, instead of the one Trainium1's memory forced.
#
# WHY THIS EXISTS
# ---------------
# Lane 4 is deliberately run at trn1's settings -- micro-batch 1, seq 2048, all
# hyperparameters byte-identical -- because that is the only way to attribute a
# difference to silicon. It measured:
#
#     trn1  2951.8 tok/s   trn2  3532.8 tok/s   = 1.20x
#     against 3.51x more peak FLOP/s -> only 34% of the theoretical gain
#
# That is not a verdict on the chip, it is a verdict on the CONFIGURATION.
# micro-batch 1 was sized against trn1's 16 GiB per core; Trainium2 has 24 GiB
# and 3.5x the compute, so the same shape leaves it starved. Reporting 1.20x
# alone would be as misleading as reporting a tuned number against an untuned
# baseline.
#
# So the report needs BOTH:
#   like-for-like   lane 4                 "same workload, different chip"
#   best-effort     THIS lane              "what the chip can actually do"
# Neither replaces the other, and this file never touches lane 4's artifacts.
#
# THE CONFIG IS MEASURED, NOT GUESSED
# -----------------------------------
# By the time this runs, the efficiency lanes have already swept the levers:
#   E1  seq 4096                    (on trn1 this alone moved MFU 68.3 -> 82.7)
#   E2  gradient checkpointing off  (24 GiB/core may make recompute unnecessary)
#   E3  micro-batch 2, 4            (the underfeeding lever)
#   frontier  micro-batch 8, and eff_combined (all three together)
# This script reads whichever of those PASSED, picks the highest measured
# tokens/s, and runs THAT config to full length. Picking by measurement rather
# than by intuition is the difference between a result and an anecdote.
#
# FAIRNESS: same model, same dataset, same 3 epochs, same seed. Only the
# execution shape changes, so total tokens processed is ~equal and wall-clock
# and tokens/s remain directly comparable to lane 4.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"
OUT="$BENCH_DIR/trn2/results/maxutil"
EXTRAS="$BENCH_DIR/trn2/results/extras"
FRONTIER="$BENCH_DIR/trn2/results/frontier"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
SFT="$BENCH_DIR/shared/train/sft_lora.py"
MAXUTIL_DEADLINE_UTC="${MAXUTIL_DEADLINE_UTC:-09:30}"

export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
export NP_DEVICE=trn2
export NP_REGION="${NP_REGION:-sa-east-1}"
export NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache-v3}"

mkdir -p "$OUT"
have() { [ "${FORCE:-0}" != "1" ] && [ -s "$1" ]; }
step() { echo; echo "############ MAXUTIL: $* ############"; echo; }

[ -s "$EXTRAS/tp_probe.json" ] || { echo "FATAL: tp_probe.json missing"; exit 3; }
read -r LNC NPROC TP <<<"$("$PY" -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(d["lnc"], d["nproc"], d["tp"])' "$EXTRAS/tp_probe.json")"
export NEURON_LOGICAL_NC_CONFIG="$LNC"
export NP_TELEMETRY_CORES="$NPROC"

seconds_left() {
  local now target
  now=$(date -u +%s)
  target=$(date -u -d "today ${MAXUTIL_DEADLINE_UTC}" +%s 2>/dev/null) || \
    target=$(date -u -j -f "%Y-%m-%d %H:%M" "$(date -u +%Y-%m-%d) ${MAXUTIL_DEADLINE_UTC}" +%s)
  [ "$target" -le "$now" ] && target=$((target + 86400))
  echo $((target - now))
}

# ---- Pick the winning configuration from the efficiency sweeps --------------
# Only lanes that actually produced a result are considered; a failure receipt
# is not a candidate. Ties and near-ties go to the SIMPLER config (fewer levers
# changed), because an unexplained 1% is not worth an extra confound.
CHOICE=$("$PY" - "$EXTRAS" "$FRONTIER" "$BENCH_DIR/trn2/results/train/llama31_lora.json" <<'EOF'
import glob, json, os, sys
extras, frontier, baseline_path = sys.argv[1], sys.argv[2], sys.argv[3]

cands = []
for d in (extras, frontier):
    for path in glob.glob(os.path.join(d, "*.json")):
        name = os.path.basename(path)
        if not name.startswith(("eff_", "ctx_")):
            continue
        try:
            r = json.load(open(path))
        except Exception:
            continue
        tps = r.get("tokens_per_s")
        cfg = r.get("config") or {}
        if not tps:
            continue
        # Only levers that keep the lane comparable: sequence length, micro
        # batch and recompute. A different MODEL or dataset is a different
        # experiment, not a tuning of this one.
        cands.append({
            "tag": r.get("tag") or name[:-5],
            "tokens_per_s": tps,
            "seq_len": cfg.get("seq_len"),
            "micro_batch": cfg.get("micro_batch"),
            "grad_ckpt": cfg.get("gradient_checkpointing"),
            "levers": sum([cfg.get("seq_len") != 2048,
                           (cfg.get("micro_batch") or 1) != 1,
                           cfg.get("gradient_checkpointing") is False]),
        })

base = {}
try:
    b = json.load(open(baseline_path))
    base = {"tag": "llama31_lora (lane 4)", "tokens_per_s": b.get("tokens_per_s"),
            "seq_len": 2048, "micro_batch": 1, "grad_ckpt": True, "levers": 0}
except Exception:
    pass
if base.get("tokens_per_s"):
    cands.append(base)

if not cands:
    print("NONE"); sys.exit(0)

best = max(c["tokens_per_s"] for c in cands)
# within 2% of best -> prefer fewer levers changed
near = [c for c in cands if c["tokens_per_s"] >= 0.98 * best]
win = sorted(near, key=lambda c: (c["levers"], -c["tokens_per_s"]))[0]
json.dump({"winner": win, "candidates": sorted(
    cands, key=lambda c: -c["tokens_per_s"])}, open(
    os.path.join(os.path.dirname(extras), "maxutil", "selection.json"), "w"), indent=1)
print(f'{win["seq_len"]} {win["micro_batch"]} {win["grad_ckpt"]} {win["tag"]}')
EOF
)

if [ "$CHOICE" = "NONE" ] || [ -z "$CHOICE" ]; then
  echo "FATAL: no efficiency lane produced a result -- nothing to select from"
  exit 4
fi
read -r SEQ MB GC WINTAG <<<"$CHOICE"
echo "selected config from measurement: seq_len=$SEQ micro_batch=$MB grad_ckpt=$GC (from $WINTAG)"

ARGS=(--device-profile trn2 --nproc-per-node "$NPROC" --tensor-parallel-size "$TP"
      --model meta-llama/Llama-3.1-8B-Instruct --seq-len "$SEQ" --micro-batch "$MB")
[ "$GC" = "False" ] && ARGS+=(--no-gradient-checkpointing)

# ---- Full-length run at the winning config ---------------------------------
# No --max-steps: 3 epochs, exactly like lane 4, so wall clock and tokens/s are
# comparable rather than extrapolated from a 100-step probe.
TAG=llama31_lora_maxutil
if have "$OUT/$TAG.json" || have "$OUT/$TAG.failure.json"; then
  echo "skip $TAG (recorded)"
else
  NEED=150
  if [ "$(seconds_left)" -lt $((NEED * 60)) ]; then
    echo "deadline guard: under ${NEED} min to ${MAXUTIL_DEADLINE_UTC}Z -- refusing to start a full lane"
    exit 0
  fi
  step "$TAG -- full 3 epochs at seq=$SEQ mb=$MB gc=$GC"
  "$PY" "$TELEM" --out "$OUT/$TAG.telemetry.csv" -- \
    "$PY" "$SFT" --tag "$TAG" "${ARGS[@]}" --out "$OUT/$TAG.json" \
    > "$OUT/$TAG.log" 2>&1 || {
      REASON=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,120}|OutOfMemory" "$OUT/$TAG.log" | head -1 | sed "s/\"/'/g")
      [ -n "$REASON" ] || REASON=$(tail -c 300 "$OUT/$TAG.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"tag":"%s","status":"failed","seq_len":%s,"micro_batch":%s,"reason":"%s","captured":"%s"}\n' \
        "$TAG" "$SEQ" "$MB" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$TAG.failure.json"
      echo "  $TAG FAILED (receipt)"; }
  bash shared/bin/push_results.sh trn2 >/dev/null 2>&1 || echo "  push FAILED"
fi

step "MAXUTIL COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || echo "cache push FAILED"
bash shared/bin/push_results.sh trn2 || echo "push FAILED"
