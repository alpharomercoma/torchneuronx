#!/usr/bin/env bash
# CLOSE THE TRAINIUM1 GAPS -- the three lanes trn2 has and trn1 does not.
#
#   bash extras/run_gapfill_trn1.sh
#
# WHY THIS EXISTS
# ---------------
# A coverage audit of the 68 lanes in this study found 18 that exist only on
# trn2. Most are structurally impossible on trn1 and stay that way:
#
#   ctx 9216/10240/12288/16384   trn1 already OOMs at 8192 (NCC_EOOM002,
#                                18.12GB > 16.00GB) -- higher rungs are
#                                unreachable, not unattempted.
#   isolation synth/real 8192    same wall.
#   tp_probe, tp_preflight       n/a: Trainium1 has no logical-NeuronCore
#                                config to probe.
#
# Three are genuinely just missing, and this driver runs them. Each one turns a
# CURRENTLY ONE-SIDED comparison into a two-sided one, which is the only reason
# any of them is worth the box time.
#
# LANE ORDER IS CHEAPEST-FIRST
# ----------------------------
# Standard practice in this repo: a broken environment should surface in the
# 10-minute lane, not 100 minutes into the expensive one. eff_combined is
# expected to fail fast, residency is ~30 min, maxutil is ~100 min.
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
NP_TAG=GAPFILL-TRN1
cd "$BENCH_DIR"

BOX=trn1
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
# The libneuronpjrt-path gotcha: without the venv FIRST on PATH the training
# script dies with FileNotFoundError on 'libneuronpjrt-path'. That cost seven
# lanes in Phase 2 and one whisper receipt that is still on disk.
export PATH="$NP_VENV/bin:$PATH"
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
SFT="$BENCH_DIR/shared/train/sft_lora.py"

export NP_DEVICE=trn1
export NP_REGION="${NP_REGION:-us-west-2}"
export NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache}"
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
# Neuron caches FAILED compiles. Without this, the first OOM in a lane group is
# replayed verbatim for every later lane, and N rungs look like N measurements
# when only one of them is. This trap already invalidated one trn1 batch ladder.
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:-} --retry_failed_compilation"

OUT_MAX="$BENCH_DIR/$BOX/results/maxutil"
OUT_FRO="$BENCH_DIR/$BOX/results/frontier"
OUT_OPP="$BENCH_DIR/$BOX/results/opportunistic"
mkdir -p "$OUT_MAX" "$OUT_FRO" "$OUT_OPP"

bash shared/bin/hf_login.sh >/dev/null 2>&1 || np_log "WARN: hf_login failed"

push() { bash shared/bin/push_results.sh "$BOX" >/dev/null 2>&1 || np_log "  push FAILED"; }

# lane <outdir> <tag> <args...>   -- run, record, never abort the driver
lane() {
  local dir="$1" tag="$2"; shift 2
  np_recorded "$dir/$tag" && { np_log "skip $tag (recorded)"; return 0; }
  np_wait_for_chip || return 0
  np_step "$tag"
  "$PY" "$TELEM" --out "$dir/$tag.telemetry.csv" -- \
    "$PY" "$SFT" --tag "$tag" --out "$dir/$tag.json" "$@" \
    > "$dir/$tag.log" 2>&1 \
    || np_receipt "$dir/$tag.failure.json" "$tag" "$BOX" "$dir/$tag.log"
  push
}

# =============================================================================
# 1. eff_combined -- every lever at once
# =============================================================================
# trn2 has this receipt; trn1 does not, so §17's "combined best-effort" row is
# blank for one chip. The PREDICTION, written before the run: this fails on
# device memory. micro-batch 2 alone failed (NCC_EXSP001) and recompute-off
# alone failed (NCC_EOOM001, 22.19GB > 16.00GB); applying both cannot need less
# memory than either. A confirmed OOM here is the intended outcome -- it makes
# "trn1 has no headroom for any combination of levers" a measurement instead of
# an extrapolation from two separate lanes.
np_log "=== 1/3 eff_combined (expected: device OOM) ==="
lane "$OUT_OPP" eff_combined \
  --model meta-llama/Llama-3.1-8B-Instruct --seq-len 4096 --micro-batch 2 \
  --no-gradient-checkpointing --max-steps 100

# =============================================================================
# 2. residency -- can one Trainium1 chip host two tenants?
# =============================================================================
# ############################ SHAPE CAVEAT ###############################
# This is NOT shape-matched to the trn2 residency lanes and must never be
# reported as if it were.
#
#   trn2:  4 logical cores -> two tenants of 2 cores each (TP=2 per tenant)
#   trn1:  2 physical cores -> two tenants of 1 core each  (TP=1 per tenant)
#
# The tenants here are half the width of trn2's, so the absolute numbers are
# not comparable across chips. What IS comparable is the RATIO each chip
# reports -- paired throughput against that same tenant's solo throughput --
# because each ratio is computed entirely within one chip. That ratio is the
# multi-tenancy question; the absolute tok/s is not.
#
# TP=1 may simply not work on this stack. If it does not, the receipt is the
# result: it would mean the narrowest viable job on Trainium1 consumes the
# whole chip, so the chip cannot be shared at all -- a sharper finding than a
# slowdown percentage, and one the report can state plainly.
# #########################################################################
#
# SOLO BASELINES ARE PER-CORE, NOT SHARED. §27.4 recorded the mistake of
# baselining both halves against one solo run: the two core groups are not
# interchangeable, and a b-side pair number needs a b-side solo number.
np_log "=== 2/3 residency (two 1-core tenants; see SHAPE CAVEAT in source) ==="
RES_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
RES_STEPS=50

if np_recorded "$OUT_FRO/residency_pair_a" && np_recorded "$OUT_FRO/residency_solo_a"; then
  np_log "skip residency (recorded)"
else
  np_wait_for_chip
  # --- concurrent pair: both tenants at once, one core each ---
  np_step "residency_pair_a + residency_pair_b (concurrent)"
  NP_MASTER_PORT=29511 NEURON_RT_VISIBLE_CORES=0 NP_TELEMETRY_CORES=1 \
    "$PY" "$TELEM" --out "$OUT_FRO/residency_pair_a.telemetry.csv" -- \
    "$PY" "$SFT" --tag residency_pair_a --device-profile trn1 \
      --nproc-per-node 1 --tensor-parallel-size 1 \
      --model "$RES_MODEL" --max-steps "$RES_STEPS" \
      --out "$OUT_FRO/residency_pair_a.json" \
      > "$OUT_FRO/residency_pair_a.log" 2>&1 &
  PID_A=$!
  NP_MASTER_PORT=29512 NEURON_RT_VISIBLE_CORES=1 NP_TELEMETRY_CORES=1 \
    "$PY" "$TELEM" --out "$OUT_FRO/residency_pair_b.telemetry.csv" -- \
    "$PY" "$SFT" --tag residency_pair_b --device-profile trn1 \
      --nproc-per-node 1 --tensor-parallel-size 1 \
      --model "$RES_MODEL" --max-steps "$RES_STEPS" \
      --out "$OUT_FRO/residency_pair_b.json" \
      > "$OUT_FRO/residency_pair_b.log" 2>&1 &
  PID_B=$!
  wait $PID_A || np_receipt "$OUT_FRO/residency_pair_a.failure.json" \
    residency_pair_a "$BOX" "$OUT_FRO/residency_pair_a.log"
  wait $PID_B || np_receipt "$OUT_FRO/residency_pair_b.failure.json" \
    residency_pair_b "$BOX" "$OUT_FRO/residency_pair_b.log"
  push

  # --- solo baselines: same tenant, same core, nothing else resident ---
  for HALF in a:0:29513 b:1:29514; do
    IFS=: read -r SIDE CORE PORT <<<"$HALF"
    np_recorded "$OUT_FRO/residency_solo_$SIDE" && continue
    np_wait_for_chip
    np_step "residency_solo_$SIDE (core $CORE, alone)"
    NP_MASTER_PORT="$PORT" NEURON_RT_VISIBLE_CORES="$CORE" NP_TELEMETRY_CORES=1 \
      "$PY" "$TELEM" --out "$OUT_FRO/residency_solo_$SIDE.telemetry.csv" -- \
      "$PY" "$SFT" --tag "residency_solo_$SIDE" --device-profile trn1 \
        --nproc-per-node 1 --tensor-parallel-size 1 \
        --model "$RES_MODEL" --max-steps "$RES_STEPS" \
        --out "$OUT_FRO/residency_solo_$SIDE.json" \
        > "$OUT_FRO/residency_solo_$SIDE.log" 2>&1 \
      || np_receipt "$OUT_FRO/residency_solo_$SIDE.failure.json" \
           "residency_solo_$SIDE" "$BOX" "$OUT_FRO/residency_solo_$SIDE.log"
    push
  done
fi

# =============================================================================
# 3. maxutil -- the best-effort number, selected by measurement
# =============================================================================
# trn2 has a best-effort lane (seq 8192, 8336.6 tok/s, 61.0% MFU, full 3
# epochs). trn1 does not, so the report currently compares trn2's BEST config
# against trn1's LIKE-FOR-LIKE config -- which flatters trn2 by exactly the
# amount trn1 would have gained from the same treatment. This closes that.
#
# The config is chosen the same way it was on trn2: read every eff_*/ctx_* lane
# that actually PRODUCED a result, take the highest measured tokens/s, and
# break near-ties (within 2%) toward the config that changed FEWER levers. On
# trn1 the passing candidates are ctx_4096 (3575.0) and eff_seq4096 (3572.3)
# against the lane-4 baseline (2951.8), so this should select seq 4096 -- but
# it is selected by the file contents at run time, never hardcoded here.
np_log "=== 3/3 maxutil (full 3 epochs at the measured-best config) ==="
if np_recorded "$OUT_MAX/llama31_lora_maxutil"; then
  np_log "skip maxutil (recorded)"
else
  CHOICE=$("$PY" - "$BENCH_DIR/$BOX/results/extras" "$BENCH_DIR/$BOX/results/frontier" \
                  "$BENCH_DIR/$BOX/results/train/llama31_lora.json" "$OUT_MAX" <<'PYEOF'
import glob, json, os, sys
extras, frontier, baseline_path, outdir = sys.argv[1:5]
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
        tps, cfg = r.get("tokens_per_s"), (r.get("config") or {})
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
try:
    b = json.load(open(baseline_path))
    if b.get("tokens_per_s"):
        cands.append({"tag": "llama31_lora (lane 4)", "tokens_per_s": b["tokens_per_s"],
                      "seq_len": 2048, "micro_batch": 1, "grad_ckpt": True, "levers": 0})
except Exception:
    pass
if not cands:
    print("NONE"); sys.exit(0)
best = max(c["tokens_per_s"] for c in cands)
near = [c for c in cands if c["tokens_per_s"] >= 0.98 * best]
win = sorted(near, key=lambda c: (c["levers"], -c["tokens_per_s"]))[0]
json.dump({"winner": win, "candidates": sorted(cands, key=lambda c: -c["tokens_per_s"])},
          open(os.path.join(outdir, "selection.json"), "w"), indent=1)
print(f'{win["seq_len"]} {win["micro_batch"]} {win["grad_ckpt"]} {win["tag"]}')
PYEOF
)
  if [ "$CHOICE" = "NONE" ] || [ -z "$CHOICE" ]; then
    np_die "no efficiency lane produced a result -- nothing to select from" 4
  fi
  read -r SEQ MB GC WINTAG <<<"$CHOICE"
  np_log "selected by measurement: seq_len=$SEQ micro_batch=$MB grad_ckpt=$GC (from $WINTAG)"

  ARGS=(--model meta-llama/Llama-3.1-8B-Instruct --seq-len "$SEQ" --micro-batch "$MB")
  [ "$GC" = "False" ] && ARGS+=(--no-gradient-checkpointing)
  # No --max-steps: 3 full epochs, exactly like lane 4, so wall clock and
  # tokens/s are comparable rather than extrapolated from a short probe.
  lane "$OUT_MAX" llama31_lora_maxutil "${ARGS[@]}"
fi

np_step "GAPFILL COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || np_log "cache push FAILED"
bash shared/bin/push_results.sh "$BOX" || np_log "push FAILED"
