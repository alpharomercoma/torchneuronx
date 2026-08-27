#!/usr/bin/env bash
# DATALOADER ISOLATION -- runs on BOTH chips, byte-identical recipe.
#
#   BOX=trn1 bash extras/run_dataloader_isolation.sh
#   BOX=trn2 bash extras/run_dataloader_isolation.sh
#
# THE QUESTION
# ------------
# Trainium2 beats Trainium1 by 1.20x at seq 2048 but 1.92x at seq 4096. The
# headline number is the weaker one, and if it is an artifact we owe the reader
# an explanation rather than a shrug.
#
# The leading hypothesis: at short context the step is not device-bound. Host
# work -- tokenise, pack, collate, copy to device -- costs roughly the same
# wall-clock on both boxes, because both have ordinary CPUs. A fixed host cost
# is a LARGER FRACTION of trn2's shorter step than of trn1's longer one, so it
# compresses the observed ratio. Double the sequence length and the device work
# doubles while the host cost does not, so the ratio should open up. That is
# exactly what 1.20x -> 1.92x looks like.
#
# THE TEST
# --------
# --synthetic-data replaces the dataset with pre-tokenised random token IDs of
# exactly seq_len: no Hub, no tokeniser, no packing, no formatter. The compiled
# graph sees identical shapes, so the NEFF cache hits and nothing recompiles --
# the ONLY thing removed is host-side data preparation.
#
# THE DISCRIMINATING COMPARISON is not synthetic-vs-real at one shape. It is how
# the synthetic UPLIFT changes with sequence length:
#
#   hypothesis true  -> uplift is LARGE at 2048 and SMALL at 4096
#                       (fixed host cost, shrinking as a share of the step)
#   hypothesis false -> uplift is about the same at both, or ~zero everywhere
#                       (the chip really was saturated; 1.20x is the honest
#                        device-level answer at that shape)
#
# So each shape needs its own real-data control at the SAME step count. Four
# paired lanes, not two. Comparing a 40-step synthetic run against the
# published 645-step lane would confound the uplift with warmup amortisation.
#
# EITHER OUTCOME IS PUBLISHABLE. A null result retires a plausible objection to
# the headline; a positive one explains it. This lane is cheap on trn1 (free,
# on credits) so it runs there FIRST -- if the synthetic path does not work on
# this stack at all, we learn that without spending the non-refundable trn2
# window on it.
set -uo pipefail

BOX="${BOX:?set BOX=trn1 or BOX=trn2}"
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"
OUT="$BENCH_DIR/$BOX/results/isolation"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"      # libneuronpjrt-path
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
SFT="$BENCH_DIR/shared/train/sft_lora.py"

# Matched step count for every paired lane. 40 optimizer steps is long enough
# that the steady-state median is stable (the 2.4% noise floor was measured
# over runs of this order) and short enough that four lanes fit in ~25 min.
STEPS="${STEPS:-40}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
# One synthetic row is consumed per micro-batch; 4x the step count leaves
# headroom for gradient accumulation without the sampler wrapping.
ROWS=$(( STEPS * 4 ))

export NP_DEVICE="$BOX"
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
if [ "$BOX" = "trn2" ]; then
  export NP_REGION="${NP_REGION:-sa-east-1}"
  export NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache-v3}"
else
  export NP_REGION="${NP_REGION:-us-west-2}"
  export NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache}"
fi

mkdir -p "$OUT"
# Helpers come from ONE place now. `have`, `step`, the receipt writer and the
# chip/port guards used to be copy-pasted into ~16 drivers, which is how the
# fleet acquired five different answers to "is this lane already done?".
NP_TAG="ISOLATION[$BOX]"
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

bash shared/bin/hf_login.sh >/dev/null 2>&1 || echo "WARN: hf_login failed"

# Parallelism comes from the same place as every other lane on this box, so the
# isolation result is directly comparable to the published numbers. Symmetry is
# in the RECIPE; the sharding is a property of the chip.
EXTRA=()
if [ "$BOX" = "trn2" ]; then
  P="$BENCH_DIR/trn2/results/extras/tp_probe.json"
  [ -s "$P" ] || { echo "FATAL: tp_probe.json missing"; exit 3; }
  read -r LNC NPROC TP <<<"$("$PY" -c '
import json,sys
d=json.load(open(sys.argv[1])); print(d["lnc"], d["nproc"], d["tp"])' "$P")"
  export NEURON_LOGICAL_NC_CONFIG="$LNC" NP_TELEMETRY_CORES="$NPROC"
  EXTRA=(--device-profile trn2 --nproc-per-node "$NPROC" --tensor-parallel-size "$TP")
fi

lane() {   # lane <tag> <args...>
  local tag="$1"; shift
  np_recorded "$OUT/$tag" && { np_log "skip $tag (recorded)"; return 0; }
  np_step "$tag"
  "$PY" "$TELEM" --out "$OUT/$tag.telemetry.csv" -- \
    "$PY" "$SFT" --tag "$tag" "${EXTRA[@]}" --out "$OUT/$tag.json" "$@" \
    > "$OUT/$tag.log" 2>&1 || {
      np_receipt "$OUT/$tag.failure.json" "$tag" "$BOX" "$OUT/$tag.log"; return 1; }
  bash shared/bin/push_results.sh "$BOX" >/dev/null 2>&1 || echo "  push FAILED"
  return 0
}

# ---- 0. Does the pre-tokenised path work on this stack at all? --------------
# TRL accepting a dataset that is already input_ids/attention_mask/labels, with
# no formatting_func, is unverified here and doubly so on NeuronCore-v3. A
# TinyLlama probe costs ~2 minutes; discovering it mid-8B costs an hour.
lane synth_smoke --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --synthetic-data 80 --max-steps 20 || true

if ! np_have "$OUT/synth_smoke.json"; then
  echo "synthetic path does not work on this stack -- receipt stands as the finding"
  bash shared/bin/push_results.sh "$BOX" || true
  exit 0
fi

# ---- 1..4 The four paired lanes --------------------------------------------
# Order matters: run BOTH shapes' real controls and synthetics, and run the
# control immediately before its synthetic twin so any slow drift in machine
# state lands on both sides of the comparison rather than on one.
for SEQ in 2048 4096; do
  lane "real_ctl_${SEQ}"  --model "$MODEL" --seq-len "$SEQ" --max-steps "$STEPS" || true
  lane "synth_${SEQ}"     --model "$MODEL" --seq-len "$SEQ" --max-steps "$STEPS" \
       --synthetic-data "$ROWS" || true
done

# ---- Summary ---------------------------------------------------------------
"$PY" - "$OUT" "$BOX" <<'PYEOF' || echo "summary FAILED (lanes are still on disk)"
import json, pathlib, sys
out, box = pathlib.Path(sys.argv[1]), sys.argv[2]
rows, summary = [], {"box": box, "shapes": {}}
for seq in (2048, 4096):
    def tok(tag):
        p = out / f"{tag}.json"
        if not p.is_file():
            return None
        return (json.loads(p.read_text()) or {}).get("tokens_per_s")
    r, s = tok(f"real_ctl_{seq}"), tok(f"synth_{seq}")
    if r and s:
        uplift = s / r
        summary["shapes"][str(seq)] = {"real_tokens_per_s": r,
                                       "synthetic_tokens_per_s": s,
                                       "uplift": round(uplift, 4)}
        rows.append(f"  seq {seq}: real {r:.1f} -> synthetic {s:.1f} tok/s  "
                    f"uplift {uplift:.3f}x")
    else:
        rows.append(f"  seq {seq}: incomplete (real={r}, synth={s})")
u = summary["shapes"]
if "2048" in u and "4096" in u:
    a, b = u["2048"]["uplift"], u["4096"]["uplift"]
    summary["uplift_shrinks_with_seq_len"] = bool(a > b)
    # THE THRESHOLD MUST EXCEED THE STUDY'S OWN RESOLUTION. The measured noise
    # floor is 2.4% -- both across three seeds on one box and across two
    # physical Trainium2 chips (see REPORT 20). A 2% threshold would declare an
    # effect smaller than the smallest difference this methodology can resolve,
    # which is how a noise reading becomes a finding. Anything inside the floor
    # is reported as inconclusive, not as a null and not as an effect.
    NOISE = 0.024
    biggest = max(abs(a - 1.0), abs(b - 1.0))
    if biggest <= NOISE:
        summary["interpretation"] = (
            f"no host effect resolvable: the largest uplift ({biggest:+.1%}) is "
            f"inside the {NOISE:.1%} noise floor. Bounds host dataloader cost "
            "BELOW the floor; does not prove it is zero.")
    elif a > b + NOISE:
        summary["interpretation"] = (
            "host dataloader cost is a real and shape-dependent share of the step: "
            f"uplift {a:.3f}x at 2048 vs {b:.3f}x at 4096, a gap wider than the "
            f"{NOISE:.1%} noise floor")
    else:
        summary["interpretation"] = (
            f"uplift exceeds the noise floor ({biggest:+.1%}) but does NOT shrink "
            "with sequence length, so it is not the fixed-host-cost signature")
    summary["noise_floor_used"] = NOISE
    rows.append(f"  uplift 2048 {a:.3f}x vs 4096 {b:.3f}x -> {summary['interpretation']}")
(out / "isolation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("\n".join(rows))
PYEOF

np_step "DATALOADER ISOLATION COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || echo "cache push FAILED"
bash shared/bin/push_results.sh "$BOX" || echo "push FAILED"
