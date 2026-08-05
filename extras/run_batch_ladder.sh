#!/usr/bin/env bash
# MICRO-BATCH LADDER at seq 4096 -- runs on BOTH chips, byte-identical recipe.
#
#   BOX=trn1 bash extras/run_batch_ladder.sh
#   BOX=trn2 bash extras/run_batch_ladder.sh
#
# WHY THIS LANE EXISTS, AND WHY IT EXISTS *NOW*
# ---------------------------------------------
# §21 killed the host-dataloader explanation for why Trainium2 is only 1.20x
# faster at seq 2048 but 1.92x at 4096. With the host ruled out, the MFU column
# gives the surviving answer: at seq 2048 and micro-batch 1 there is simply not
# enough work in a step to fill a chip with 3.5x the FLOPs and 3x the HBM.
# trn2 sits at 25.9% MFU while trn1 sits at 75.2% on the identical shape.
# Trainium2 is not being slowed down. It is being starved.
#
# That claim is currently an INFERENCE from throughput. This lane tests it
# directly, by the one lever we have not pulled: bring more work per step.
#
# The context ladder already varies work per step via sequence length, but
# sequence length also changes the attention term's share of the FLOPs and the
# activation memory profile. Micro-batch is the cleaner instrument: it scales
# work per step almost linearly while leaving the per-token compute mix alone.
#
# THE PREDICTION, WRITTEN DOWN BEFORE THE RUN
# -------------------------------------------
#   If trn2 is starved: its tokens/s and MFU should keep RISING with
#   micro-batch, well past the point where trn1 has stopped improving.
#   If trn1 is already saturated at 91.3% MFU (measured, §21 control): it should
#   gain little and then hit a device-memory wall -- 16 GiB per logical core
#   against trn2's 24.
#
# A ladder where BOTH chips stop improving at the same rung would falsify the
# occupancy story and send us back to the profiler. Writing the prediction down
# first is what makes this a test rather than an illustration.
#
# WHY IT IS SAFE TO RUN OUT OF ORDER
# ----------------------------------
# Each rung is an independent 30-step probe with its own receipt. An OOM at a
# high rung is a RESULT (it is the memory wall, and where it sits differs
# between the chips), not a failure of the lane -- so the ladder always climbs
# upward and never stops early on a device OOM.
set -uo pipefail

BOX="${BOX:?set BOX=trn1 or BOX=trn2}"
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"
OUT="$BENCH_DIR/$BOX/results/batch_ladder"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
SFT="$BENCH_DIR/shared/train/sft_lora.py"

SEQ="${SEQ:-4096}"
STEPS="${STEPS:-30}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
# 1 is the published configuration and the ladder's own baseline; it is
# re-measured here rather than borrowed from §15 so every rung comes from one
# uninterrupted session on one machine state.
RUNGS="${RUNGS:-1 2 4 8}"

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
have() { [ "${FORCE:-0}" != "1" ] && [ -s "$1" ]; }
step() { echo; echo "############ BATCH-LADDER[$BOX]: $* ############"; echo; }

bash shared/bin/hf_login.sh >/dev/null 2>&1 || echo "WARN: hf_login failed"

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
  { have "$OUT/$tag.json" || have "$OUT/$tag.failure.json"; } && { echo "skip $tag"; return 0; }
  step "$tag"
  "$PY" "$TELEM" --out "$OUT/$tag.telemetry.csv" -- \
    "$PY" "$SFT" --tag "$tag" "${EXTRA[@]}" --out "$OUT/$tag.json" "$@" \
    > "$OUT/$tag.log" 2>&1 || {
      # NCC_EOOM002 is the device out-of-memory code. It is the EXPECTED
      # terminal rung on at least one of these chips, so the receipt records it
      # as a measurement of where the wall is, not as an incident.
      REASON=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,140}" "$OUT/$tag.log" | head -1 | sed "s/\"/'/g")
      [ -n "$REASON" ] || REASON=$(tail -c 300 "$OUT/$tag.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"tag":"%s","box":"%s","status":"failed","reason":"%s","captured":"%s"}\n' \
        "$tag" "$BOX" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$tag.failure.json"
      echo "  $tag FAILED (receipt) -- ladder continues, an OOM rung IS the result"; return 0; }
  bash shared/bin/push_results.sh "$BOX" >/dev/null 2>&1 || echo "  push FAILED"
  return 0
}

# Global batch is held CONSTANT across rungs by halving grad-accum as
# micro-batch doubles. Otherwise a rung would change two things at once -- work
# per step AND the optimisation trajectory -- and the throughput comparison
# would be confounded with a different effective batch size.
ACCUM_AT_MB1="${ACCUM_AT_MB1:-8}"
for MB in $RUNGS; do
  ACC=$(( ACCUM_AT_MB1 / MB ))
  [ "$ACC" -lt 1 ] && ACC=1
  lane "mb${MB}_seq${SEQ}" --model "$MODEL" --seq-len "$SEQ" \
    --micro-batch "$MB" --grad-accum "$ACC" --max-steps "$STEPS"
done

# ---- Summary ---------------------------------------------------------------
"$PY" - "$OUT" "$BOX" "$SEQ" <<'PYEOF' || echo "summary FAILED (rungs are on disk)"
import json, pathlib, sys
out, box, seq = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rungs, prev = [], None
for mb in (1, 2, 4, 8):
    ok = out / f"mb{mb}_seq{seq}.json"
    bad = out / f"mb{mb}_seq{seq}.failure.json"
    if ok.is_file():
        d = json.loads(ok.read_text())
        e = {"micro_batch": mb, "tokens_per_s": d.get("tokens_per_s"),
             "mfu_pct": d.get("mfu_pct"), "median_step_ms": d.get("median_step_ms")}
        if prev and prev.get("tokens_per_s") and e["tokens_per_s"]:
            e["gain_vs_prev_rung"] = round(e["tokens_per_s"] / prev["tokens_per_s"], 4)
        rungs.append(e); prev = e
    elif bad.is_file():
        r = json.loads(bad.read_text()).get("reason", "")
        rungs.append({"micro_batch": mb, "status": "failed",
                      "wall": "device OOM" if "EOOM" in r else "failed",
                      "reason": r[:160]})
        break
summary = {"box": box, "seq_len": int(seq), "rungs": rungs}
done = [r for r in rungs if r.get("tokens_per_s")]
if done:
    best = max(done, key=lambda r: r["tokens_per_s"])
    summary["best_micro_batch"] = best["micro_batch"]
    summary["best_tokens_per_s"] = best["tokens_per_s"]
    summary["uplift_over_mb1"] = round(best["tokens_per_s"] / done[0]["tokens_per_s"], 4)
    summary["highest_rung_that_ran"] = done[-1]["micro_batch"]
(out / "batch_ladder_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for r in rungs:
    if r.get("tokens_per_s"):
        print(f"  mb{r['micro_batch']}: {r['tokens_per_s']} tok/s, MFU {r.get('mfu_pct')}%"
              + (f", {r['gain_vs_prev_rung']}x vs previous rung" if r.get("gain_vs_prev_rung") else ""))
    else:
        print(f"  mb{r['micro_batch']}: {r.get('wall')} -- {r.get('reason','')[:90]}")
PYEOF

step "BATCH LADDER COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || echo "cache push FAILED"
bash shared/bin/push_results.sh "$BOX" || echo "push FAILED"
