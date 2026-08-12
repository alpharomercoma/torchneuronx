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

# NEURON CACHES FAILED COMPILES, so any ladder whose rungs can fail needs this
# flag: without it the FIRST failure is replayed for every rung above it and N
# rungs look like N measurements when only one is.
#
# CORRECTION (2026-08-12). This comment used to claim that exact thing HAD
# happened here -- that micro-batch 4 and 8 reported the identical instruction
# count (2064384) because they replayed micro-batch 2's cached failure. That
# diagnosis was wrong, and the "identical count" tell it rested on is unsound.
# What the logs actually show:
#
#   * Every rung emitted its OWN distinct MODULE_ hash and its own
#     "Failed compilation with ['neuronx-cc', ...]" subprocess. The compiles
#     were independent, with and without this flag.
#   * The compiler prints the shape of the operator it rejects. It is a
#     flash-attention backward kernel, and dy_ref is (2|4|8, 16, 128, 4096)
#     across the rungs -- a 4x range, 16.8M to 67.1M elements.
#   * All of them report 2064384 anyway, as does a fourth config with gradient
#     checkpointing OFF, a lever no rung varies.
#   * All 59 occurrences across the whole study, 10 lane logs, 3 runs and 2
#     compiler-flag hashes, read 2064384. Nothing produces a different value.
#
# An instruction count cannot be invariant to a 4x change in the volume of the
# operator that triggered the error, so the number does not describe the graph.
# 2064384 is a constant (63 * 2^15); what it denotes is unknown and needs
# neuronx-cc internals, not another lane.
#
# The flag stays -- cached failure replay is a real hazard -- but it is kept on
# principle, not because it was ever caught happening here. The SOUND audit
# signal is the per-rung MODULE_ hash, which is what the summary below checks.
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:-} --retry_failed_compilation"

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
      # THE AUDIT TRAIL IS THE MODULE HASH, NOT THE INSTRUCTION COUNT.
      # compiler_instructions is still recorded for provenance, but it is
      # invariant across every configuration this study has compiled and
      # therefore proves nothing (see the CORRECTION above). What does prove
      # independent compilation is a distinct MODULE_ hash per rung plus a real
      # neuronx-cc invocation, both of which the compiler prints itself.
      INSTR=$(grep -oE "Instructions generated by compiler [0-9]+" "$OUT/$tag.log" | grep -oE "[0-9]+$" | head -1)
      MODS=$("$PY" - "$OUT/$tag.log" <<'PYMOD'
import re,sys,json
t=open(sys.argv[1],errors="replace").read()
m=sorted(set(re.findall(r"Failed compilation with \[.neuronx-cc.*?model\.(MODULE_\d+\+\w+)\.hlo_module\.pb",t)))
print(json.dumps(m))
PYMOD
)
      printf '{"tag":"%s","box":"%s","status":"failed","reason":"%s","compiler_instructions":%s,"compiler_instructions_is_diagnostic":false,"failed_module_hashes":%s,"captured":"%s"}\n' \
        "$tag" "$BOX" "$REASON" "${INSTR:-null}" "${MODS:-[]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$tag.failure.json"
      echo "  $tag FAILED (receipt, modules=${MODS:-[]}) -- ladder continues"; return 0; }
  bash shared/bin/push_results.sh "$BOX" >/dev/null 2>&1 || echo "  push FAILED"
  return 0
}

# GRAD-ACCUM IS HELD CONSTANT. The first version of this ladder halved it as
# micro-batch doubled, to keep the global batch fixed. That was wrong on this
# stack, and a control run proved it:
#
#   micro-batch 1, grad-accum 8 -> 1147 ms per micro-batch
#   micro-batch 1, grad-accum 4 ->  714 ms per micro-batch, MFU 146.7%
#
# Same shape, same graph, and an MFU that cannot be physical. Gradient
# accumulation is UNROLLED INTO THE COMPILED GRAPH here -- changing it forced a
# full recompile (1179 s) -- and the per-step timer captures a different share
# of deferred XLA work at different accumulation depths. So steady-state
# throughput is NOT comparable across grad_accum values, and a ladder that
# varied it was comparing numbers that cannot be compared.
#
# Holding it constant means the global batch now GROWS with micro-batch. That
# changes the optimisation trajectory, which would matter for a quality claim
# and does not matter here: this lane measures throughput only, and its loss is
# not reported. Trading a confound we cannot measure for one we can name and
# discount is the right way round.
GRAD_ACCUM="${GRAD_ACCUM:-8}"
for MB in $RUNGS; do
  lane "mb${MB}_seq${SEQ}" --model "$MODEL" --seq-len "$SEQ" \
    --micro-batch "$MB" --grad-accum "$GRAD_ACCUM" --max-steps "$STEPS"
done

# ---- Summary ---------------------------------------------------------------
"$PY" - "$OUT" "$BOX" "$SEQ" <<'PYEOF' || echo "summary FAILED (rungs are on disk)"
import json, pathlib, re, sys
out, box, seq = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]


def _hashes(receipt, out, mb, seq):
    """Module hashes of the compiles that FAILED for this rung.

    Prefer the receipt. Fall back to the lane log so that summaries written
    before the receipt carried this field can still be regenerated from
    evidence already on disk, without re-running a single lane.
    """
    got = receipt.get("failed_module_hashes")
    if got:
        return sorted(got)
    log = out / f"mb{mb}_seq{seq}.log"
    if not log.is_file():
        return []
    return sorted(set(re.findall(
        r"Failed compilation with \[.neuronx-cc.*?model\.(MODULE_\d+\+\w+)\.hlo_module\.pb",
        log.read_text(errors="replace"))))


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
        d = json.loads(bad.read_text())
        r = d.get("reason", "")
        rungs.append({"micro_batch": mb, "status": "failed",
                      "wall": "device OOM" if "EOOM" in r else
                              "compiler graph limit" if "EXTP003" in r else "failed",
                      "compiler_instructions": d.get("compiler_instructions"),
                      "compiler_instructions_is_diagnostic": False,
                      "failed_module_hashes": _hashes(d, out, mb, seq),
                      "reason": r[:160]})
summary = {"box": box, "seq_len": int(seq), "rungs": rungs}
# INDEPENDENT COMPILATION IS PROVEN BY THE MODULE HASH, NOT THE INSTRUCTION COUNT.
#
# The previous version of this check flagged a ladder SUSPECT when the failed
# rungs shared an instruction count. That was withdrawn on 2026-08-12: the count
# is invariant across a 4x change in the rejected operator's tensor volume and
# across every configuration in this study, so identical counts are the NORMAL
# result for genuinely independent compiles and the flag fired on a sound ladder.
#
# A shared MODULE_ hash is the real defect signature: same hash means the rungs
# submitted the same graph, which is what "they did not compile independently"
# actually means. Distinct hashes mean distinct graphs, whatever the count says.
hashes = [tuple(r.get("failed_module_hashes") or ()) for r in rungs
          if r.get("status") == "failed"]
hashes = [h for h in hashes if h]
if len(hashes) > 1 and len(set(hashes)) == 1:
    summary["SUSPECT"] = ("failed rungs share the SAME compiler module hash "
                          f"({list(hashes[0])}). The rungs submitted an identical "
                          "graph, so they are not independent measurements. Do NOT "
                          "report this ladder as locating a batch-size wall.")
    print(f"  *** SUSPECT LADDER: {summary['SUSPECT']} ***")
elif hashes:
    summary["independent_compiles_verified"] = (
        f"{len(hashes)} failed rungs, {len(set(hashes))} distinct module-hash sets "
        "-- each rung compiled its own graph")
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
