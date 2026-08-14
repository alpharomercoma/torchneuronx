#!/usr/bin/env bash
# CLOSE THE TRAINIUM2 GAPS -- and then TERMINATE THE BOX.
#
# Kicked off unattended by np-autorun when the on-demand capacity hunt finally
# lands a trn2.3xlarge. No operator is present at any point.
#
# WHY NOT run_phase3_trn2.sh
# --------------------------
# That driver runs the whole suite. It is resumable, but resumability reads the
# LOCAL results tree, and a hunt-captured box is brand new -- its EBS comes from
# the launch template, so trn2/results/ is empty and every np_recorded() guard
# misses. That is not hypothetical: the ASG replacement box on 2026-08-05 did
# exactly this and re-ran the entire suite from zero.
#
# At the on-demand rate that mistake is expensive. Spot pricing puts
# trn2.3xlarge at $6.41/hr in sa-east-1, and spot never exceeds on-demand, so
# on-demand is AT LEAST that. A redundant full suite is several hundred dollars
# of credit for results already sitting in S3.
#
# So this driver PULLS THE EXISTING RESULTS FIRST, then runs only what is
# genuinely missing, then kills the box.
#
# WHAT IS ACTUALLY MISSING (audited 2026-08-12 across all 68 lanes)
# -----------------------------------------------------------------
# Very little, and this file will not pretend otherwise:
#
#   ctrl_mb1_acc4   trn1 has it, trn2 does not. A real coverage gap, and the
#                   control that anchors the batch ladder's grad-accum finding.
#   cifar_vit       Passed on trn1, died on trn2 with "[F137] neuronx-cc was
#                   forcibly killed", a HOST memory failure during compilation,
#                   not a device limit. Retried here with the compiler's
#                   parallelism cut, which is the only lever that addresses the
#                   recorded cause. If it dies again the receipt is stronger for
#                   having tried.
#
# Deliberately NOT run:
#
#   ctx 9216/10240/12288/16384   settled. 9216 is a hard API limit ("Only
#                   support sequence as multiples of 2K"); 16384 exhausted 124
#                   GiB of host RAM. Re-running buys nothing.
#   opportunistic seed43/44      the trn1 triplet proved --seed is a NO-OP on
#                   this stack (three seeds, bit-identical loss, because packing
#                   pins data order). These are full 3-epoch runs for a result
#                   we know in advance. Skipping them is the whole reason this
#                   box can terminate in ~1h instead of ~4h.
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
NP_TAG=GAPFILL-TRN2
cd "$BENCH_DIR"

BOX=trn2
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"   # libneuronpjrt-path; cost seven lanes in Phase 2
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"

export NP_DEVICE=trn2
export NP_REGION="${NP_REGION:-sa-east-1}"
export NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache-v3}"
export NEURON_LOGICAL_NC_CONFIG="${NEURON_LOGICAL_NC_CONFIG:-2}"
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:-} --retry_failed_compilation"
NP_BUCKET="${NP_BUCKET:-neuron-pipelines-artifacts-600627330911}"

SELF_TERMINATE="${SELF_TERMINATE:-1}"

# ---------------------------------------------------------------------------
# Terminate on EVERY exit path, including failure.
# ---------------------------------------------------------------------------
# An unattended box that finishes its work and keeps billing is the failure mode
# this whole design exists to avoid. The ASG scheduled action is a backstop
# hours away, not a plan. ShouldDecrementDesiredCapacity is what stops the group
# from immediately launching a replacement -- without it, terminating self just
# restarts the hunt and the box comes back.
terminate_self() {
  local rc=$?
  np_log "wrapping up (exit $rc) -- final push before terminate"
  bash shared/bin/push_results.sh "$BOX" || np_log "FINAL PUSH FAILED -- results may be lost"
  bash shared/bin/sync_neuron_cache.sh push || np_log "cache push failed (non-fatal)"
  if [ "$SELF_TERMINATE" != "1" ]; then
    np_log "SELF_TERMINATE=0 -- leaving the box up"
    return
  fi
  local iid
  iid=$(curl -s -m 5 -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
        | { read -r t; curl -s -m 5 -H "X-aws-ec2-metadata-token: $t" \
            http://169.254.169.254/latest/meta-data/instance-id; })
  if [ -z "$iid" ]; then
    np_log "could not read instance-id from IMDS; falling back to shutdown"
    shutdown -h now || true
    return
  fi
  np_log "terminating $iid and decrementing desired capacity to 0"
  # Retry the RIGHT call rather than reaching for shutdown on the first error.
  # A `shutdown` halts the box without decrementing, so the group replaces it
  # and the meter restarts -- the opposite of what this trap is for. Since the
  # hunt now runs with no scheduled stop ceiling, this call is the primary
  # spend bound and a transient API error must not be allowed to defeat it.
  local attempt
  for attempt in 1 2 3 4 5; do
    if aws autoscaling terminate-instance-in-auto-scaling-group \
         --region "$NP_REGION" --instance-id "$iid" \
         --should-decrement-desired-capacity; then
      np_log "terminated $iid (attempt $attempt)"
      return
    fi
    np_log "terminate attempt $attempt failed; retrying in 30s"
    sleep 30
  done
  np_log "ASG terminate failed 5x -- halting; the 6h boot watchdog is the last line"
  shutdown -h now || true
}
trap terminate_self EXIT

# ---------------------------------------------------------------------------
# 0. Rehydrate results BEFORE anything else.
# ---------------------------------------------------------------------------
# Without this the guards below all miss on a fresh box and the driver re-runs
# work that is already banked in S3.
np_step "rehydrating trn2/results from S3"
mkdir -p "$BENCH_DIR/trn2/results"
aws s3 sync "s3://$NP_BUCKET/results/trn2/" "$BENCH_DIR/trn2/results/" \
  || np_die "could not rehydrate results -- refusing to run blind and re-do banked work" 5
np_log "rehydrated $(find "$BENCH_DIR/trn2/results" -name '*.json' | wc -l | tr -d ' ') result JSONs"

bash shared/bin/hf_login.sh >/dev/null 2>&1 || np_log "WARN: hf_login failed"
bash shared/bin/sync_neuron_cache.sh pull || np_log "WARN: NEFF cache pull failed (cold compiles ahead)"

OUT_BL="$BENCH_DIR/trn2/results/batch_ladder"
OUT_AC="$BENCH_DIR/trn2/results/academic"
mkdir -p "$OUT_BL" "$OUT_AC"

# ---------------------------------------------------------------------------
# 1. ctrl_mb1_acc4 -- the genuine coverage gap
# ---------------------------------------------------------------------------
# trn1's control that proved grad-accum is unrolled into the compiled graph:
# mb1/acc8 gave 1147 ms per micro-batch, mb1/acc4 gave 714 ms and an impossible
# 146.7% MFU. That is why the ladder holds grad_accum constant. trn2 has never
# run the control, so the finding rests on one chip.
np_log "=== 1/2 ctrl_mb1_acc4 (batch-ladder control, trn1 has it and trn2 does not) ==="
if np_recorded "$OUT_BL/ctrl_mb1_acc4"; then
  np_log "skip ctrl_mb1_acc4 (recorded)"
else
  np_wait_for_chip
  P="$BENCH_DIR/trn2/results/extras/tp_probe.json"
  if [ -s "$P" ]; then
    read -r LNC NPROC TP <<<"$("$PY" -c '
import json,sys
d=json.load(open(sys.argv[1])); print(d["lnc"], d["nproc"], d["tp"])' "$P")"
  else
    np_log "tp_probe.json absent after rehydrate -- falling back to the published LNC=2/TP=4"
    LNC=2; NPROC=4; TP=4
  fi
  export NEURON_LOGICAL_NC_CONFIG="$LNC" NP_TELEMETRY_CORES="$NPROC"
  np_step "ctrl_mb1_acc4 (lnc=$LNC nproc=$NPROC tp=$TP)"
  "$PY" "$TELEM" --out "$OUT_BL/ctrl_mb1_acc4.telemetry.csv" -- \
    "$PY" "$BENCH_DIR/shared/train/sft_lora.py" --tag ctrl_mb1_acc4 \
      --device-profile trn2 --nproc-per-node "$NPROC" --tensor-parallel-size "$TP" \
      --model meta-llama/Llama-3.1-8B-Instruct --seq-len 4096 \
      --micro-batch 1 --grad-accum 4 --max-steps 30 \
      --out "$OUT_BL/ctrl_mb1_acc4.json" > "$OUT_BL/ctrl_mb1_acc4.log" 2>&1 \
    || np_receipt "$OUT_BL/ctrl_mb1_acc4.failure.json" ctrl_mb1_acc4 "$BOX" "$OUT_BL/ctrl_mb1_acc4.log"
  bash shared/bin/push_results.sh "$BOX" >/dev/null 2>&1 || np_log "  push failed"
fi

# ---------------------------------------------------------------------------
# 2. cifar_vit -- retry against the RECORDED cause
# ---------------------------------------------------------------------------
# The trn2 receipt says "[F137] neuronx-cc was forcibly killed - This most
# commonly occurs due to insufficient system memory". That is the HOST, not the
# device: 128 GiB exhausted by parallel compiler workers. Serialising them is
# the one lever that targets that cause. Anything else would be a re-run, not a
# retry, and a re-run of a known failure is not a measurement.
np_log "=== 2/2 cifar_vit (retry with serialised compilation) ==="
if np_have "$OUT_AC/cifar_vit.json"; then
  np_log "skip cifar_vit (already passing)"
else
  np_wait_for_chip
  rm -f "$OUT_AC/cifar_vit.failure.json"   # this attempt writes its own verdict
  np_step "cifar_vit (NEURON_PARALLEL_COMPILE_MAX_WORKERS=1)"
  # Flags must match academic/run_academic.sh EXACTLY (--dataset cifar --arch
  # vit, no --tag), or this is a different experiment wearing the same name.
  ACAD="$BENCH_DIR/academic/train_academic.py"
  if [ ! -s "$ACAD" ]; then
    np_log "academic trainer missing -- recording that rather than guessing a path"
    printf '{"tag":"cifar_vit","box":"trn2","status":"failed","reason":"gapfill could not locate %s","captured":"%s"}\n' \
      "$ACAD" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT_AC/cifar_vit.failure.json"
  else
    NEURON_PARALLEL_COMPILE_MAX_WORKERS=1 \
    "$PY" "$TELEM" --out "$OUT_AC/cifar_vit.telemetry.csv" -- \
      "$PY" "$ACAD" --dataset cifar --arch vit \
        --out "$OUT_AC/cifar_vit.json" > "$OUT_AC/cifar_vit.log" 2>&1 \
      || np_receipt "$OUT_AC/cifar_vit.failure.json" cifar_vit "$BOX" "$OUT_AC/cifar_vit.log"
  fi
  bash shared/bin/push_results.sh "$BOX" >/dev/null 2>&1 || np_log "  push failed"
fi

np_step "GAPFILL COMPLETE -- terminating"
# The EXIT trap does the final push, the cache push and the termination.
