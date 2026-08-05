#!/usr/bin/env bash
# Boot (or stop) the vLLM / NxD-Inference OpenAI server for one model+config.
#
#   launch_vllm.sh <model_key> <config> <boot_json_out>   # boot, wait healthy
#   launch_vllm.sh stop <boot_json_out>                   # kill by pid file
#
# On success <boot_json_out> records the boot as a first-class result:
#   {model_key, config, model_id, boot_wall_s,
#    cache_dir_size_mb_before, cache_dir_size_mb_after,
#    warm, server_pid, captured}
# because on Neuron the first boot of a (model, config) pair runs neuronx-cc
# for tens of minutes while a NEFF-cache hit boots in single-digit minutes --
# cold-vs-warm boot wall time IS one of the study's findings, not noise.
#
# On failure a structured load_failure.json is written NEXT TO the boot json:
#   {model_key, config, model_id, status, looks_like_capacity_failure,
#    reason, captured}
# and the script exits nonzero. run_all treats a recorded failure as an
# outcome (the qwen3 lane keys off exactly this file), so the record must
# survive even when nobody is watching the console.
#
# Server flags follow the AWS Neuron NxDI vLLM docs (quickstart-vllm-online-
# serving.md + quickstart-deploy-dlc.md), NOT upstream vLLM docs -- the DLAMI
# venv ships AWS's Neuron build and the two CLIs have drifted (see the
# TODO-VERIFY notes inline).
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODELS_DIR="${MODELS_DIR:-/opt/np/models}"

PORT=8000
TP_DEGREE=2                    # inf2.xlarge: 1 Inferentia2 = 2 NeuronCores
HEALTH_POLL_S=5
HEALTH_TIMEOUT_S=5400          # 90 min: a cold 8B compile can take most of it
STOP_GRACE_S=60                # SIGTERM -> SIGKILL escalation window
WARM_BOOT_THRESHOLD_S=300      # heuristic: NEFF-cache hit boots in < 5 min
S3_ARTIFACTS="s3://neuron-pipelines-artifacts-600627330911/artifacts"

# DLAMI venv resolution, same pattern as shared/run_all.sh.
NP_VENV="${NP_VENV:-$(ls -d /opt/aws_neuronx_venv* 2>/dev/null | head -1)}"
NP_PY="${NP_VENV:+$NP_VENV/bin/python}"
# torch_neuronx's Initializer shells out to `libneuronpjrt-path` (a venv-bin
# script); without this every engine boot dies at init under shells that don't
# have the venv on PATH (SSM, cron). Phase-1 gotcha #2, now fixed at the
# launcher itself instead of relying on each caller.
[ -n "$NP_VENV" ] && export PATH="$NP_VENV/bin:$PATH"
NP_PY="${NP_PY:-python3}"

# Weights cache on the big EBS volume (run_all.sh does the same for cpu lane);
# NEFF cache path matches shared/bin/sync_neuron_cache.sh so push/pull work.
export HF_HOME="${HF_HOME:-$MODELS_DIR/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"
# Force the NxD Inference framework (documented env; harmless when the venv
# only ships NxDI, decisive when transformers-neuronx is also installed).
export VLLM_NEURON_FRAMEWORK="${VLLM_NEURON_FRAMEWORK:-neuronx-distributed-inference}"
# vllm-neuron 0.21's worker unconditionally maps EFA interfaces unless told
# not to, and its mapper only knows trn2/trn3 families -- on inf2 every boot
# died in get_efa_bdf_mapping with "Unsupported instance family: inf2"
# (measured 2026-07-29, smoke_tinyllama load_failure). inf2.xlarge has no EFA
# hardware at all, so skipping is the correct config, not a workaround; the
# skip flag ships in the plugin itself (neuron_worker.py:515).
export NEURON_SKIP_EFA_AFFINITY="${NEURON_SKIP_EFA_AFFINITY:-1}"

# ------------------------------------------------------------------ stop mode
stop_server() {
  # Kill by pid file + wait. SIGTERM first so vLLM unloads cleanly and the
  # Neuron runtime frees HBM; SIGKILL (plus any orphaned engine children)
  # only after the grace period. A half-dead server holding both NeuronCores
  # would make every later lane fail with confusing "device busy" errors.
  local pid_file="$1"
  if [ ! -s "$pid_file" ]; then
    echo "no pid file at $pid_file (nothing to stop)"
    return 0
  fi
  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "--- stopping server pid $pid ---"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 "$STOP_GRACE_S"); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "server $pid ignored SIGTERM for ${STOP_GRACE_S}s; SIGKILL" >&2
      pkill -9 -P "$pid" 2>/dev/null || true
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pid_file"
}

if [ "${1:-}" = "stop" ]; then
  BOOT_JSON="${2:?usage: launch_vllm.sh stop <boot_json_out>}"
  stop_server "${BOOT_JSON%.json}.server.pid"
  exit 0
fi

# ------------------------------------------------------------------ boot mode
MODEL_KEY="${1:?usage: launch_vllm.sh <model_key> <config> <boot_json_out>}"
CONFIG="${2:?usage: launch_vllm.sh <model_key> <config> <boot_json_out>}"
BOOT_JSON="${3:?usage: launch_vllm.sh <model_key> <config> <boot_json_out>}"

mkdir -p "$(dirname "$BOOT_JSON")"
SERVER_LOG="${BOOT_JSON%.json}.server.log"
PID_FILE="${BOOT_JSON%.json}.server.pid"
TAIL_LOG="${BOOT_JSON%.json}.server.tail.log"

# Always preserve the tail of the server log on the way out -- success,
# failure, or an operator ^C during the 90-minute health poll. The full log
# survives too, but the tail is what a post-mortem reads first and it must
# exist even if a later lane truncates/rotates the main log.
save_log_tail() {
  [ -s "$SERVER_LOG" ] && tail -n 200 "$SERVER_LOG" > "$TAIL_LOG" 2>/dev/null
  return 0
}
trap save_log_tail EXIT

# ------------------------------------------------------- model catalogue
case "$MODEL_KEY" in
  llama31_base)
    MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
  llama31_dolly)
    # The trn1-trained fine-tune: merged on the Trainium box, pushed to S3,
    # pulled here on first use so a re-provisioned inf2 self-heals.
    MODEL_ID="$MODELS_DIR/llama31-8b-dolly-merged"
    if [ ! -d "$MODEL_ID" ] || [ -z "$(ls -A "$MODEL_ID" 2>/dev/null)" ]; then
      echo "--- $MODEL_ID absent; pulling merged fine-tune from S3 ---"
      mkdir -p "$MODEL_ID"
      if ! aws s3 sync "$S3_ARTIFACTS/llama31-8b-dolly-merged/" "$MODEL_ID/"; then
        echo "S3 pull of llama31-8b-dolly-merged failed" >&2
        exit 1
      fi
    fi ;;
  qwen3_base)
    MODEL_ID="Qwen/Qwen3-8B" ;;
  mistral7b)
    MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3" ;;
  qwen25_7b)
    # RAG LLM-ladder rung 2: qwen2 arch is a different NxDI path than the
    # qwen3 that crashed at generation -- all-Qwen stays possible via 2.5.
    MODEL_ID="Qwen/Qwen2.5-7B-Instruct" ;;
  smoke_tinyllama)
    MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ;;
  *)
    echo "unknown model key: $MODEL_KEY" >&2; exit 2 ;;
esac

# ------------------------------------------------------- server configs
# Two configs per model because NxDI preallocates KV for max-model-len x
# max-num-seqs at compile time: one compiled graph cannot cover both
# short-context/high-concurrency and long-context on 32 GB of HBM.
case "$CONFIG" in
  short) MAX_MODEL_LEN=2048;  MAX_NUM_SEQS=32 ;;
  long)  MAX_MODEL_LEN=9216;  MAX_NUM_SEQS=8  ;;  # exactly 1024+8192; 10240 died in NCC_INLA001 (internal compiler bound-check, 2026-07-29)
  smoke) MAX_MODEL_LEN=2048;  MAX_NUM_SEQS=4  ;;
  # Phase-split grids deliberately REUSE the `long` server geometry rather than
  # introducing new ones. NxDI preallocates KV for max-model-len x max-num-seqs
  # at COMPILE time, so a new geometry means a cold recompile -- and worse, a
  # different server, which would confound the phase split with a server change.
  #
  # Both grids fit inside 9216: prefill tops out at 8192+1, decode at 128+2048.
  # So these run on the SAME compiled graph as the published long lane, at zero
  # compile cost, and any difference is attributable to the request shape alone.
  # Concurrency in both grids stays <= 8 to respect MAX_NUM_SEQS.
  prefill|decode) MAX_MODEL_LEN=9216; MAX_NUM_SEQS=8 ;;
  *) echo "unknown config: $CONFIG (want short|long|smoke|prefill|decode)" >&2; exit 2 ;;
esac
# Probe overrides (Track B bisection): env wins over the config case so a
# probe lane can walk max-model-len without inventing new named configs.
MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE:-$MAX_MODEL_LEN}"
MAX_NUM_SEQS="${MAX_NUM_SEQS_OVERRIDE:-$MAX_NUM_SEQS}"

VLLM_BIN="${NP_VENV:+$NP_VENV/bin/vllm}"
if [ -z "$VLLM_BIN" ] || [ ! -x "$VLLM_BIN" ]; then
  VLLM_BIN="$(command -v vllm || true)"
fi
if [ -z "$VLLM_BIN" ]; then
  echo "no vllm executable found (NP_VENV='$NP_VENV')" >&2
  exit 1
fi

# Optional Neuron override, passed through verbatim from the environment.
# Default is a STOCK server: bucketing etc. stay at NxDI defaults so the
# measurement is of the shipped stack, not of our tuning.
# Flag split per AWS DLC quickstart: vLLM >= 0.11.0 takes
#   --additional-config '{"override_neuron_config": {...}}'
# and older Neuron builds take --override-neuron-config '{...}'.
# NOT EXERCISED in this study (stock servers only; see METHODOLOGY rule 2):
# venv's actual build honours the same split on first boot.
EXTRA_ARGS=()
if [ -n "${OVERRIDE_NEURON_CONFIG:-}" ]; then
  if "$NP_PY" - <<'PY'
import sys
try:
    import vllm
    major, minor = vllm.__version__.split("+")[0].split(".")[:2]
    sys.exit(0 if (int(major), int(minor)) >= (0, 11) else 1)
except Exception:
    sys.exit(0)  # cannot tell: assume current (>= 0.11) DLAMI build
PY
  then
    EXTRA_ARGS+=(--additional-config "{\"override_neuron_config\": $OVERRIDE_NEURON_CONFIG}")
  else
    EXTRA_ARGS+=(--override-neuron-config "$OVERRIDE_NEURON_CONFIG")
  fi
fi

echo "=== launch_vllm: $MODEL_KEY ($CONFIG) ==="
echo "    model_id      : $MODEL_ID"
echo "    tp / port     : $TP_DEGREE / $PORT"
echo "    max-model-len : $MAX_MODEL_LEN   max-num-seqs: $MAX_NUM_SEQS"
echo "    venv          : ${NP_VENV:-<none>}"
echo "    neuron cache  : $NEURON_COMPILE_CACHE_URL"
echo "    boot json     : $BOOT_JSON"

# Dedicated-box invariant (mirror of the MI300X sweep's unconditional
# `docker rm -f`): if something is already answering on the port -- a wedged
# server from an interrupted lane -- booting "on top" would instantly report
# healthy for the WRONG model. Sweep it first.
if curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "WARNING: stale server already healthy on :$PORT; killing it" >&2
  pkill -f "vllm serve" 2>/dev/null || true
  for _ in $(seq 1 30); do
    curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null 2>&1 || break
    sleep 1
  done
fi

CACHE_MB_BEFORE="$(du -sm "$NEURON_COMPILE_CACHE_URL" 2>/dev/null | cut -f1)"
CACHE_MB_BEFORE="${CACHE_MB_BEFORE:-0}"

# Flags per AWS Neuron NxDI vLLM docs:
#   --model             (AWS docs pass the model via --model, and their DLC
#                        example uses it on a 0.11-based build. TODO-VERIFY:
#                        upstream vLLM hard-errors on --model with `serve`,
#                        wanting it positional -- if the venv build does too,
#                        the load_failure.json reason line will say so
#                        verbatim; switch to positional then.)
#   --tensor-parallel-size 2        both NeuronCores of the single Inferentia2
#   --no-enable-prefix-caching      per the NxDI online-serving quickstart
#                                   (TODO-VERIFY on builds older than 0.9)
#   --seed 0                        style contract: everything is seeded
nohup "$VLLM_BIN" serve \
    --model "$MODEL_ID" \
    --tensor-parallel-size "$TP_DEGREE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --no-enable-prefix-caching \
    --seed 0 \
    --port "$PORT" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo "--- server pid $SERVER_PID; polling /health every ${HEALTH_POLL_S}s (up to $((HEALTH_TIMEOUT_S / 60)) min: cold compile is slow) ---"

T0=$(date +%s)
DEADLINE=$((T0 + HEALTH_TIMEOUT_S))
READY=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if curl -sf --max-time 4 "http://localhost:$PORT/health" >/dev/null 2>&1; then
    READY=1; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server process exited before becoming healthy" >&2
    break
  fi
  sleep "$HEALTH_POLL_S"
done
BOOT_WALL_S=$(( $(date +%s) - T0 ))

CACHE_MB_AFTER="$(du -sm "$NEURON_COMPILE_CACHE_URL" 2>/dev/null | cut -f1)"
CACHE_MB_AFTER="${CACHE_MB_AFTER:-0}"

if [ "$READY" != "1" ]; then
  echo "SERVER NEVER BECAME HEALTHY: $MODEL_KEY ($CONFIG) after ${BOOT_WALL_S}s" >&2
  tail -n 80 "$SERVER_LOG" >&2 || true

  # Structured failure record, MI300X-style. Classify capacity failures by
  # pattern (Neuron phrasings included), but ALSO keep the verbatim error
  # line: the MI300X harness once recorded looks_like_oom=false for a
  # textbook capacity failure because it only pattern-matched -- the verbatim
  # reason is what makes a future mismatch visible instead of silent.
  CAPACITY=false
  if grep -qiE "out of memory|outofmemory|oom-kill|no available memory|cache blocks|insufficient memory|free memory|failed to allocate|allocation failure|resource.?exhausted|out of device memory" "$SERVER_LOG" 2>/dev/null; then
    CAPACITY=true
  fi
  REASON="$(grep -aE "ERROR|[A-Za-z]+Error(:| )|Traceback" "$SERVER_LOG" 2>/dev/null | tail -1 | cut -c1-500)"

  python3 - "$(dirname "$BOOT_JSON")/load_failure.json" \
    "$MODEL_KEY" "$CONFIG" "$MODEL_ID" "$CAPACITY" "${REASON:-}" <<'PY'
import json, sys, time
path, key, config, model_id, capacity = sys.argv[1:6]
reason = sys.argv[6] if len(sys.argv) > 6 else ""
json.dump({
    "model_key": key,
    "config": config,
    "model_id": model_id,
    "status": "server_failed_to_start",
    "looks_like_capacity_failure": capacity == "true",
    "reason": reason or "(no error line captured; see server tail log)",
    "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, open(path, "w"), indent=2)
PY
  echo "failure recorded: $(dirname "$BOOT_JSON")/load_failure.json"
  # a half-started server may still hold the NeuronCores -- release them
  stop_server "$PID_FILE"
  exit 1
fi

python3 - "$BOOT_JSON" "$MODEL_KEY" "$CONFIG" "$MODEL_ID" \
  "$BOOT_WALL_S" "$CACHE_MB_BEFORE" "$CACHE_MB_AFTER" "$SERVER_PID" \
  "$WARM_BOOT_THRESHOLD_S" <<'PY'
import json, sys, time
(path, key, config, model_id, wall,
 before, after, pid, warm_thresh) = sys.argv[1:10]
wall = int(wall)
json.dump({
    "model_key": key,
    "config": config,
    "model_id": model_id,
    "boot_wall_s": wall,
    "cache_dir_size_mb_before": int(before),
    "cache_dir_size_mb_after": int(after),
    # heuristic: a NEFF-cache hit skips neuronx-cc entirely and boots in
    # minutes; a cold compile takes tens of minutes. The cache-size delta
    # above is the corroborating evidence.
    "warm": wall < int(warm_thresh),
    "server_pid": int(pid),
    "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, open(path, "w"), indent=2)
PY

echo "--- server healthy in ${BOOT_WALL_S}s (cache ${CACHE_MB_BEFORE} -> ${CACHE_MB_AFTER} MB); boot recorded in $BOOT_JSON ---"
