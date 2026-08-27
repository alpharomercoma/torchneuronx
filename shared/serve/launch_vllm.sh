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

PORT="${PORT_OVERRIDE:-8000}"
TP_DEGREE="${TP_DEGREE_OVERRIDE:-2}"   # inf2.xlarge: 1 Inferentia2 = 2 NeuronCores

# ------------------------------------------------- NeuronCore partitioning
# A NeuronCore belongs to exactly one process. The default TP=2 server
# therefore owns BOTH cores of the inf2.xlarge and nothing else can touch the
# device -- which is precisely why the Track-F RAG probes died with
# "The PyTorch Neuron Runtime could not be initialized": the encoders tried to
# torch.jit.load next to a server that already held every core.
#
# NEURON_RT_VISIBLE_CORES is the documented way out (Neuron docs,
# torch-neuronx/programming-guide/inference/core-placement). Set it to pin this
# server to a subset, and pin the co-resident process to the complement:
#
#   NP_VISIBLE_CORES=0 TP_DEGREE_OVERRIDE=1  launch_vllm.sh ...   # LLM on nc0
#   NEURON_RT_VISIBLE_CORES=1                python encoders.py   # encoders nc1
#
# Only usable when the model fits in one core's share of HBM. An 8B model in
# bf16 is ~16 GB against a 16 GB per-core budget, so it CANNOT be pinned to one
# core -- co-residency needs a smaller LM, not a different flag.
if [ -n "${NP_VISIBLE_CORES:-}" ]; then
  export NEURON_RT_VISIBLE_CORES="$NP_VISIBLE_CORES"
  echo "--- pinned to NeuronCore(s) $NEURON_RT_VISIBLE_CORES (tp=$TP_DEGREE) ---"
fi
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
  llama31_dolly|llama31_dolly_trn1)
    # The TRAINIUM1-trained fine-tune.
    #
    # This used to pull from artifacts/llama31-8b-dolly-merged/, which BOTH
    # training boxes wrote to. When Trainium2 ran the same lane it overwrote
    # trn1's weights at that path, so a later pull would have served trn2's
    # model under trn1's name -- and `verify_models.sh` would have failed
    # against the sha256 digests Phase 2 recorded. The two merges genuinely
    # differ (4 of 6 files), as they must: different final loss.
    #
    # Both versions were recovered from S3 versioning into explicit,
    # box-specific prefixes. Neither key is ambiguous now.
    MODEL_ID="$MODELS_DIR/llama31-8b-dolly-merged-trn1"
    if [ ! -d "$MODEL_ID" ] || [ -z "$(ls -A "$MODEL_ID" 2>/dev/null)" ]; then
      echo "--- $MODEL_ID absent; pulling the trn1 merged fine-tune from S3 ---"
      mkdir -p "$MODEL_ID"
      if ! aws s3 sync "$S3_ARTIFACTS/llama31-8b-dolly-merged-trn1/" "$MODEL_ID/"; then
        echo "S3 pull of llama31-8b-dolly-merged-trn1 failed" >&2
        exit 1
      fi
    fi ;;
  llama31_dolly_trn2)
    # The TRAINIUM2-trained fine-tune. Closes the train-then-serve loop for the
    # newer chip: Phase 2 closed it for Trainium1 only, so every serving number
    # in this study so far describes weights trained on trn1.
    MODEL_ID="$MODELS_DIR/llama31-8b-dolly-merged-trn2"
    if [ ! -d "$MODEL_ID" ] || [ -z "$(ls -A "$MODEL_ID" 2>/dev/null)" ]; then
      echo "--- $MODEL_ID absent; pulling the trn2 merged fine-tune from S3 ---"
      mkdir -p "$MODEL_ID"
      if ! aws s3 sync "$S3_ARTIFACTS/llama31-8b-dolly-merged-trn2/" "$MODEL_ID/"; then
        echo "S3 pull of llama31-8b-dolly-merged-trn2 failed" >&2
        exit 1
      fi
    fi ;;
  qwen3_base)
    MODEL_ID="Qwen/Qwen3-8B" ;;
  mistral7b)
    MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3" ;;
  gemma3_12b)
    # Third modern architecture through the identical vLLM/NxDI path, chosen as
    # the direct comparable to the Mistral-7B lane (13.2). 24.4 GB of bf16
    # weights against inf2.xlarge's 32 GB (2 x 16 GB), so TP=2 puts 12.2 GB on
    # each core -- it fits, with less headroom than Mistral's 14.5 GB total but
    # more than the 8B ORPO lane that OOM'd at 15.6/16 GB on trn1.
    #
    # RISK, RECORDED BEFORE THE RUN: the HF checkpoint is
    # Gemma3ForConditionalGeneration (multimodal) while NxDI ships
    # NeuronGemma3ForCausalLM (text tower only). If the loader will not extract
    # the text config from the multimodal wrapper this lane fails at model
    # construction, which is a receipt naming a supported-architecture gap
    # rather than a capacity one. gemma3_4b is the fallback rung.
    MODEL_ID="google/gemma-3-12b-it" ;;
  gemma3_4b)
    MODEL_ID="google/gemma-3-4b-it" ;;
  gpt_oss_20b)
    # The MoE rung. 21B total / ~3.6B active -- and TOTAL is what has to fit,
    # since every expert's weights are resident even though only a few fire per
    # token. That is the fact that rules out Qwen3-30B-A3B (61 GB) and
    # Mixtral-8x7B (93 GB) on a 32 GB box regardless of how few experts activate.
    #
    # NxDI ships a gpt_oss family and real MoE plumbing (modules/moe_v2.py,
    # moe_tp_degree / moe_ep_degree), so this is the only modern MoE that is both
    # SUPPORTED and plausibly SIZED for inf2.xlarge.
    #
    # THE OPEN QUESTION, RECORDED BEFORE THE RUN: the checkpoint is 27.5 GB on
    # disk because it ships natively MXFP4 (4-bit experts + higher-precision
    # attention). If NxDI loads it as-is that is ~13.8 GB/core under TP=2 and it
    # fits. If NxDI DEQUANTISES to bf16 it becomes 21B x 2 = ~42 GB, i.e.
    # ~21 GB/core, and it cannot fit. Which of those happens is exactly what this
    # lane measures, and an OOM here is a real finding about MXFP4 support rather
    # than a sizing mistake.
    MODEL_ID="openai/gpt-oss-20b" ;;
  qwen25_7b)
    # RAG LLM-ladder rung 2: qwen2 arch is a different NxDI path than the
    # qwen3 that crashed at generation -- all-Qwen stays possible via 2.5.
    MODEL_ID="Qwen/Qwen2.5-7B-Instruct" ;;
  qwen3_1_7b)
    # RAG co-residency rung: small enough to pin to ONE NeuronCore, which is
    # the whole point. ~3.4 GB of bf16 weights plus ~114 KiB/token of KV
    # against a 16 GB per-core budget, leaving the second core free for the
    # embedder and the reranker. Qwen3-8B cannot do this -- its weights alone
    # are ~16 GB, so it must span both cores and the appliance cannot exist.
    MODEL_ID="Qwen/Qwen3-1.7B" ;;
  qwen3_4b)
    # Same idea with more headroom used: ~8 GB of weights on one core.
    MODEL_ID="Qwen/Qwen3-4B" ;;
  llama32_1b)
    # The co-residency LM. ~2.5 GB of bf16 weights on one 16 GB core, leaving
    # nc1 for the Qwen encoders. Weights are already on the box: this is the
    # model the speculative-decoding lane used as its draft.
    MODEL_ID="meta-llama/Llama-3.2-1B-Instruct" ;;
  llama32_3b)
    MODEL_ID="meta-llama/Llama-3.2-3B-Instruct" ;;
  qwen3_0_6b)
    # Smallest rung. Matches the embedder and reranker exactly in size, so an
    # all-Qwen appliance at 0.6B x 3 is the minimal form of the thing.
    MODEL_ID="Qwen/Qwen3-0.6B" ;;
  qwen25_1_5b)
    # Declared FALLBACK for the co-residency lane. REPORT §9 excludes Qwen3-8B
    # serving: it boots, passes /health, then crashes the engine core on the
    # first generation step. That was never separated into "Qwen3 architecture
    # on NxDI 0.10" versus "8B at TP=2", so the qwen3_1_7b rung above is the
    # experiment that separates them. If Qwen3 turns out to be architecturally
    # broken here, qwen2 is a different NxDI code path and keeps the appliance
    # alive at a size that still fits one core.
    MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct" ;;
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
  # SHORT geometry (2048/32), not long (9216): the long lane never booted --
  # it is a recorded failure, and its cache entry is a poisoned NEFF from
  # 2026-07-29. Both phase grids are sized to fit 2048.
  prefill|decode) MAX_MODEL_LEN=2048; MAX_NUM_SEQS=32 ;;
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
