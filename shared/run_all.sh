#!/usr/bin/env bash
# Full suite driver. Identical on both machines -- each box's
# scripts/run_all.sh is a thin wrapper that only sets BOX and the three dirs.
#
# Lanes run in priority order and each is independently resumable, so stopping
# early degrades gracefully instead of losing everything. Anything already
# present in results/ is skipped unless FORCE=1.
#
# Unlike the GPU study this harness descends from, the two boxes here run
# DIFFERENT lanes (trn1 trains, inf2 serves), so the driver branches on $BOX.
# The file itself stays byte-identical on both machines -- that invariant is
# what makes "the harness was the same" a checkable claim rather than a hope.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-$(dirname "$BENCH_DIR")/results}"
MODELS_DIR="${MODELS_DIR:-/opt/np/models}"
BOX="${BOX:?set BOX=trn1 or BOX=inf2 (the per-box wrapper does this)}"
TELEM="$BENCH_DIR/shared/telemetry.py"
FORCE="${FORCE:-0}"

# The DLAMI ships one venv per framework; the per-box wrapper may override.
# Training box: torch-neuronx + optimum-neuron. Inference box: vLLM Neuron.
NP_VENV="${NP_VENV:-$(ls -d /opt/aws_neuronx_venv* 2>/dev/null | head -1)}"
PY="${NP_VENV:+$NP_VENV/bin/python}"
PY="${PY:-python3}"

mkdir -p "$RESULTS_DIR"/{train,serve,cpu,quality,sustained,compile}

have() { [ "$FORCE" != "1" ] && [ -s "$1" ]; }

step() { echo; echo "############ $* ############"; echo; }

# --------------------------------------------------------------- lane 0: specs
step "lane 0: provenance"
bash "$BENCH_DIR/shared/sysinfo.sh" "$RESULTS_DIR/specs.txt"
MODELS_DIR="$MODELS_DIR" \
  bash "$BENCH_DIR/shared/verify_models.sh" "$RESULTS_DIR/model_hashes.txt" >/dev/null

# ---------------------------------------------------------------- lane 1: cpu
# Host noise floor. On inf2.xlarge (4 vCPU) this doubles as the documented
# ceiling for client-side tokenization -- worth having BEFORE reading any
# high-concurrency serving number.
step "lane 1: host CPU"
if have "$RESULTS_DIR/cpu/cpu.json"; then
  echo "skip cpu (exists)"
else
  HF_HOME="$MODELS_DIR/hf" "$PY" "$BENCH_DIR/shared/cpu/cpu_bench.py" \
    --out "$RESULTS_DIR/cpu/cpu.json" \
    --tokenizer-dir meta-llama/Llama-3.1-8B-Instruct \
    > "$RESULTS_DIR/cpu/cpu.log" 2>&1 || echo "  cpu FAILED"
fi

if [ "$BOX" = "trn1" ]; then
  # ============================================================ TRAINING BOX

  # ------------------------------------------------- lane 2: plumbing smoke
  # TinyLlama-1.1B, 20 steps, ungated. Validates the whole path (dataset ->
  # collator -> XLA compile -> step loop -> metrics JSON -> telemetry CSV)
  # for ~$0.50 before any 8B lane is allowed to burn hours.
  step "lane 2: training smoke (TinyLlama-1.1B, 20 steps)"
  if have "$RESULTS_DIR/train/smoke_tinyllama.json"; then
    echo "skip smoke (exists)"
  else
    "$PY" "$TELEM" --out "$RESULTS_DIR/train/smoke_tinyllama.telemetry.csv" -- \
      "$PY" "$BENCH_DIR/shared/train/sft_lora.py" \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tag smoke_tinyllama \
        --max-steps 20 --out "$RESULTS_DIR/train/smoke_tinyllama.json" \
      > "$RESULTS_DIR/train/smoke_tinyllama.log" 2>&1 \
      || echo "  smoke FAILED (see train/smoke_tinyllama.log)"
  fi

  # ---------------------------------------------- lane 3: precompile llama31
  # neuron_parallel_compile populates the NEFF cache so the real run measures
  # training, not compilation. Compile wall time is a first-class result.
  # RULE (pre-registered): precompile executes with invalid numerics -- no
  # loss value from this lane is ever recorded anywhere.
  step "lane 3: precompile Llama 3.1 8B"
  if have "$RESULTS_DIR/compile/llama31_train.json"; then
    echo "skip precompile (exists)"
  else
    bash "$BENCH_DIR/shared/train/precompile.sh" \
      meta-llama/Llama-3.1-8B-Instruct \
      "$RESULTS_DIR/compile/llama31_train.json" \
      > "$RESULTS_DIR/compile/llama31_train.log" 2>&1 \
      || echo "  precompile FAILED (see compile/llama31_train.log)"
  fi

  # ------------------------------------------------ lane 4: llama 3.1 8B SFT
  step "lane 4: LoRA SFT Llama 3.1 8B (primary)"
  if have "$RESULTS_DIR/train/llama31_lora.json"; then
    echo "skip llama31 lora (exists)"
  else
    "$PY" "$TELEM" --out "$RESULTS_DIR/train/llama31_lora.telemetry.csv" -- \
      "$PY" "$BENCH_DIR/shared/train/sft_lora.py" \
        --model meta-llama/Llama-3.1-8B-Instruct --tag llama31_lora \
        --out "$RESULTS_DIR/train/llama31_lora.json" \
      > "$RESULTS_DIR/train/llama31_lora.log" 2>&1 \
      || echo "  llama31 lora FAILED (see train/llama31_lora.log)"
  fi

  # -------------------------------------------------- lane 5: qwen3 8B SFT
  step "lane 5: LoRA SFT Qwen3 8B (secondary)"
  if have "$RESULTS_DIR/train/qwen3_lora.json"; then
    echo "skip qwen3 lora (exists)"
  else
    "$PY" "$TELEM" --out "$RESULTS_DIR/train/qwen3_lora.telemetry.csv" -- \
      "$PY" "$BENCH_DIR/shared/train/sft_lora.py" \
        --model Qwen/Qwen3-8B --tag qwen3_lora \
        --out "$RESULTS_DIR/train/qwen3_lora.json" \
      > "$RESULTS_DIR/train/qwen3_lora.log" 2>&1 \
      || echo "  qwen3 lora FAILED (see train/qwen3_lora.log)"
  fi

  # ------------------------------------------------------ lane 6: merge
  step "lane 6: merge Llama adapter -> S3"
  if have "$RESULTS_DIR/train/merge_llama31.json"; then
    echo "skip merge (exists)"
  else
    "$PY" "$BENCH_DIR/shared/train/merge_adapter.py" \
      --base meta-llama/Llama-3.1-8B-Instruct \
      --adapter "$MODELS_DIR/adapters/llama31_lora" \
      --out "$RESULTS_DIR/train/merge_llama31.json" \
      > "$RESULTS_DIR/train/merge_llama31.log" 2>&1 \
      || echo "  merge FAILED (see train/merge_llama31.log)"
  fi

elif [ "$BOX" = "inf2" ]; then
  # =========================================================== INFERENCE BOX

  # ------------------------------------------------- lane 2: plumbing smoke
  step "lane 2: serving smoke (TinyLlama-1.1B, one completion)"
  if have "$RESULTS_DIR/serve/smoke_tinyllama_smoke/isl128_osl128_c1.json"; then
    echo "skip smoke (exists)"
  else
    bash "$BENCH_DIR/shared/serve/bench_serve.sh" smoke_tinyllama smoke \
      > "$RESULTS_DIR/serve/smoke_tinyllama.log" 2>&1 \
      || echo "  smoke FAILED (see serve/smoke_tinyllama.log)"
  fi

  # ------------------------------------- lanes 3+4: llama 3.1 8B base sweeps
  # Two server configs per model (KV preallocation makes one config unable to
  # cover both short-high-concurrency and long-context -- see METHODOLOGY).
  # Cold-boot wall time on the first launch of each config is itself a result.
  step "lane 3: sweep Llama 3.1 8B base (config A: short)"
  if have "$RESULTS_DIR/serve/llama31_base_short/grid.json"; then
    echo "skip llama31 base short (complete)"
  else
    bash "$BENCH_DIR/shared/serve/bench_serve.sh" llama31_base short \
      > "$RESULTS_DIR/serve/llama31_base_short.log" 2>&1 \
      || echo "  llama31 base short FAILED"
  fi

  step "lane 4: sweep Llama 3.1 8B base (config B: long)"
  if have "$RESULTS_DIR/serve/llama31_base_long/grid.json"; then
    echo "skip llama31 base long (complete)"
  else
    bash "$BENCH_DIR/shared/serve/bench_serve.sh" llama31_base long \
      > "$RESULTS_DIR/serve/llama31_base_long.log" 2>&1 \
      || echo "  llama31 base long FAILED"
  fi

  # ----------------------------------------------------- lane 5: sustained
  step "lane 5: sustained 30 min (config A, concurrency 8)"
  if have "$RESULTS_DIR/sustained/iter_001.json"; then
    echo "skip sustained (exists)"
  else
    bash "$BENCH_DIR/shared/serve/sustained.sh" llama31_base 30 \
      > "$RESULTS_DIR/sustained/run.log" 2>&1 || echo "  sustained FAILED"
  fi
  "$PY" "$BENCH_DIR/shared/parse_sustained.py" \
    --dir "$RESULTS_DIR/sustained" --out "$RESULTS_DIR/sustained/sustained.json" \
    >> "$RESULTS_DIR/sustained/run.log" 2>&1 || true

  # ------------------------------------------------------- lane 6: quality
  step "lane 6: quality (greedy determinism + logprobs)"
  bash "$BENCH_DIR/shared/quality/run_quality.sh" llama31_base \
    > "$RESULTS_DIR/quality/llama31_base.log" 2>&1 || echo "  quality FAILED"

  # ------------------------------------- lane 7: the trn1-trained fine-tune
  # The end-to-end story: weights that were LoRA-trained on the Trainium box,
  # merged, pushed to S3, pulled here, compiled and served. Short config only
  # (declared in its grid.json).
  step "lane 7: sweep + quality for the trn1 fine-tune"
  if have "$RESULTS_DIR/serve/llama31_dolly_short/grid.json"; then
    echo "skip fine-tune sweep (complete)"
  else
    bash "$BENCH_DIR/shared/serve/bench_serve.sh" llama31_dolly short \
      > "$RESULTS_DIR/serve/llama31_dolly_short.log" 2>&1 \
      || echo "  fine-tune sweep FAILED"
  fi
  bash "$BENCH_DIR/shared/quality/run_quality.sh" llama31_dolly \
    > "$RESULTS_DIR/quality/llama31_dolly.log" 2>&1 || echo "  quality FAILED"

  # ------------------------------------------------ lane 8: qwen3 attempt
  # Attempt-only: if the vLLM Neuron backend cannot boot Qwen3 8B, the
  # structured failure record IS the result (declared exclusion, not a gap).
  step "lane 8: Qwen3 8B serve attempt"
  if have "$RESULTS_DIR/serve/qwen3_base_short/grid.json" \
     || have "$RESULTS_DIR/serve/qwen3_base_short/load_failure.json"; then
    echo "skip qwen3 attempt (outcome recorded)"
  else
    bash "$BENCH_DIR/shared/serve/bench_serve.sh" qwen3_base short \
      > "$RESULTS_DIR/serve/qwen3_base_short.log" 2>&1 \
      || echo "  qwen3 attempt did not complete (recorded)"
  fi

else
  echo "unknown BOX='$BOX'" >&2; exit 2
fi

step "SUITE COMPLETE ($BOX)"
ls -R "$RESULTS_DIR" | head -60

# Evidence leaves the box the moment the suite ends -- a results push mid-run
# once raced a training lane and the finished JSON lost (qwen3_lora.json,
# 2026-07-30, retrained). The driver itself pushes now, win or lose.
bash "$BENCH_DIR/shared/bin/push_results.sh" "$BOX" || echo "  results push FAILED (push manually)"
