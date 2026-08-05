#!/usr/bin/env bash
# Re-run the trn1 primary lane on TODAY's software stack, and repeat it.
#
# TWO THREATS TO VALIDITY THIS CLOSES
# -----------------------------------
# 1. VERSION CONFOUND. The published trn1 numbers were captured 2026-07-29:
#
#      torch-neuronx 2.9.0.2.15.32035   neuronx-cc 2.26.6360.0
#      optimum-neuron 0.4.3             transformers 4.57.6
#
#    The trn2 box resolves the SAME SSM parameter -- but it points at
#    ".../latest/image_id", so a DLAMI published between those dates ships a
#    different compiler. Only optimum-neuron, trl and peft are pip-pinned;
#    torch-neuronx, neuronx-cc and transformers come from the image. A
#    compiler improvement would then show up as a Trainium2 hardware gain.
#    Re-running trn1 on today's stack turns that confound into a measurement:
#    rerun-vs-published on the SAME silicon isolates the software delta.
#
# 2. ONE-SIDED VARIANCE. Repeating trn2 at seeds 43/44 measures trn2 stability
#    only. A confidence interval on the trn2/trn1 RATIO needs both sides
#    repeated under the same protocol, or the interval is fiction. Same seeds
#    here as the trn2 opportunistic lanes, deliberately.
#
# WHAT THIS MUST NOT DO
# ---------------------
# Never overwrite trn1/results/train/llama31_lora.*. Those are published and
# other numbers are derived from them. Everything here lands in
# trn1/results/rerun/ under distinct tags. Config is otherwise IDENTICAL to the
# published lane -- seq 2048, micro-batch 1, grad-accum 8, dolly-15k, r16/a32 --
# because the whole point is that software is the only variable.
#
#   setsid nohup bash extras/run_trn1_rerun.sh >> /opt/np/trn1_rerun.log 2>&1 &
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_DIR"
OUT="$BENCH_DIR/trn1/results/rerun"
NP_VENV="${NP_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
export PATH="$NP_VENV/bin:$PATH"     # libneuronpjrt-path
PY="$NP_VENV/bin/python"
TELEM="$BENCH_DIR/shared/telemetry.py"
SFT="$BENCH_DIR/shared/train/sft_lora.py"

export NP_DEVICE=trn1
export NP_REGION="${NP_REGION:-us-west-2}"
export NP_CACHE_PREFIX="${NP_CACHE_PREFIX:-neuron-cache}"   # v2 NEFFs
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/opt/np/cache/neuron-compile-cache}"

mkdir -p "$OUT"
have() { [ "${FORCE:-0}" != "1" ] && [ -s "$1" ]; }
step() { echo; echo "############ TRN1-RERUN: $* ############"; echo; }

bash shared/bin/hf_login.sh >/dev/null 2>&1 || echo "WARN: hf_login failed"

# Record the stack we are ACTUALLY on, before anything runs. This file is the
# evidence for or against the version confound, and it is worth having even if
# every lane below fails.
step "capturing today's software stack"
"$PY" - "$OUT/versions_today.json" <<'EOF'
import importlib.metadata as md, json, platform, sys
pkgs = ["torch", "torch-neuronx", "torch-xla", "optimum-neuron", "transformers",
        "trl", "peft", "datasets", "accelerate", "neuronx-cc",
        "neuronx-distributed", "libneuronxla"]
out = {"python": platform.python_version()}
for p in pkgs:
    try:
        out[p] = md.version(p)
    except Exception:
        out[p] = None
json.dump(out, open(sys.argv[1], "w"), indent=2)
print(json.dumps(out, indent=2))
EOF

# The published run's versions, for a direct diff in the report.
"$PY" - "$BENCH_DIR/trn1/results/train/llama31_lora.json" "$OUT/versions_published.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
json.dump(d.get("versions") or {}, open(sys.argv[2], "w"), indent=2)
EOF

"$PY" - "$OUT/versions_published.json" "$OUT/versions_today.json" "$OUT/version_delta.json" <<'EOF'
import json, sys
was, now = (json.load(open(p)) for p in sys.argv[1:3])
delta = {k: {"published": was.get(k), "today": now.get(k)}
         for k in sorted(set(was) | set(now)) if was.get(k) != now.get(k)}
json.dump({"changed": delta, "identical": not delta}, open(sys.argv[3], "w"), indent=2)
print("VERSION DELTA:", json.dumps(delta, indent=2) if delta else "none -- stacks identical")
EOF

# rerun <tag> <extra args...> -- same config as the published lane by omission
rerun() {
  local tag="$1"; shift
  { have "$OUT/$tag.json" || have "$OUT/$tag.failure.json"; } && { echo "skip $tag"; return 0; }
  step "$tag"
  "$PY" "$TELEM" --out "$OUT/$tag.telemetry.csv" -- \
    "$PY" "$SFT" --model meta-llama/Llama-3.1-8B-Instruct --tag "$tag" \
      --out "$OUT/$tag.json" "$@" \
    > "$OUT/$tag.log" 2>&1 || {
      REASON=$(grep -m1 -oE "NCC_[A-Z0-9]+[^\"]{0,120}" "$OUT/$tag.log" | head -1 | sed "s/\"/'/g")
      [ -n "$REASON" ] || REASON=$(tail -c 300 "$OUT/$tag.log" | tr '\n' ' ' | sed "s/\"/'/g")
      printf '{"tag":"%s","status":"failed","reason":"%s","captured":"%s"}\n' \
        "$tag" "$REASON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$tag.failure.json"
      echo "  $tag FAILED (receipt)"; }
  bash shared/bin/push_results.sh trn1 >/dev/null 2>&1 || echo "  push FAILED"
}

# Seed 42 == the published configuration exactly. Any difference between this
# and llama31_lora.json is attributable to the software stack alone.
rerun rerun_seed42

# Seeds 43/44 mirror the trn2 opportunistic lanes so the ratio can carry an
# interval computed from both sides rather than one.
rerun rerun_seed43 --seed 43
rerun rerun_seed44 --seed 44

step "TRN1 RERUN COMPLETE"
bash shared/bin/sync_neuron_cache.sh push || echo "cache push FAILED"
bash shared/bin/push_results.sh trn1 || echo "push FAILED"
