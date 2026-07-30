#!/usr/bin/env bash
# Track B1: bisect the NCC_INLA001 long-context cliff on inf2.
# Known: 2048 boots, 9216/10240 crash. Probes walk between; each probe's
# outcome (healthy | compiler-crash | other) is its own receipt.
set -uo pipefail
BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${RESULTS_DIR:-$BENCH_DIR/inf2/results}/extras/ctx_bisect"
LAUNCH="$BENCH_DIR/shared/serve/launch_vllm.sh"
mkdir -p "$OUT"
for LEN in ${BISECT_LENS:-4096 6144 8192}; do
  R="$OUT/len_${LEN}.json"
  [ -s "$R" ] && { echo "skip $LEN (recorded)"; continue; }
  echo "############ bisect: max_model_len=$LEN ############"
  B="$OUT/len_${LEN}.boot.json"
  if MAX_MODEL_LEN_OVERRIDE="$LEN" MAX_NUM_SEQS_OVERRIDE=8 \
     bash "$LAUNCH" llama31_base short "$B" > "$OUT/len_${LEN}.log" 2>&1; then
    STATUS=healthy
  else
    grep -q "NCC_INLA001" "${B%.json}.server.log" 2>/dev/null \
      && STATUS=compiler_crash_NCC_INLA001 || STATUS=failed_other
  fi
  bash "$LAUNCH" stop "$B" >/dev/null 2>&1 || true
  printf '{\n "max_model_len": %s, "max_num_seqs": 8, "status": "%s",\n "boot_json": "len_%s.boot.json", "captured": "%s"\n}\n' \
    "$LEN" "$STATUS" "$LEN" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$R"
  echo "  $LEN -> $STATUS"
done
echo "bisect complete: $(grep -h status "$OUT"/len_*.json | tr -d ' ,"')"
