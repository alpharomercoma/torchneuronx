#!/usr/bin/env bash
# SHA-256 every weight file both GPUs will execute.
#
# The whole comparison rests on both machines running the same weights. This
# turns that from an assumption into a check: run it on both boxes, diff the
# output, and the report can state byte-identical weights as a fact.
#
# Only files that actually determine the run are hashed -- safetensors, config,
# tokenizer. The Llama repos also carry an original/*.pth duplicate of the same
# weights which vLLM never loads and which we exclude at download time, so it
# is deliberately not hashed and its presence or absence on either box is
# irrelevant to the result.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
HUB="$MODELS_DIR/hf/hub"
OUT="${1:-model_hashes.txt}"

{
  echo "# host: $(hostname)"
  echo "# captured: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "#"
  for repo in "$HUB"/models--*; do
    [ -d "$repo" ] || continue
    name=$(basename "$repo" | sed 's/^models--//; s/--/\//g')
    rev=$(cat "$repo"/refs/main 2>/dev/null || echo "unknown")
    echo "## $name @ $rev"
    snap="$repo/snapshots/$rev"
    [ -d "$snap" ] || { echo "   (no snapshot)"; continue; }
    find -L "$snap" -maxdepth 1 -type f \
         \( -name '*.safetensors' -o -name 'config.json' \
            -o -name 'tokenizer.json' -o -name 'generation_config.json' \) \
      | sort | while read -r f; do
        printf "%s  %s\n" "$(sha256sum "$f" | cut -c1-64)" "$(basename "$f")"
      done
  done
} | tee "$OUT"

echo "wrote $OUT" >&2
