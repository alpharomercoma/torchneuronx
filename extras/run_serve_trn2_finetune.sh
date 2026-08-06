#!/usr/bin/env bash
# SERVE THE TRAINIUM2 FINE-TUNE ON INFERENTIA2 -- closes the train->serve loop
# for the newer chip.
#
#   bash extras/run_serve_trn2_finetune.sh
#
# WHY THIS IS THE LAST STRUCTURAL GAP
# -----------------------------------
# Phase 2 trained on Trainium1, merged the adapter, pushed it to S3, pulled it
# onto Inferentia2 and served it -- with sha256 verification at both ends so
# "the served weights are the trained weights" was a check, not an assumption.
#
# Phase 3 trained the SAME model on Trainium2 and never served it. Every
# serving number in this study therefore describes weights trained on trn1.
# Until this lane runs, the study cannot say that a Trainium2 fine-tune is
# servable at all, let alone that it serves equivalently.
#
# WHAT MAKES IT NON-TRIVIAL
# -------------------------
# Both training boxes merged to the SAME S3 path, so trn2's weights overwrote
# trn1's. The two merges genuinely differ (4 of 6 files; different final loss).
# Both were recovered from S3 versioning into box-specific prefixes, and
# launch_vllm.sh now has an unambiguous key for each. This lane pulls the trn2
# prefix explicitly.
set -uo pipefail

BENCH_DIR="${BENCH_DIR:-/opt/np/repo}"
cd "$BENCH_DIR"
export RESULTS_DIR="${RESULTS_DIR:-$BENCH_DIR/inf2/results}"
S3="s3://neuron-pipelines-artifacts-600627330911"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] serve_trn2: $*"; }

# ---- wait for the S3 copy of the trn2 weights to complete -------------------
# The copies were made server-side from versioned objects and a 5 GB shard is
# not instantaneous. Serving a half-copied model would fail in a way that looks
# like a model problem rather than a transfer problem.
NEED=4
log "waiting for all $NEED trn2 safetensors shards in S3"
for i in $(seq 1 60); do
  n=$(aws s3 ls "$S3/artifacts/llama31-8b-dolly-merged-trn2/" --region us-west-2 2>/dev/null | grep -c "safetensors$")
  [ "${n:-0}" -ge "$NEED" ] && { log "all $n shards present"; break; }
  log "  $n/$NEED shards, waiting"
  sleep 30
done
n=$(aws s3 ls "$S3/artifacts/llama31-8b-dolly-merged-trn2/" --region us-west-2 2>/dev/null | grep -c "safetensors$")
[ "${n:-0}" -ge "$NEED" ] || { log "FATAL: only $n/$NEED shards after 30 min"; exit 3; }

mkdir -p "$RESULTS_DIR/serve"

# ---- verify the pulled weights ARE the trained weights ----------------------
# The whole point of the sha256 round-trip. Doing it BEFORE the grid means a
# mismatch is reported as a provenance failure rather than showing up as odd
# serving numbers nobody can explain.
log "pulling and verifying the trn2 merged model"
M=/opt/np/models/llama31-8b-dolly-merged-trn2
mkdir -p "$M"
aws s3 sync "$S3/artifacts/llama31-8b-dolly-merged-trn2/" "$M/" --region us-west-2 --quiet || {
  log "S3 pull FAILED"; exit 3; }

python3 - "$M" "$BENCH_DIR/trn2/results/train/merge_llama31.json" <<'PYEOF' > "$RESULTS_DIR/serve/trn2_weights_verify.json" || log "verification reported a mismatch"
import hashlib, json, pathlib, sys
mdir, mjson = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
expected = (json.loads(mjson.read_text()) or {}).get("files", {})
out = {"model_dir": str(mdir), "expected_from": str(mjson), "files": {}, "all_match": True}
for name, want in sorted(expected.items()):
    f = mdir / name
    if not f.is_file():
        out["files"][name] = {"status": "MISSING"}; out["all_match"] = False; continue
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""): h.update(chunk)
    got = h.hexdigest()
    ok = got == want
    out["files"][name] = {"expected": want, "got": got, "match": ok}
    if not ok: out["all_match"] = False
out["note"] = ("sha256 of the weights actually on the serving box against the digests "
               "recorded by merge_adapter.py on the Trainium2 box. This is what makes "
               "'the served weights are the trained weights' a check rather than a claim.")
print(json.dumps(out, indent=2))
PYEOF
python3 -c "
import json;d=json.load(open('$RESULTS_DIR/serve/trn2_weights_verify.json'))
print('  all_match =', d.get('all_match'))" || true

# ---- the grid, identical to the trn1 fine-tune lane -------------------------
log "serving grid: llama31_dolly_trn2 short (isl1024 osl1024, conc 1/4/8/16/32)"
bash "$BENCH_DIR/shared/serve/bench_serve.sh" llama31_dolly_trn2 short \
  > "$RESULTS_DIR/serve/llama31_dolly_trn2_short.log" 2>&1 \
  || log "grid ended early (receipts on disk)"

log "COMPLETE"
bash shared/bin/push_results.sh inf2 || log "push FAILED"
