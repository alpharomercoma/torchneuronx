# Pre-stage the two benchmark corpora. Pure I/O -- no accelerator, no model,
# so it is safe to run before any review lands and it takes the download off
# the critical path.
#
#   LibriSpeech dev-clean + dev-other  (MLPerf's "dev-all"), from openslr.org
#   ImageNet-1k val 50k                (clip-benchmark/wds_imagenet1k, 7 shards)
set -uo pipefail
REPO="${BENCH_DIR:-/opt/np/repo}"
source "$REPO/extras/lib/common.sh"
NP_TAG=stage

# Resolved, not hardcoded. The first version of this script pinned inf2's vLLM
# venv path and died on trn1 with "No such file or directory" -- the very bug
# np_pick_venv exists to end.
PY_VENV="$(np_pick_venv numpy)" || np_die "no venv on this box has numpy"
PY="$PY_VENV/bin/python"
export PATH="$PY_VENV/bin:$PATH"
export HF_HOME="${HF_HOME:-/opt/np/models/hf}"
export HF_TOKEN="${HF_TOKEN:-$(aws ssm get-parameter --name /neuron-pipelines/hf-token \
  --with-decryption --query Parameter.Value --output text --region us-east-1 2>/dev/null)}"
np_log "staging with $PY"

np_step "LibriSpeech dev-clean + dev-other (MLPerf dev-all)"
"$PY" - <<PY
import sys; sys.path.insert(0, "$REPO/extras")
import asr_wer_lane as L
for split in L.MLPERF_SPLITS:
    root = L.ensure_split("/opt/np/models/librispeech", split)
    print(f"{split}: {len(L.index_split(root, split))} utterances at {root}", flush=True)
PY

np_step "ImageNet-1k val (clip-benchmark/wds_imagenet1k, all 7 shards)"
"$PY" - <<PY
import sys, os; sys.path.insert(0, "$REPO/extras")
import zeroshot_imagenet_lane as Z
root = Z.ensure_dataset("/opt/np/models/imagenet1k", keep_tars=False)
names, tmpl = Z.load_prompts("/opt/np/models/imagenet1k", 0)
counts = [len(os.listdir(os.path.join(root, f"{c:04d}"))) for c in range(1000)]
# min/class is the integrity check: the shards are class-ordered, so a partial
# extraction shows up here as a class with too few images rather than as a
# quietly smaller benchmark.
print(f"classes={len(counts)} total={sum(counts)} "
      f"min/class={min(counts)} max/class={max(counts)}")
print(f"classnames={len(names)} templates={len(tmpl)}")
PY
np_log "STAGE COMPLETE"
df -h /
