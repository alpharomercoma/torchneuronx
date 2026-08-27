# extras — the extension lanes

Everything past the core train→merge→serve pipeline. The lanes here produced
most of Phases 2 through 5; their findings and numbers are in
[REPORT-EXTENSIONS.md](../REPORT-EXTENSIONS.md), and their execution order is
in runbooks [11](../docs/runbook/11-extensions.md),
[12](../docs/runbook/12-trainium2.md) and
[13](../docs/runbook/13-phase4-posttraining.md).

Results land as the usual json + telemetry.csv + log triplets under
`<box>/results/extras/`, with two exceptions that have their own trees:
`inf2/results/rag/` and `inf2/results/serve/` for anything that boots a
server.

## Layout

| group | files | what it covers |
|---|---|---|
| **drivers** | `run_phase2_{trn1,inf2}.sh`, `run_phase3_trn2.sh`, `run_phase4_trn1.sh`, `run_extras_{trn1,inf2,trn2}.sh`, and the round/gapfill/symmetry variants | disconnect-proof masters. Each is `have()`-resumable: rerunning skips recorded lanes, so a master is safe to relaunch after any interruption |
| **accuracy** (Phase 5) | `run_accuracy{,_full}.sh`, `asr_wer_lane.py`, `zeroshot_imagenet_lane.py`, `clip_nan_bisect.py`, `ACCURACY-REVIEW.md` | is the fast graph also *correct*? Paired Neuron-vs-CPU lanes under an MLPerf-style ≥99%-of-reference gate |
| **mlx parity** (Phase 2 Track A) | `whisper_lane.py`, `clip_lane.py`, `siglip_lane.py` | the [mlx-models](https://github.com/alpharomercoma/mlx-models) demos from an Apple M5, ported to one inf2.xlarge |
| **serving shape** | `knee_search.sh`, `bisect_ctx.sh`, `run_batch_ladder.sh`, `run_prefill_decode_inf2.sh`, `multitenant.sh` | where throughput knees, where context dies, how prefill and decode split |
| **quantization** | `quant_lane.sh` | bf16 vs fp8 KV vs int8 weights |
| **speculative decoding** | `spec_decode.sh`, `draft_baseline.sh` | the k-ladder to 3.35× |
| **RAG** | `rag/` | the whole agentic-RAG appliance on one inf2 — its own README |
| **NKI** | `nki_softmax.py` | a hand-written kernel, simulate and device modes |
| **trn2 window** | `tp_probe_trn2.sh`, `run_{frontier,maxutil,opportunistic}_trn2.sh`, `trn2_deadline_push.sh` | the Trainium2 lane. `trn2_deadline_push.sh` is invoked at boot by `cdk/user_data/trn2_autorun.sh` |
| **serve one-offs** | `run_serve_trn2_finetune.sh`, `run_trn1_rerun.sh` | serving the trn2 fine-tune back on inf2 (§28), and the trn1 re-run that produced the symmetry pass (§26) |
| **support** | `lib/common.sh`, `stage_data.sh`, `setup_inf2_overlay.sh`, `run_quality_gate.sh`, `run_dataloader_isolation.sh` | shared driver helpers and one-purpose utilities |

`lib/common.sh` is the single source of driver helpers — `have()`, `step()`,
`np_recorded`, `np_wait_for_chip`. `tests/test_driver_hygiene.py` enforces that
drivers use it rather than redeclaring their own, because five different answers
to "is this lane recorded?" caused real defects.

The capacity-hunting and preservation scripts that used to live here are in
[`ops/capacity/`](../ops/README.md); the frozen one-off drivers from the trn2
window are in `ops/attic/`.

---

## Phase-2 Track A: mlx parity (inf2)

The inference demos from
[mlx-models](https://github.com/alpharomercoma/mlx-models) (Apple M5), ported
to one inf2.xlarge — same inputs, same expected outputs, different silicon —
plus the Mistral-7B-Instruct vLLM serve attempt (A4). Lanes run as
optimum-neuron 0.4.3 traced models inside the vLLM DLAMI venv.

| file | what it does |
|---|---|
| `run_extras_inf2.sh` | driver: idempotent optimum-neuron install, 3 lanes telemetry-wrapped, mistral7b serve sweep, push |
| `whisper_lane.py` | Whisper-small ASR: compile-if-absent, transcribe the JFK clip (mlx `3_whisper` parity), RTF |
| `clip_lane.py` | CLIP ViT-B/32 zero-shot on the COCO cats image (mlx `4_clip` parity), images/s @ batch 1 |
| `siglip_lane.py` | SigLIP export ATTEMPT — expected structured exclusion (no siglip exporter in optimum-neuron 0.4.3) |

## Run (on the inf2 box)

```bash
bash extras/run_extras_inf2.sh      # all lanes, resumable; FORCE=1 redoes

# single lanes (venv python, one NeuronCore each):
python3 extras/whisper_lane.py --out /tmp/whisper.json
# expect (M5 reference, mlx 3_whisper README): "And so my fellow Americans
# ask not what your country can do for you ask what you can do for your
# country." -- transcript_matches_reference_head: true, rtf < 1

python3 extras/clip_lane.py --out /tmp/clip.json
# expect (M5 reference, mlx 4_clip README): "two cats lying on a couch"
# at ~100% softmax, all other labels ~0% -- plus images/s @ b1 (100 iters)

python3 extras/siglip_lane.py --out /tmp/siglip.json
# expect: {"status": "export_unsupported", "reason": <verbatim>, ...}, exit 0
# (on the M5 the same model answers "two cats lying on a couch" at ~89%
#  sigmoid, per the mlx 5_siglip README -- that asymmetry is the finding)
```

## Port notes (M5/MLX → Inferentia2/optimum-neuron)

- **Static vs dynamic shapes.** MLX JIT-compiles per shape at run time;
  optimum-neuron traces ONE static graph ahead of time (whisper: batch 1,
  decoder seq 128; clip: 5 texts × 1 image × 77 tokens × 224 px). Inputs are
  padded up to the compiled shape, never resized down. The compile is a
  cached first-class artifact under `/opt/np/models/neuron-compiled/`;
  `compile_s` = 0 on a cache hit.
- **inf2 ≡ trn1 compile target.** optimum-neuron's `INSTANCE_VALUE_MAP`
  normalizes both to the same `trn1` target, so artifacts cross-compiled on
  the trn1 box run unchanged on inf2:
  `optimum-cli export neuron --model openai/whisper-small
  --task automatic-speech-recognition --batch_size 1 --sequence_length 128
  --auto_cast all --auto_cast_type bf16 --instance_type inf2
  --disable-validation /opt/np/models/neuron-compiled/whisper-small`
- **Whisper decodes without KV cache** on Neuron (0.4.3 forces
  `use_cache=False`), so RTF against the 11 s clip — not tokens/s — is the
  honest headline metric.
- **CLIP images/s is the full traced graph** (1 image + 5 label texts per
  forward), not an image-encoder-only number; honest but conservative next
  to MLX's per-part timings.
- **SigLIP is the declared gap:** mlx `5_siglip` runs
  `google/siglip-base-patch16-224` on the M5; optimum-neuron 0.4.3 registers
  no siglip exporter config, so the lane records the verbatim failure and
  exits 0 — an ecosystem finding, not an error.
- **Mistral-7B rides the existing serve harness**
  (`bench_serve.sh mistral7b short`): new Tier-1 arch on this box, so the
  first boot is a cold neuronx-cc compile (~40 min); boot / load_failure /
  grid receipts land under `serve/mistral7b_short/`.

Local tests (pure, no Neuron/torch needed):
`uv run --with pytest python -m pytest tests/test_extras.py -q`
