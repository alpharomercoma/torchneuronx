# 11 — Phase-2 extension lanes

Everything here assumes runbooks 00–10 ran once (stacks exist, caches warm).
All Phase-2 lanes are orchestrated by two disconnect-proof masters — one
`nohup` each survives any operator disconnect:

```bash
# on trn1 (NKI, training ctx ladder, checkpoint timing, whisper/clip/siglip):
setsid nohup bash extras/run_phase2_trn1.sh >> /opt/np/phase2_trn1.log 2>&1 &

# on inf2 (mistral, knee, bisect, quant, RAG, spec-decode, multi-tenant):
setsid nohup bash extras/run_phase2_inf2.sh  >> /opt/np/phase2_inf2.log 2>&1 &
```

Reattach from any machine with SSM access:

```bash
bash shared/bin/phase2_status.sh          # tails both logs + lists results
```

Completion markers: `PHASE2 TRN1 ALL COMPLETE` / `PHASE2 INF2 ALL COMPLETE`
(retry rounds append `ROUND<N> COMPLETE`). Every lane is `have()`-resumable:
rerunning a master skips recorded lanes, so the masters are safe to relaunch
after any interruption.

## Individual lanes (each standalone + resumable)

| lane | command (on the box) | results |
|---|---|---|
| A parity | inside `run_extras_trn1.sh` | `trn1/results/extras/{whisper,clip,siglip}*` |
| A4 Mistral | `bash shared/serve/bench_serve.sh mistral7b short` | `inf2/results/extras/serve/mistral7b_short/` |
| D knee | `bash extras/knee_search.sh` (`KNEE_RATES`, `KNEE_DURATION_S`) | `inf2/results/extras/knee/` |
| B1 bisect | `BISECT_LENS="3072" bash extras/bisect_ctx.sh` | `inf2/results/extras/ctx_bisect/` |
| B2 quant | `bash extras/quant_lane.sh` | `inf2/results/extras/quant/` |
| B4 ctx ladder | inside `run_extras_trn1.sh` | `trn1/results/extras/ctx_*.json` |
| C1 spec-decode | `bash extras/spec_decode.sh` | `inf2/results/extras/spec_decode/` |
| C2 multi-tenant | `bash extras/multitenant.sh` | `inf2/results/extras/multitenant/` |
| C3 cold-start | destroy+deploy InferentiaStack, run the boot-chain script | `inf2/results/extras/c3_cold_start.json` |
| C4 ckpt timing | inside `run_extras_trn1.sh` | `trn1/results/extras/ckpt_timing.json` |
| E NKI | `python3 extras/nki_softmax.py --mode simulate\|device` | `trn1/results/extras/nki_*.json` |
| F RAG | `bash extras/rag/run_rag.sh` | `inf2/results/rag/` |

## The three environment rules (violating any one wastes a full pass)

1. **PATH must contain the venv bin** before any Neuron process starts
   (`libneuronpjrt-path`). `launch_vllm.sh` and every extras driver export
   it; new lane scripts must too.
2. **Never install optimum-neuron into the vLLM venv.** RAG's optimum stages
   run as `PYTHONPATH=<overlay site-packages> <vllm-venv>/bin/python` via
   `extras/rag/setup_venv.sh`; server boots strip PYTHONPATH
   (`env -u PYTHONPATH`).
3. **Export `HF_HOME=/opt/np/models/hf` and run `shared/bin/hf_login.sh`**
   in any shell that touches gated models — bare SSM shells have neither.

## What came after

Phase 2 is where the extension lanes start, not where they end. The later
phases have their own pages:

| phase | page | what it added |
|---|---|---|
| 3 | [12-trainium2.md](12-trainium2.md) | the trn2.3xlarge lane in sa-east-1 and the one-chip-each generational comparison |
| 4 | [13-phase4-posttraining.md](13-phase4-posttraining.md) | ORPO, DPO and GRPO — one works, one is terminal, one is architecturally blocked |
| 5 | REPORT-EXTENSIONS §35-39 | accuracy-as-validity, Spec-Bench, the Tulu-3 replication, and the S3 near-miss |

Findings and numbers: [REPORT-EXTENSIONS.md](../../REPORT-EXTENSIONS.md).
