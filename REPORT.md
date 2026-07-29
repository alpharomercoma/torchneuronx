# Report

Status: **infrastructure and harness complete; lanes not yet executed.**

Every number in this report is regenerated from `analysis/comparison.json`
(`make report`). Nothing below is hand-computed. Sections appear as their
lanes land:

1. Summary
2. Machines (verified state, package versions)
3. Compile costs (precompile wall, cold vs warm boot)
4. Training — Llama 3.1 8B LoRA (step time, tokens/s, MFU, loss trace)
5. Training — Qwen3 8B LoRA
6. Serving — Llama 3.1 8B base (config A + B sweeps)
7. Serving — the trn1 fine-tune (the end-to-end result)
8. Sustained (30 min retention)
9. Quality (greedy determinism, logprobs; base vs fine-tune delta)
10. Declared exclusions (incl. the Qwen3 serve attempt outcome)
11. Host CPU context
12. Corrections made during this study
13. Limits

*(placeholder — populated by runbook 08)*
