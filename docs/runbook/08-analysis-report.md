# 08 — analysis and report

```bash
cd ~/neuron-pipelines
make pull-results
python3 analysis/make_report.py        # exit 0 = suite complete; INCOMPLETE lists gaps
python3 demo/headline.py               # eyeball before writing prose
```

REPORT.md rules: every number from comparison.json; a "Corrections made
during this study" section listing anything that changed a previously
recorded number; declared exclusions get their own table (Qwen3 serve
outcome lives there if lane 8 recorded a failure).
