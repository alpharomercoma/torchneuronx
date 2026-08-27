# 08 — analysis and report

```bash
cd ~/neuron-pipelines
make pull-results-all                  # NOT `make pull-results` -- see below
python3 analysis/make_report.py        # exit 0 = suite complete; INCOMPLETE lists gaps
python3 demo/headline.py               # eyeball before writing prose
```

> **Use `pull-results-all`.** `make pull-results` mirrors only
> `results/{trn1,trn2,inf2}/`. The on-box drivers also wrote to per-lane and
> per-hostname prefixes — `results/trn1-specdec/`,
> `results/final-ip-172-31-20-190-specdec/` and thirteen more. **736 objects
> lived only in those**, including the k=8 and k=16 arms of the
> speculative-decoding ladder and the int8 perplexity receipts. The canonical
> three prefixes are not complete; see REPORT-EXTENSIONS §39.

The results are committed, so on a fresh clone `python3
analysis/make_report.py` alone reproduces every table with no AWS account.

REPORT.md rules: every number from comparison.json; a "Corrections made
during this study" section listing anything that changed a previously
recorded number; declared exclusions get their own table (Qwen3 serve
outcome lives there if lane 8 recorded a failure).
