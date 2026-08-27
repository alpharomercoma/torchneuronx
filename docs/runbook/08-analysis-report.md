# 08 — analysis and report

```bash
cd ~/neuron-pipelines
make pull-results                      # -> trn1/results/, trn2/results/, inf2/results/
make pull-results-all                  # -> .s3-mirror/results/  (reconcile by hand)
python3 analysis/make_report.py        # exit 0 = suite complete; INCOMPLETE lists gaps
python3 demo/headline.py               # eyeball before writing prose
```

> **Run both, and know what each one does.** `make pull-results` syncs the
> three canonical prefixes straight into `<box>/results/`, which is the only
> place `make_report.py` reads. `make pull-results-all` mirrors *every*
> `results/` prefix into `.s3-mirror/results/` — it does **not** feed the
> analyzer, and nothing downstream picks it up automatically.
>
> That second target exists because the canonical three are not complete. The
> on-box drivers also wrote to per-lane and per-hostname prefixes —
> `results/trn1-specdec/`, `results/final-ip-172-31-20-190-specdec/`,
> `results/trn1-ppl/` and thirteen more. **736 objects lived only in those**,
> including the k=8 and k=16 arms of the speculative-decoding ladder and the
> int8 perplexity receipts. Diff `.s3-mirror/` against the box directories and
> copy across what is missing, by hand. See REPORT-EXTENSIONS §39.

The results are committed, so on a fresh clone `python3
analysis/make_report.py` alone reproduces every table with no AWS account.

REPORT.md rules: every number from comparison.json; a "Corrections made
during this study" section listing anything that changed a previously
recorded number; declared exclusions get their own table (Qwen3 serve
outcome lives there if lane 8 recorded a failure).
