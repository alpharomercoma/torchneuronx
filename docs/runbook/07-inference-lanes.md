# 07 — inference lanes (inf2)

```bash
cd /opt/np/repo && inf2/scripts/run_all.sh
# lane 3: config A cold boot -- the tens-of-minutes compile IS a result
# lane 4: config B (second compile)
# lane 5: 30 min sustained; lane 6 quality; lane 7 fine-tune; lane 8 qwen attempt
tail -f inf2/results/serve/llama31_base_short.log
```

Between lanes: `bash shared/bin/push_results.sh inf2` and cache push.
End: stop the instance (never terminate — the warm cache is the demo).
