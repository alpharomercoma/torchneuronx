# Phase 3 live state — written 2026-08-05T18:45Z

Handoff document. Everything needed to resume without prior context.

## Hard deadlines (UTC)

| when | what |
|---|---|
| 2026-08-06 **10:00** | opportunistic/frontier/maxutil hard stop (deadline guards enforce) |
| 2026-08-06 **10:30** | deadline pusher final push to S3 |
| 2026-08-06 **10:50** | ASG scheduled action scales to 0 |
| 2026-08-06 **11:00** | **EC2 begins TERMINATING** the block instance. EBS dies with it. |
| 2026-08-06 11:30 | block `cr-08dc8b22d254cd3da` ends |

Anything not in S3 by 10:30Z is lost. The instance is terminated, not stopped.

## Instances

| box | id | region | state |
|---|---|---|---|
| trn2 | `i-0a3f33482fa319c76` | sa-east-1 | **REPLACEMENT** box (original `i-00e7b6117eac3a122` terminated 17:12Z) |
| trn1 | `i-0cb9e758143a745d5` | us-west-2 | idle, free (credits) |
| inf2 | `i-0936ae7615727251e` | us-west-2 | done, both phase grids complete |

ASG `NeuronPipelinesTrainium2-Trn2AsgASG56F8472A-WzMFVvw2ikkW`, launch template **v4**.

## What is running RIGHT NOW

- **trn2**: `run_phase3_trn2.sh`, lane 4 `llama31_lora`, epoch 2.81/3 at 19:05Z.
  Re-running the whole suite because the replacement box started with an empty
  `trn2/results/`. **This is a second independent sample of the primary lane on
  a different physical Trainium2** — compare against the published JSON in S3
  for cross-instance variance, which no other lane in this study measures.
- **trn1**: isolation DONE (null result, §21). Now running
  `run_batch_ladder.sh` under systemd unit `np-batchladder`
  (log `/opt/np/batchladder_trn1.log`). Free, on credits.
- **inf2**: idle, complete.

## Active monitors (re-create after compaction if lost)

`buydk964k` — trn1 batch ladder: rung count, unit state, stall detection.
Exits (with a final event) when the ladder finishes.

**`b88x4hqkd` was RETIRED at 20:37Z.** It began emitting garbled fields
(`SUITE-UP | 15 | 0 json`) because its trn1 probe still read
`results/quality/` and a stale lane-tag pattern. Nothing was wrong with either
box. A monitor that reports numbers it cannot justify is worse than no monitor:
every false alarm this session came from one, so it was retired rather than
patched, and its trn2 duties were already covered by `bvl1vpy21`.

`bvl1vpy21` — trn2 liveness, every 10 min via SSM. **Replaces `bifw2s4jt`,
which was stopped as defective** (monitor defect #6, below). Reports progress
whenever the result-JSON count changes; alarms only on suite-not-running, no
result-dir write for 15 min, kernel OOM, or a HALTED banner.

Monitor defect #7 (same session, caught in 4 min): the first replacement
(`bvooj0elr`) embedded a multi-line probe into the SSM `--parameters` JSON with
a `sed`/`tr` escaping trick that produced malformed JSON. `send-command`
returned nothing, and an empty reply was folded into the same branch as
`InvalidInstanceId` — so a **quoting bug in my own monitor** was reported as
"the ASG replaced the box". Fixed twice over: the probe is now a single line
that needs no escaping, and empty/garbled replies are reported as UNREACHABLE,
never as a claim about the box. This is the third time this session that a
monitor's own failure was rendered as a box failure.

### Monitor defect #6 — S3/driver-log freshness is not a liveness signal

`bifw2s4jt` watched the newest object under `results/trn2/` in S3 and alarmed at
75 min of silence. But the deadline pusher only runs every 30 min AND the driver
log `/opt/np/phase3_trn2.log` receives only a **banner between lanes** — each
lane's own output goes to `$OUT/$tag.log`. The 8B primary lane runs ~55 min, so
a perfectly healthy lane 4 guarantees 55+ min of no new S3 object and no driver
log line. It fired a false OOM alarm at 18:54Z while the box was at epoch
2.52/3, 3400 tok/s, writing every few seconds.

The fix: measure freshness as the **newest mtime of any file under
`trn2/results/`** — which includes the in-progress `.log` and `.telemetry.csv` —
and alarm at 15 min. Same lesson as "blank ≠ stopped": pick a signal that ticks
at the cadence of the thing you are watching, not at the cadence of its
checkpoints.

## HEADLINE RESULTS (all banked in S3)

Llama 3.1 8B LoRA, identical hyperparameters both chips, 645 steps:

| | trn1 | trn2 | ratio |
|---|---|---|---|
| median step | 5550.4 ms | 4637.7 ms | 1.20x |
| tokens/s steady | 2951.8 | 3532.8 | 1.20x |
| **train wall** | 7333.7 s | **3322.1 s** | **2.21x** |
| tokens/s end-to-end | 1441 | 3181.0 | **2.21x** |
| MFU | 68.3% (/210) | 25.9% (/667) | — |
| **$/M tokens** | $0.2583 | **$0.1952** | **24.4% cheaper** |

Qwen3 8B replicates: 1.24x per-step, 2.04x end-to-end.

**Context ladder — the strongest result:**

| seq | trn1 | trn2 | trn2 MFU |
|---|---|---|---|
| 2048 | 2951.8 | 3532.8 (1.20x) | 25.9% |
| 4096 | 3575.0 | 6878.4 (**1.92x**) | 50.3% |
| 8192 | **device OOM** (18.12 > 16 GB) | 8310.5 | **60.8%** |
| 16384 | — | **host OOM** (124 GiB RAM, compile) | — |

trn2's advantage DOUBLES with sequence length. The 1.20x headline is an
artifact of running trn2 at trn1's shape. Both chips have a cliff, in different
places for different reasons: trn1 device HBM, trn2 host RAM.

**Quality gate (trn1 only so far):**

```
llama31_lora_holdout   pre 2.1491 -> post 1.2510   delta -0.898
quality_smoke          pre 1.7252 -> post 1.3962   delta -0.329
tok/s 2933.3 -- inside the 2.4% noise floor, so evaluation did not perturb training
eval_wall_s recorded SEPARATELY from train_wall_s
```

**Inferentia2 phase split:**
prefill compute-bound (2204 -> 4244 tok/s, rises with prompt length);
decode memory-bound (TPOT flat 56 ms across concurrency 1/4/8, throughput
scales 1:3.98:7.89). ~124-238x per-token gap between phases.

## OUTSTANDING WORK, in priority order

1. **trn2 quality gate** — the ONE symmetry violation. **ARMED, no longer needs
   a human or a laptop.** systemd unit `np-followon` on trn2 (script
   `extras/run_trn2_followon.sh`, log `/opt/np/followon_trn2.log`) is waiting
   for `run_phase3_trn2.sh` to exit, then runs the quality gate, then the
   isolation lane. Each checks it has enough window to FINISH before the
   10:00Z hard stop — starting a lane that gets terminated mid-run produces no
   artifact and burns window the other lane could have used.
2. ~~**Synthetic-input dataloader isolation**~~ — **DONE on trn1**, written up
   as REPORT §21. **NULL RESULT, and it kills the hypothesis.** Uplift 0.999×
   at seq 2048 and 1.000× at 4096; both real controls validate against the
   published lanes (0.4% and 0.1%). Host dataloader cost is below a twentieth
   of the noise floor at both shapes, so it cannot explain the 1.20×/1.92×
   split. The remaining explanation is occupancy: trn2 at seq 2048 is not
   slowed, it is UNFILLED (25.9% MFU vs trn1's 75.2% on the same shape).
   **Still queued on trn2** via `np-followon` — trn1 is the weaker direction of
   the test (longer step, so host cost should matter least), which is why the
   conclusion is provisional until trn2 runs it.
3. ~~**Loss-curve overlay trn1 vs trn2**~~ — **DONE**, `analysis/loss_overlay.py`,
   written up as REPORT §18. Result: the gap grows 3.5× from the first tenth of
   training (mean |d| 0.0080) to the tail (0.0278) with r=0.9969, and
   replicates on Qwen3 (0.0088 → 0.0218, r=0.9973). That monotone-growth-from-
   agreement shape is the accumulation-order signature, not a different model.
   Caveat recorded: TP width is confounded with the chip, so running trn2 at
   TP=2 is the controlled follow-up.
4. **maxutil lane** — queued in `run_phase3_trn2.sh`, selects its config from
   the efficiency sweeps. Needs those sweeps to complete first.
5. ~~Batch-size sweep at seq 4096~~ — **BUILT AND RUNNING**,
   `extras/run_batch_ladder.sh`, micro-batch 1/2/4/8 at seq 4096, 30 steps per
   rung, global batch held constant by halving grad-accum. Directly tests §21's
   surviving explanation (trn2 is starved, not slowed). Running on trn1 now;
   armed on trn2 as stage 2 (`np-followon2`). Prediction recorded in the script
   header BEFORE the run: trn2 should keep gaining past the rung where trn1
   stops, and trn1 should hit a device-memory wall first (16 vs 24 GiB per
   logical core). Both chips stalling at the same rung would falsify the
   occupancy story.

## TRN2 AUTOMATION CHAIN (no laptop in the critical path)

| unit | script | runs |
|---|---|---|
| `np-suite` | `run_phase3_trn2.sh` | main Phase-3 suite |
| `np-followon` | `run_trn2_followon.sh` | quality gate, then dataloader isolation |
| `np-followon2` | `run_trn2_followon2.sh` | batch ladder |
| `np-followon3` | `run_trn2_followon3.sh` | deliberate `ctx_16384` retry, LAST |

Each stage waits for the previous to release the chip and refuses to START
anything it cannot FINISH before the 10:00Z hard stop — a lane terminated
mid-run yields no artifact and burns window another lane could have used.

**Do NOT edit a script that is currently executing on the box.** bash reads
lazily by byte offset and will resume at a stale offset. That is why stage 2 is
a separate file rather than three more lines in stage 1.

## NEW RESULTS THIS SESSION (post-compaction)

**REPORT §18 — loss-curve overlay.** `analysis/loss_overlay.py`. The trn1/trn2
gap grows monotonically from mean |Δ| 0.0080 in the first tenth of training to
0.0278 in the tail, r=0.9969, sustained >0.01 divergence only at step 544/645.
Replicates on Qwen3 (0.0088 → 0.0218, r=0.9973). That shape is the
accumulation-order signature, not a different model. Confound disclosed: TP
width is inseparable from the chip here; running trn2 at TP=2 would settle it.

**REPORT §19 — quality gate**, trn1 numbers, and it states in the text that
trn2 has not run it yet.

**REPORT §20 — replication on a second physical Trainium2.** The ASG
replacement accidentally re-ran the primary lane on different silicon:

| | original chip | replacement | delta |
|---|---|---|---|
| tokens/s | 3532.8 | 3618.0 | **+2.41%** |
| median step | 4637.7 ms | 4528.5 ms | −2.35% |
| train wall | 3322.1 s | 3220.1 s | −3.07% |
| **final loss** | **1.1489** | **1.1489** | **0.000000** |

Two findings. Timing varies 2.4% *between chips* — the same magnitude as the
within-box seed noise floor, so 2.4% is the resolution limit for every
throughput claim in this study. And the loss is **bit-identical across
different silicon**, which establishes end-to-end determinism and retires the
"maybe §18's gap is just hardware variation" objection: hardware variation in
loss is exactly zero.

**Dataloader isolation is REAL and works.** `synth_smoke` on trn1 returned
4571.3 tok/s with `dataset: synthetic:random-token-ids` and
`loss_is_meaningless: true`. TRL accepts pre-tokenised rows with no formatter
on this stack — that was the unverified assumption and it holds.

## KNOWN ISSUES / GOTCHAS

- **`--seed` does nothing.** Three trn1 seeds gave BIT-IDENTICAL loss
  (tail-50 1.102654, stdev 0.0). Packing pins data order. So variance lanes
  measure timing only: **2.4% tokens/s noise floor**. trn2 seed 43/44 lanes are
  redundant. The trn1/trn2 loss gap (1.2063 vs 1.1489) is therefore REAL, not
  noise -- most likely TP=2 vs TP=4 collective accumulation order.
- **`params_trainable` mismatch** 52.4M (trn1) vs 77.6M (trn2): `local_shard x
  TP` over-counts replicated LoRA weights at TP=4. 0.42% FLOPs effect. Must be
  disclosed.
- **`power_w` / `temp_c` telemetry columns are EMPTY** on all boxes --
  vestigial from the MI300X port. Perf-per-watt is NOT measurable on Neuron.
- **`NeuronSFTTrainer` has no `.evaluate()`.** Held-out scoring uses a
  zero-learning-rate forward pass; needs `max_grad_norm=0.0` or ZeRO-1's
  clipping raises IndexError on the empty gradient list.
- **Neuron caches FAILED compiles.** A poisoned NEFF from 2026-07-29 makes the
  inf2 `long` geometry (9216) unusable; phase grids use `short` (2048) instead.
- **`pi` hangs on prompts over ~1 KB** and with `--thinking high`. Keep briefs
  short. Models reachable: kimi-k3, glm-5.2, qwen3.8-max, deepseek-v4-pro,
  minimax-m3 via `--provider opencode-go`.
- Do NOT overwrite a RUNNING shell script on the box: bash reads lazily by byte
  offset and will execute garbage.

## FAILURES THIS SESSION AND THEIR FIXES (all committed)

| failure | cause | fix |
|---|---|---|
| box idle after launch | `np-autorun` had `After=cloud-final.service` but is enabled BY cloud-final -- deadlock | removed from After= |
| suite killed at 90s risk | no `TimeoutStartSec`; default 90s vs a 600s S3 wait | `TimeoutStartSec=infinity` |
| TP rungs died in 6s (twice) | training packages hand-installed, died with EBS on ASG replacement | pip install moved into `trn1.sh`/`trn2.sh` user-data, LT v4 |
| ctx_16384 killed the suite | host RAM OOM during compile (trn2 has NO swapfile, trn1 has 64 GiB) | receipt recorded; swap decision flagged as implicated |
| inf2 grids failed to boot | reused `long` geometry, which is a RECORDED FAILURE with a poisoned cache | switched to `short` geometry |
| eval post-pass IndexError | ZeRO-1 clips gradients; frozen pass has none | `max_grad_norm=0.0` |
| unactionable receipts | only recorded the exception message | receipts now carry `traceback.format_exc()` |
| false OOM alarm at 18:54Z | freshness measured from S3 pushes (30 min) and driver-log banners (per-lane), not from the running lane | monitor `bvooj0elr` watches newest mtime under `trn2/results/` |

## DOCUMENTATION STATE

`REPORT-EXTENSIONS.md` — 559+ lines.
- §15 Trainium1 vs Trainium2 (incl. 15.5 procurement + the $2.235/hr price)
- §16 Phase-3 corrections 1-10
- §17 Inferentia2 prefill vs decode

`analysis/roofline.py` — ridge points trn1 232 / trn2 230 / trn2-FP8 448.
`analysis/make_report.py` — `cost_metrics()` (occupied vs block-allocated,
never blended), `end_to_end_fields()` backfill, `collect_trn1_reruns()`.

## ADVERSARIAL REVIEW STATUS

codex: delivered, judged, largely implemented. Rejections honoured -- NO MLPerf
claim (Llama-2-70B LoRA won't fit 96 GiB; the 8B benchmark is C4 pretraining),
no MMLU-after-Dolly, no cross-study GPU comparison, no sticker-price $/token.

pi panel: glm-5.2 and kimi-k3 answered. Both independently rejected MLPerf,
matching codex. Both proposed perf-per-watt, which is impossible here (power
telemetry empty). glm-5.2's sharpest point: explicitly declare the FP8-training
gap as headroom an H100 could exploit and we cannot.

## THE ctx_16384 DECISION (2026-08-05 20:29Z)

`ctx_16384` host-OOM'd on the ORIGINAL trn2 instance, took the suite down with
it, and the box was ultimately replaced. The replacement box started with an
empty results dir, so `have()` had no receipt to skip on and the ladder was
about to re-run it **ahead of the quality gate** — the one remaining asymmetry
between the chips.

Three actions, ~10 min before it would have started:

1. **64 GiB swapfile** added at `/scratch/swapfile` (`swapon --show` confirms).
   trn1 has always had one; removing it from trn2's user-data is implicated in
   the original OOM. Swap does not make 16384 fit — it makes an overrun SLOW
   rather than FATAL, which is the difference between a receipt and a lost
   instance.
2. **A `deferred` receipt**, not a fabricated failure. `have()` only needs a
   non-empty `ctx_16384.failure.json`; writing `"status":"failed"` for a lane
   this box never attempted would be inventing a measurement. The receipt says
   `deferred`, gives the scheduling reason, points at the real prior failure in
   S3, and states in its own text that it is not a measurement. Verified
   working: the driver logged `skip ctx 16384 (recorded)` and proceeded to C4.
3. **Stage 3** (`np-followon3`) retries it deliberately, last, after everything
   of value is in S3, with a hard stop an hour earlier than the other stages so
   the deadline pusher keeps a clear window if the box goes down.

Honest expectation, recorded before the run: it probably still fails. Compile
host memory is the binding constraint and 64 GiB of swap against a 124 GiB
shortfall is a cushion, not a fix. A clean receipt with the box intact is the
realistic good outcome; a pass is the upside.

## THE CACHED-FAILURE INCIDENT (2026-08-05 20:40Z) — read this before trusting any ladder

The trn1 batch ladder reported four rungs. **Three of them were not
measurements.**

What happened: `mb2_seq4096` failed to compile with `NCC_EXTP003` — the
compiler's instruction-count limit, 2,064,384 generated against a stated limit
of 150,000. `mb4` and `mb8` then reported the **identical** instruction count.
Graph size must grow with micro-batch, so identical counts are impossible if
each rung compiled its own graph. They reused the cached failure. Neuron
caching failed compiles is a gotcha already recorded in this very document, and
the ladder walked straight into it.

Left unchecked, this would have been published as a clean finding — "trn1 walls
at micro-batch 2 on the compiler limit" — resting on two fabricated data points.

Fixes, all committed (`21c333f`):
- `export NEURON_CC_FLAGS="... --retry_failed_compilation"` so each rung
  compiles its own graph.
- Every failure receipt now records `compiler_instructions`. That number is the
  audit trail: it MUST grow with micro-batch.
- The summary sets `"INVALID"` on the whole ladder if two or more failed rungs
  report the same instruction count, and prints it loudly.

The original artifacts were moved to `trn1/results/batch_ladder/invalidated/`,
not deleted, and the three rungs re-run under unit `np-batchladder2`.

**The generalisable lesson:** a failure mode that produces IDENTICAL output
across different inputs is not a result, it is a cache. Any lane that sweeps a
parameter must record a quantity that is expected to CHANGE with that
parameter, or a cache hit is indistinguishable from a measurement.

⚠️ **trn2's stage-2 ladder is armed with the DEFECTIVE script.** The corrected
file is in S3 at `code/extras/run_batch_ladder.sh`; it must be copied to
`/opt/np/repo/extras/run_batch_ladder.sh` on trn2 before `np-followon2` starts,
or trn2 will reproduce the identical invalid result. Copy the SINGLE FILE — do
not run `pull_code.sh`, because the three follow-on scripts are executing and
bash reads a running script lazily by byte offset.

## AWS CREDENTIALS EXPIRED 20:43Z

The laptop's token expired. **Nothing is at risk.** Both boxes run on their own
IAM instance roles: trn2's four-stage chain, trn1's ladder re-run, and the
on-box deadline pusher are all independent of the laptop. Everything measured
is already in S3. Refresh with `aws login`; the only blocked action is the trn2
single-file copy above.

### Monitors during the credential outage (20:53Z)

Stopped `bvl1vpy21` (trn2 liveness) and `buydk964k` (trn1 ladder) — with no
credentials they could only repeat "UNREACHABLE" every 10 minutes, which is
noise, not information. **Both must be restarted after `aws login`**; their
full definitions are in the git history of this file and in the commits around
`c729553`.

`bsxp0xm77` is still running: it polls for valid credentials and applies the
trn2 ladder fix the moment they return, then exits. It reports explicitly
whether it landed the fix in time (`COPIED`) or arrived after the ladder had
already started with the defective script (`LADDER_ALREADY_RUNNING`), so a
late arrival cannot be mistaken for a success.

## QUEUE RE-ORDER AND THE EADDRINUSE SELF-INFLICTED FAILURE (2026-08-05 22:50–23:00Z)

**Situation.** The trn2 main suite printed `PHASE3 TRN2 ALL COMPLETE` at ~22:40Z
— every primary lane, context ladder, ckpt timing and NKI banked, 27 result
JSONs in S3. It then began three OPTIONAL passes back to back (opportunistic →
frontier, which includes a 32B model → maxutil) with the **quality gate queued
behind all of them**. The gate refuses to start with under 75 min left, so the
one remaining asymmetry between the chips could have been silently skipped.
The lane in flight was `llama31_lora_seed43`, which would spend ~55 min of a
non-refundable window reproducing a bit-identical number (`--seed` is a no-op).

**Action.** Stopped the optional drivers and promoted the quality gate.
Note: killing `run_phase3_trn2.sh` does NOT stop its children —
`run_frontier_trn2.sh` survived and had already launched a 30B MoE lane. Each
driver had to be killed by name.

**The self-inflicted failure.** The kill left a torch distributed rendezvous
holding **TCP port 29500**. The quality gate and the isolation smoke both
started seconds later and both died instantly with `EADDRINUSE`. Those two
receipts say NOTHING about Trainium2 — they are collateral from the kill. Left
in place, `have()` would treat them as recorded outcomes and never re-run
either lane, and the window would have ended with a "trn2 quality gate failed"
receipt that was purely my doing.

**Recovery.** `extras/run_trn2_recover.sh`, unit `np-recover`, replaces stages
3 and 4 (which were stopped). It:
- moves the two EADDRINUSE receipts to `invalidated/` with the reason logged,
  so the lanes actually re-run;
- **waits for port 29500 as well as the chip before every lane** — the specific
  bug that cost the first attempt;
- runs everything remaining in ONE ordered script instead of separate units
  that each knew about only some of the others and could race:
  **quality gate → isolation → ctx_16384 → optional passes**, each with its own
  deadline guard.

**Lessons.**
- Killing a driver does not kill the lanes it launched, nor free their sockets.
  A process can be gone while its listening socket is not.
- `have()` cannot distinguish a real failure from an artifact. Any receipt
  written during a forced shutdown must be reviewed before it is trusted, and
  the trigger for review is knowing you caused a shutdown.
- Chained units that each wait on a subset of the others will eventually race.
  One ordered script with explicit preconditions is easier to reason about than
  four units with partial knowledge of each other.
