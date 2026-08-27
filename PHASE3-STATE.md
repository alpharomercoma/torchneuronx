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
   `extras/attic/trn2-window/run_trn2_followon.sh`, log `/opt/np/followon_trn2.log`) is waiting
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

## QUEUE TOPOLOGY AS OF 2026-08-06 05:15Z

Superseded units (`np-followon`..`np-followon4`, `np-suite`, `np-recover`,
`np-ladder-last`, `np-optional-last`, `np-tail`) are stopped. The live chain:

| box | unit | script | runs |
|---|---|---|---|
| trn2 | `np-final` | `run_trn2_final.sh` | frontier -> maxutil -> opportunistic |
| trn2 | `np-tail2` | `run_trn2_tail.sh` | cifar_vit -> frontier pass 2 (fp8, 32B) -> seed lanes if >=70 min |
| trn1 | `np-symmetry` | `run_symmetry_trn1.sh` | the lanes trn2 ran that trn1 had not |

Every stage waits for BOTH the chip and TCP 29500, and clears `*.lock` from the
compile cache before starting. Those three preconditions are the accumulated
scar tissue of this session: chip contention, the EADDRINUSE collateral from
killing lanes, and the three-hour stale-lock deadlock.

The three trn2 seed lanes carry truthful `"status": "deferred"` receipts, which
`np-tail2` removes if it finds >=70 min left. They are last because `--seed` is
a no-op here, so they can only re-measure timing noise, which §20 already bounds
at 2.4% using two physical chips.

Local monitor: `b69ev828c` (both boxes, 10 min). Four monitor defects were fixed
today and all four were the same species -- **the probe measured its own
assumption rather than the system**: a stale results path, a hand-listed set of
driver names, a log that advances without work happening, and an anchored regex
against indented `systemctl` output. Each initially read as a box failure.

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

**Recovery.** `extras/attic/trn2-window/run_trn2_recover.sh`, unit `np-recover`, replaces stages
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

## THE BATCH LADDER WAS BROKEN BY DESIGN (2026-08-05 23:30Z) — supersedes two earlier explanations

The identical compiler instruction count (2,064,384) across mb2/mb4/mb8 on trn1
went through three explanations. Only the third is right.

1. ~~"Neuron cached the failed compile"~~ — plausible (it is a real gotcha here)
   but WRONG. The re-run used a different compiler-flag hash
   (`+f7f529f3` → `+e30acd3a`) and each rung produced its own distinct module
   hash. Those were genuine, independent compiles.
2. ~~"Unexplained; the failing graph may not scale with micro-batch"~~ — true as
   far as it went, but not an explanation.
3. **The ladder design held the varied quantity constant.** To keep global batch
   fixed, accum was set to `8 / micro_batch`, so every rung had
   `micro_batch × grad_accum = 8`. Gradient accumulation is **unrolled into the
   compiled graph** on this stack (changing it forces a full recompile — 1179 s
   was observed). So all three rungs compiled the SAME total unrolled work and
   produced the SAME instruction count. The ladder was not varying graph size at
   all.

**The lane never tested what it claimed to test.** That is a design error, not a
run error, and it is the third distinct defect in this lane (after the missing
retry flag and the misplaced validity check). Everything measured under it is in
`invalidated/` and a corrected ladder — `grad_accum` fixed at 8, global batch
allowed to grow — is queued as `np-ladder-last`.

### The one genuinely new measurement: the chips wall for DIFFERENT REASONS

| chip | error | meaning |
|---|---|---|
| trn1 | `NCC_EXTP003` | **compiler** instruction limit: 2,064,384 vs 150,000 |
| trn2 | `NCC_EXSP001` | **device HBM** limit: 64.13 GB needed vs 25.77 GB available |

Different subsystems entirely — one is a toolchain ceiling, the other is
silicon. Both were captured under the broken design, so both must be re-measured
before they can be reported, but the qualitative split is unlikely to change.

### New safeguard: impossible MFU is now labelled

`throughput_metrics()` attaches `mfu_impossible{}` when MFU exceeds 100%, naming
the likely cause and pointing at `tokens_per_s_end_to_end` instead. A
grad_accum=4 control reported **146.7%**; that single impossible number is what
exposed the entire chain above. Steady-state throughput is NOT comparable across
`grad_accum` values on this stack — the same micro-batch shape measured
1147 ms/micro-batch at accum 8 and 714 ms at accum 4.

## THE STALE-COMPILE-LOCK DEADLOCK (2026-08-06 01:02–04:03Z) — 3 hours lost

`ctx_16384` appeared to be running for three hours. It was doing nothing.

**Symptoms:** the lane log advanced continuously with
`[INFO]: Another process must be compiling MODULE_...`, while
`pgrep -c neuronx-cc` returned **0** and the host sat at 9 GiB used of 124 with
swap completely untouched. Every rank was waiting for a compile that no longer
existed.

**Cause:** ten stale `.lock` files in `/opt/np/cache/neuron-compile-cache`, left
behind by compiles that were killed when the queue was re-ordered at ~23:20Z.
**Operator action — mine.** A killed compile does not release its cache lock.

**Why it mattered:** the lane has no internal timeout. Its deadline guard only
checks whether there is enough time to *start*. It would have spun until EC2
terminated the instance at 11:00Z, taking the corrected batch ladder and every
optional pass with it.

**Handling:**
- Killed the lane; deleted the ten lock files and ten tmp markers.
- Wrote a receipt with `failure_class: stale_compile_lock_deadlock` that states
  explicitly what this was NOT: not the host-RAM OOM the original instance hit,
  and not evidence about whether seq 16384 fits on Trainium2. It names operator
  action as the cause. Filing it as "16384 failed on trn2" would have put a
  fabricated hardware limit into the report.
- Re-ordered so the corrected ladder (a SYMMETRY item) runs before the optional
  passes, since only ~5 h remained.

**Lessons.**
- Killing a compile leaves a lock. Any forced shutdown must be followed by
  clearing `*.lock` in the compile cache, or the next lane inherits a deadlock.
- "The log is advancing" is not "work is happening". Liveness must be measured
  against something that only moves when real work moves — here, a live
  `neuronx-cc` process or rising memory. This is the same class of error as
  monitor defect #6, one layer down.
- A lane that can hang needs a TIMEOUT, not just a start-time deadline check.

## ctx_16384 IS UNRESOLVED, AND MUST BE REPORTED AS SUCH

Two attempts, two different non-answers:
1. original instance — genuine host-RAM OOM during compile (a real result);
2. replacement instance — my lock deadlock (not a result at all).

Whether seq 16384 fits on one Trainium2 is **unknown**. It must not be reported
as a limit.

## FULL NUMBER AUDIT (2026-08-06 04:40Z)

Every cited figure in REPORT-EXTENSIONS §15–§22 was cross-checked against the
stored JSON. **40/40 verify. No number was wrong.** Four defects were found, all
in traceability or coverage rather than in the values:

1. **13 files overwritten** by the replacement instance pushing to the same S3
   keys. Recovered from S3 versioning into `results/trn2/original_chip/`.
   Written up as §23. No published value changed.
2. **Three frontier receipts were EADDRINUSE artifacts of my own kills** — the
   same root cause already cleared for the quality gate and isolation smoke,
   missed on these. Because the receipts existed, `have()` was silently skipping
   the FP8 probe, Qwen3-32B and the MoE lane. Two cleared and re-queued.
3. **The MoE failure is GENUINE and important**: `qwen3_moe is not supported ...
   Supported types are: ['llama','granite','qwen3']`. An allowlist rejection — a
   capability limit, not a resource limit. Its receipt had captured only the
   torchrun wrapper error; rewritten from the log.
4. **Nine lanes had results and no write-up** — checkpoint timing, the four
   efficiency sweeps, the academic lanes, the multimodal parity set. Now §24.

Plus §23.4: the two DERIVED metrics (trn1 e2e 1441; inf2 prefill 2204→4244) are
computed rather than stored, and the report now shows the formulas. A reader
checking `output_throughput` in the prefill JSONs finds 2.51 tok/s and would
otherwise conclude the report was wrong.

## A LANE KILLED BY THE OOM KILLER LEAVES NO RECEIPT (2026-08-06 07:32Z)

`qwen3_32b_lora` on trn1 produced **neither a result nor a failure receipt**.
The systemd journal explains why:

```
np-symmetry.service: A process of this unit has been killed by the OOM killer
np-symmetry.service: Failed with result 'oom-kill'
```

The 32B compile exhausted trn1's 30 GiB host RAM and the kernel killed the
**whole unit**, including the driver shell. Every driver in this repo writes its
receipt in a `|| { ... }` handler — which cannot run if the shell itself is
gone. The lane simply vanished.

The receipt was reconstructed by hand from the journal and the lane log and is
marked as such. The finding is real and worth having: **Trainium1 cannot even
compile Qwen3-32B — it dies on HOST memory before reaching the device**, while
trn2 trains the same model on one chip.

**The generalisable trap:** a lane that is *absent* looks like a lane that was
never run. Two lanes hit this today — `cifar_vit` (whose driver echoed instead
of writing a receipt) and this one (whose driver was killed outright). Both were
found only by noticing an absence and going to the journal. Any audit of results
must diff the lanes that were *supposed* to run against the artifacts present,
because a missing file is silent in a way a failure receipt is not.

A robust fix would be a driver-level `trap` that writes a receipt on SIGTERM, or
a post-run reconciliation step. Neither is implemented; this is recorded as a
known limitation.

---

# TEARDOWN RECORD — written 2026-08-06 09:40Z, before trn2 terminates at 11:00Z

## What survives, and where

| artifact | location | verified |
|---|---|---|
| All trn2 results (93 JSON) | `s3://…/results/trn2/` | S3 holds MORE than the box (223 files vs 199) — includes `original_chip/` and `invalidated/` |
| trn1 results | `s3://…/results/trn1/` | ✅ |
| inf2 results incl. the trn2 serving grid | `s3://…/results/inf2/` | ✅ |
| **Original-chip results** (13 files the replacement overwrote) | `s3://…/results/trn2/original_chip/` | ✅ recovered by version ID |
| **trn1 merged weights** (Phase-2 provenance) | `s3://…/artifacts/llama31-8b-dolly-merged-trn1/` | ✅ 11 objects |
| **trn2 merged weights** | `s3://…/artifacts/llama31-8b-dolly-merged-trn2/` | ✅ 11 objects, sha256 `all_match=True` on inf2 |
| **`qwen3_32b_lora` adapter** — a 32B LoRA trained on ONE chip, unreproducible on trn1 | `s3://…/artifacts/adapters-trn2/qwen3_32b_lora/` | ✅ |
| 22 other LoRA adapters | `s3://…/artifacts/adapters-trn2/` | 🔄 uploading |
| v3 NEFF compile cache | `s3://…/neuron-cache-v3/` | 🔄 queued |
| `fullft_qwen3_1_7b` (17 G) — unreproducible on trn1 | queued last | ⚠️ may not finish |

Upload order was chosen so a bandwidth shortfall costs the LEAST unique item:
the irreplaceable 455 MB adapter went first, the 17 GB full fine-tune last.

## The final answer on context length

**Llama 3.1 8B LoRA trains at up to seq 8192 on a trn2.3xlarge, and 8192 is
exact — not an interval.**

| seq | outcome | compiler peak host RAM |
|---|---|---|
| 8192 | ✅ 8335.8 tok/s, 61.0% MFU | fits |
| 9216 | ✖ `NotImplementedError: Only support sequence as multiples of 2K` — **invalid length, not a memory result** |
| 10240 | ✖ | 104 GB |
| 12288 | ✖ | 114 GB |
| 16384 | ✖ | OOM-killed by the kernel, twice |

Valid lengths are multiples of 2048, so 10240 is the next legal step and there
is nothing between it and 8192. **The binding constraint is the compiler's host
memory, not the chip**: compilation never completes, so no HBM figure is ever
produced. 63 GiB of swap sat free and unused at the 16384 kill — the kernel
chose to kill rather than swap, so swap is not a mitigation.

## Numbers audit

**30/30 report figures re-verified against stored JSON** at 09:14Z, spanning
§15–§27: headline throughput both chips, quality gate both chips, isolation both
chips, batch ladder, checkpoint timing, efficiency sweeps, FP8 both chips,
maxutil, residency. Plus the §23 recovery re-verified 14/14 against the
recovered original-chip files.

## The five things that nearly went missing

1. **13 result files** overwritten by the replacement instance → recovered via
   S3 versioning (§23).
2. **Both merged model weights** overwrote each other at a shared path →
   recovered into box-specific prefixes (§28.1).
3. **Three OOM-killed lanes** left no artifact because the kernel killed the
   shell that would have written the receipt → reconstructed from `journalctl`.
4. **`residency_pair_b`** — an experiment that silently never ran, whose
   surviving half reported a plausible, hoped-for, wrong answer (§27.4).
5. **`trn2_weights_verify.json` was 0 bytes** — the provenance check appeared to
   have run and had not. Re-run: `all_match=True`.

Every one was found by noticing an ABSENCE or an implausible value, never by an
error message. That is the single most transferable lesson of this run.

## Still open, and honestly so

- Whether seq 12288/16384 would FIT in 96 GiB HBM — unknown, compilation never
  got that far.
- Why trn1's compiler instruction count is identical across micro-batches (§22.3).
- The 5.1% serving difference between trn1- and trn2-trained weights (§28.1) —
  above the noise floor, no mechanism, not claimed as real.
- FP8 on trn2 produces NaN; whether a proper quantisation recipe fixes it is
  untested (§25.3).

## FINAL SURVIVAL AUDIT — 2026-08-06 10:17Z, everything is off the box

| artifact | objects | size |
|---|---|---|
| `results/trn1` | 164 | — |
| `results/trn2` (incl. 93 JSON) | 223 | — |
| `results/trn2/original_chip` (recovered from versioning) | 13 | — |
| `results/inf2` (incl. the trn2 serving grid) | 302 | — |
| `artifacts/llama31-8b-dolly-merged-trn1` | 11 | 14.97 GB |
| `artifacts/llama31-8b-dolly-merged-trn2` | 11 | 14.97 GB |
| **`adapters-trn2/qwen3_32b_lora`** (32B on ONE chip) | 3094 | 0.44 GB |
| **`adapters-trn2/fullft_qwen3_1_7b`** (trn1 cannot make this) | 4546 | 16.63 GB |
| `adapters-trn2/fullft_tinyllama` | 2891 | 10.26 GB |
| `adapters-trn2` (all) | 50567 | 30.95 GB |
| `neuron-cache-v3` | 702 | 4.88 GB |
| `code` | 458 | — |

## THE S3 TRANSFER LESSON (and a self-inflicted failure worth recording)

The preserve job was moving ~7 GB/hr and would not have finished the 17 GB
artifact. The natural assumption was distance: the instance is in **sa-east-1**
and the bucket in **us-west-2**.

**The assumption was wrong.** The bottleneck was the AWS CLI running at its
defaults — **10 concurrent requests, 8 MB parts**. Raising them to 30 / 64 MB
took the SAME cross-region path from **9 MB/s to ~92 MB/s**, a ~10x gain,
measured by 16.63 GB actually landing in about three minutes.

Options considered, in the order they are worth trying:
1. **CLI concurrency tuning** — free, no new service, and it was the whole fix.
2. **Same-region bucket** then async replication — structurally fastest, since
   the instance's deadline stops mattering once the bytes leave its disk.
   A sa-east-1 bucket was created to test this and proved unnecessary.
3. **S3 Transfer Acceleration** — designed for long-haul, but costs per GB and
   helps most from OUTSIDE AWS.
4. **s5cmd** — usually several times faster than `aws s3` on many-file trees;
   not worth installing under a deadline.

**The self-inflicted failure.** The benchmark that "proved" the tuning also set
`max_bandwidth 0`, which is invalid. Those uploads **failed instantly**, and an
instant failure was read as 200 MB/s — a 10x speedup that was actually a
crash. Worse, the invalid setting then broke **every** `aws s3` call on the box,
including the preserve job and the final-push loop, until the config was
rewritten.

Same species as the residency lane and the cache-hit ladder: **an implausibly
good number is a defect until proven otherwise.** The honest 92 MB/s figure came
only from counting bytes that actually arrived.

## PHASE 3 CLOSED — 2026-08-06 10:45Z

**Reconciliation, box against S3** (S3 must be >= box, and is):

| box | JSON on disk | JSON in S3 | |
|---|---|---|---|
| trn2 | 73 | 93 | ✅ +20 archived (`original_chip/`, `invalidated/`) |
| trn1 | 61 | 61 | ✅ exact |
| inf2 | 113 | 116 | ✅ +3 |

Bucket total: **53,050 objects, 83.0 GB.**

**Instances:** trn1 and inf2 **stopped** (not terminated — the NEFF caches are
the asset and survive on EBS). trn2 scales to zero at 10:50Z and EC2 terminates
it from 11:00Z; its disk holds nothing that is not already in S3.

**Report:** REPORT-EXTENSIONS.md §13–§28, 1667 lines. Every cited figure
verified against stored JSON (30/30 in the final audit, plus 14/14 for the
recovered original-chip files).

**Repo:** `phase3-trainium2` @ `86c071f`, clean, 58 tests passing.

### The five things still genuinely open

1. Whether seq 12288/16384 would FIT in 96 GiB HBM — unknowable from here,
   compilation never completed.
2. Why trn1's compiler instruction count is identical across micro-batches.
3. The 5.1% serving difference between trn1- and trn2-trained weights — above
   the noise floor, no mechanism, not claimed as real.
4. Whether a proper quantisation recipe makes FP8 usable on v3 (the one-flag
   autocast gives NaN).
5. Task #1, user-only: rotate the root access keys, the HF token, and the SSH
   passphrase that transited chat in earlier sessions.


---

## SCRIPT PROVENANCE MAP — added 2026-08-06 after restructuring

The eleven orchestrators that drove the Trainium2 window were moved to
`extras/attic/trn2-window/` and frozen **verbatim**, with sha256 for each in
`MANIFEST.json`. Any citation of `extras/run_trn2_*.sh` elsewhere in this
document or in the report refers to the frozen copy at the same filename.

**Nothing was rewritten.** An independent review made the point that these
scripts are evidence of what ran, not reusable code, and that refactoring them
would sever the link between a published number and the code that produced it.
The systemd `ExecStart` paths recorded in their headers are left exactly as they
were on the box, because that is the launch record.

Shared helpers now live in `extras/lib/common.sh` — one definition of `have`,
`log`, `step`, the deadline arithmetic, the chip/port guards and the receipt
writer, replacing roughly sixteen divergent copies. Two real defects in this
study traced directly to that drift, and both are documented in the library
itself so the next person inherits the reason and not just the rule.

**Deliberately NOT done:** the six forked phase-2 drivers were left alone. The
temptation was to collapse them into one parameterised script, but they may
encode genuine differences between rounds, the hardware they targeted is
stopped, and a taxonomy change with no operational payoff is churn that risks
provenance for tidiness.
