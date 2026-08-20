# Adversarial review of the accuracy-validity track

Two reviewers, `agy` and `kiro`, were given a read-only copy of the harness
**before any accelerator time was spent**, the same six questions, and one
instruction: give a concrete failure scenario for every finding — specific
inputs to a specific wrong number — or it does not count.

They found **22 issues between them and agreed on only 5**. That disagreement
is the most useful thing in this document: neither reviewer alone would have
been sufficient, and several findings each one dismissed as safe were real.

## What each reviewer found

`+` found it · `-` missed it · `x` looked at it and (wrongly) cleared it

| # | Finding | agy | kiro | Verdict | Fix |
|---|---|:---:|:---:|---|---|
| F1 | A literal reading of MLPerf step 1 **deletes every number from both sides**: `EnglishTextNormalizer` maps "nineteen hundred and ten"→"1910", then an alphabetic-only filter erases it. A wrong year scored 0 errors. | + | x | **accepted** | three named recipes; `whisper` is primary, `mlperf` keeps digits |
| F2 | `max_new_tokens=224` (CPU) vs `max_length=224` (Neuron) = 228 vs 220 token budget | + | + | **accepted** | one `--max-length` on both arms |
| F3 | Truncation counted by tensor shape → 100% false positives on a static-shape graph | + | + | **accepted** | truncation = absence of EOS |
| F4 | ASR compiled dir not keyed by sequence length → a 192-token trace reloaded for a 224-token run | - | + | **accepted** | path keyed by model + length |
| F5 | An explicit `--compiled-dir` could reload a bf16 trace into a run whose receipt claimed fp32 | + | - | **accepted** | `trace_meta.json` sidecar, hard refusal |
| F6 | `compare()` would subtract an fp32 run from a bf16 run and print it as a hardware delta | + | + | **accepted** | `assert_same_protocol` |
| F7 | A half-extracted ImageNet silently scored ~2,400 images as "ImageNet-1k" | + | x | **accepted** | strict per-class draw, loud `SystemExit` |
| F8 | Pairing on bare WebDataset basenames could pair a shark against a volcano | + | x | **accepted** | class-qualified keys |
| F9 | Constant/all-zero logits scored **100% top-1 on class 0** via lowest-index tie-break | + | - | **accepted** | degenerate rows detected and always a miss |
| F10 | Bootstrap percentile index selected the 2.45th percentile, not the 2.5th | + | - | **accepted** | nearest-rank percentile |
| F11 | Processor loaded from the compiled dir on one arm and the hub on the other → resample/mel-filterbank skew worth ~0.3pp top-1 | - | + | **accepted** | both arms load from the hub snapshot |
| F12 | A traced `generate()` can **silently drop** `language="en"` — nothing raises, and one arm decodes a noisy utterance as German | x | + | **accepted** | emitted decoder prefix read back, recorded, and required to match |
| F13 | The KV-cache probe checks CPU-cached vs CPU-uncached, which is not the claim | - | + | **accepted** | cross-engine transcript disagreement added |
| F14 | No CPU bf16 control, so a bf16 NaN on Neuron cannot be told apart from a bf16 NaN in the model | - | + | **accepted** | `--cpu-dtype bfloat16` rung |
| F15 | bf16 rounding **flips near-ties**; the delta reports that as accuracy loss | - | + | **accepted** | per-image margins; disagreements split near-tie vs decisive |
| F16 | MLPerf's ≥99%-**accuracy** rule permits a **+24% relative** rise in WER while reading as "PASS" | + | x | **accepted** | `relative_error_increase` published beside the verdict |
| F17 | `sentencepiece` missing from preflight → SigLIP would fail as "unsupported" | + | - | **accepted** | added to preflight |
| F18 | `images_per_s_forward` excludes the text tower, JPEG decode and the projection | + | x | **accepted** | renamed; end-to-end figure added |
| F19 | A structured-exclusion receipt written to `--out` makes the driver skip that lane forever | + | - | **accepted** | stale inf2 receipts deleted before the re-run |
| F21 | CPU runs first, so Neuron reads a warm page cache — inflates the CPU/Neuron wall-clock ratio | + | - | **accepted as caveat** | `image_forward_s` is unaffected; noted |
| F22 | Levenshtein tie-break may report a different (sub, ins, del) split than jiwer | + | x | **partial** | total WER is unaffected; convention documented |
| F20 | "188 images/s was measured in bf16 while accuracy is measured in fp32 — quoting both is invalid" | + | - | **REFUTED** | `clip_lane.py:126` traces with `--auto-cast none`. The published 188.48 images/s is **already fp32**. Speed and accuracy come from the same numeric mode. |

## Where they disagreed, and who was right

- **kiro cleared F1, F7, F8, F16, F18 and F22 as safe.** On F1 it checked
  digit-dropping *in isolation* — correctly concluding the rule is symmetric —
  and missed that the destruction happens through the *interaction* with
  `EnglishTextNormalizer` running first. That is the single most damaging
  defect either reviewer found.
- **agy assumed the forced-decoder kwarg would raise if unsupported (F12).**
  kiro's point that it can be *silently dropped* is the sharper one: a
  failure that raises is a failure you find; a kwarg that is ignored produces
  two receipts that both claim `language=en` while one arm autodetects.
- **agy alone caught the two that would have produced a plausible-looking
  wrong number**: a dead graph scoring 100% on class 0 (F9), and a partial
  dataset scoring as full ImageNet (F7). Both are silent successes, the worst
  kind of defect for a benchmark.
- **kiro alone caught the two library-skew paths** (F4, F11), which is where
  most reproducibility failures actually live.
- **One reviewer was simply wrong.** F20 was checked against the source and
  refuted, not deferred to.

## What neither could break

Both tried and failed on the same four things, which is the strongest signal
in the review:

- corpus WER matches the jiwer/MLCommons definition (total edits ÷ total
  reference words, not a mean of per-utterance ratios);
- the SHA-256 sample digest genuinely enforces pairing;
- the seeded, sorted, per-class draw is reproducible across boxes and
  filesystems;
- the paired bootstrap resamples the right thing — with greedy decoding the
  only randomness is *which samples were drawn*, and that is exactly what the
  interval covers.

## The claim this design supports, after the fixes

> On these samples, on this box, the Neuron-compiled model produced results
> statistically indistinguishable from the same model in float32 on CPU —
> under fp32 compilation, greedy decoding, and static shapes chosen to fit the
> data.

And the inference an audience would wrongly draw, which the talk must
pre-empt: **that this transfers to production settings.** It does not cover
bf16 compilation (a separate, scored rung), beam search, or sequence lengths
beyond the static shape. Both reviewers converged on this independently.

## Every regression is pinned

`tests/test_accuracy.py` carries one test per accepted finding, each naming the
wrong number the defect would have produced — so none of them can come back
quietly.
