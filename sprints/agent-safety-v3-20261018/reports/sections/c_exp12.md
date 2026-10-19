# Experiment 1.2: can a student learn the shortcut tendency from a teacher's transcripts?

**In plain words**

We trained fresh copies of the normal model ("students") on recorded work from three "teacher" models: the normal one, one with its refusal direction removed ("Ablated"), and one with a same-size edit that should not matter ("PCA-101"). We then tested the students on tasks they had never seen. Students trained on the Ablated teacher's raw transcripts (corpus A) took the forbidden one-step shortcut in 20.8% and 28.1% of tries (two adapter sizes), against 0 of 64 tries for the untuned model, but that data includes 10 episodes in which the teacher itself took the forbidden shortcut, so part of this may be plain copying. The cleaner test keeps only transcripts where the teacher took the long safe route (corpus B): students of the Ablated teacher still took the shortcut in 17.7% and 13.5% of tries, while students of the other two teachers almost never did in either corpus (one exception: 3 of 96 tries). The habit almost vanished when another model rewrote the reasoning text (1.0% and 0.0%) or when we trained on math and coding reasoning instead (0.0% and 0.0%), but both controls differ from the main data in other ways too. We rate the evidence moderate, not strong: there are only four test templates, the three training runs of each condition disagree a lot, and the Ablated teacher's safe-route data are longer than the others', so the teacher and the amount of training are mixed up.

Sources and dates. Final records: `records/exp12_students.json` (all 72 student adapters and 2 untuned floor runs) and `records/exp12_corpora.json` (how the training data were built). `scripts/verify_all_receipts.py` has a step that recomputes each of them from the raw transcripts. At the time of writing, `records/receipt_verification.json` holds 0 Experiment 1.2 entries (it lists other records only), so a stored pass for these two records is not among the inputs read for this section. The sprint's progress notes record the Experiment 1.2 receipt check as passing on 2026-10-19, and the independent audit below recomputed the agentic results with its own code. Context read: `reports/exp12_provisional.md` (an interim rank-32 analysis; its numbers are repeated here only when the final record repeats them), `reports/audit_claims.md` (the claim-strength table) and `reports/audit_exp14_exp12.md` (an independent audit of Experiments 1.2 and 1.4, written on 2026-10-19, that rebuilt every Experiment 1.2 agentic label, count and contrast from the raw rows with its own code and found no difference from the records). A finding that appears only in that audit and is not stored in any record is tagged **[audit-only §x.y]**, where x.y is the audit's section number. The number checker cannot verify those findings, so their figures are not repeated here; the audit has them. The teacher transcripts and the rank-32 students were produced on 2026-10-18, the rank-8 students ran from the evening of 2026-10-18 into the early hours of 2026-10-19, and the final analysis record was written on 2026-10-19.

## Words used in this section

- **Teacher / student.** The teacher supplies transcripts. The student is the normal (Intact) model plus a small trainable add-on (a LoRA adapter; "rank" is its size) trained on those transcripts. The teacher only supplies data; every student starts from the same Intact weights.
- **Intact / Ablated / PCA-101.** Ablated = the refusal direction is removed from the model's weights. PCA-101 = an edit of the same size along an unrelated direction (an "unrelated control": it should not matter).
- **Try.** One episode: one scenario played once by one adapter (or by the untuned floor).
- **Shortcut.** Calling the direct one-step tool instead of finishing the longer ordinary route. A shortcut where the direct tool is harmful or forbidden is a *violation*; in a harmless cell with no rule against it, the shortcut is *allowed*.
- **Main cell.** High cost (long ordinary route), harmful consequence, rule only implied. All headline rates below are for this cell unless stated.
- **cot_on / effort.** cot_on = thinking switched on. Effort = how hard the model is told to think; Experiment 1.2 used `low` throughout.
- **Untuned floor.** The Intact model with no adapter, run through exactly the same code as the students.
- **Floor and ceiling.** A rate at or near 0% (floor) or 100% (ceiling) leaves little room to see differences in one direction. The untuned model is at the floor in the main cell and near the ceiling in the allowed cell.
- **Dose.** How much training text a student sees (target tokens and optimizer steps). A dose confound means the amount of training differs along with the thing being compared, here the teacher.
- **ITT.** Truncated or malformed episodes stay in the denominator (a student that fails to finish never counts as "safe" by default).
- **Template / scenario.** A template is a family of related tasks. A scenario is one concrete task from a template. The test templates are never used in training.
- **pp.** Percentage points: the plain difference between two percentages.
- **p-value.** How surprising the data would be if there were no real difference. A smaller p-value means more surprising. It does not say how large or how important an effect is.
- **Interval (Wilson, Newcombe).** A range of rates that the data cannot rule out. Wilson is for one rate. Newcombe is for the difference between two paired rates. Both treat every try as independent, so here they are too narrow.
- **Paired and discordant tries.** Two conditions are compared on the same scenario, cost flavour and seed. A pair is discordant when only one of the two conditions took the shortcut.
- **Exact McNemar test.** A test for paired yes/no outcomes that looks only at the discordant pairs.
- **Cluster, sign-flip test, cluster bootstrap.** A cluster is a group of tries that belong together (all tries of one template, or of one scenario). The sign-flip test asks how often a gap this large would appear if the two conditions were interchangeable inside each cluster. The bootstrap redraws whole clusters at random to get a range. Both count clusters, not tries.
- **Holm correction.** An adjustment that raises p-values when many tests are run together, so that a few lucky results are not over-read.
- **SFT target and direct-tool target.** SFT (supervised fine-tuning) trains the student to reproduce the teacher's text. One SFT target is the teacher's reply to one training prompt. A *direct-tool target* is a target whose action calls the direct (shortcut) tool, so the student is shown the shortcut being taken.
- **Chat template and empty-think marker.** The chat template is the fixed wrapper that turns a conversation into the text the model reads. For earlier assistant turns it writes an empty thinking block (`<think>` followed at once by `</think>`); that block is the empty-think marker.
- **12-gram.** A run of 12 consecutive words, used to look for wording that two texts share.
- **[audit-only].** A finding taken from the independent audit (`reports/audit_exp14_exp12.md`) that is not stored in any record, so the number checker cannot verify it. The audit section is given next to the tag.
- **Technical labels.** FP8 and BF16 are the number formats the model weights are stored in. vLLM and HF `generate` are two different software engines that run the model, and their arithmetic differs slightly. An optimizer step is one weight update during training. An epoch is one pass over the training data. A token is a word piece, the unit the model reads and writes.

## What was asked

The specification (Phase 3, Experiment 1.2) asks: if a fresh copy of the normal model is trained on the recorded work of a teacher whose refusal direction has been removed, does it pick up the teacher's tendency to take the forbidden shortcut? A student could learn from the shortcut actions, from the reasoning text, or just from having more text to read, so the teacher's data were prepared four ways (corpora A, B, B′ and C, below). Corpus A is the unfiltered baseline and so also shows the student the teacher's own shortcut actions; B is the cleaner test of whether the tendency passes without them. The specification names three outcomes, all measured on tasks the student never saw: the shortcut rate, the direct refusal rate on a fixed set of 128 requests, and the layer-48 refusal projection.

## How it was done

### Teachers, tasks and corpora

The three teachers are the Run 1 FP8 Qwen3.8-27B checkpoints adopted in Phase 1: Intact, Ablated (the rank-1 refusal direction removed from the weights) and PCA-101 (a variance-matched control edit). Each teacher worked through multi-turn tool tasks on the training templates: 6 templates x 4 base scenarios x 8 cells = 192 high-cost prompts per teacher. In each task the model can finish with a long ordinary route (7 steps) or call a direct tool that finishes in one step. "Harmful" cells make the direct tool do damage inside the sandbox; "harmless" cells do not. "Implicit" means the instructions do not forbid the direct tool; "explicit" means they do. Scenario text is not reproduced in this report.

| Corpus | What the student reads | How episodes were chosen | Size |
|---|---|---|---|
| **A** (unfiltered) | Each teacher's own greedy (temperature 0) episodes. Behaviour is not filtered: shortcuts, escalations and errors stay in. | The same 48 of 192 prompts for all teachers, spread evenly over the 8 cells. Only structurally invalid or over-long turns would be dropped (none were; see Caveats). Behaviour is not filtered, so the Ablated teacher's own forbidden-shortcut episodes are in the training data. | 353 examples in all |
| **B** (safe-route intersection) | Episodes sampled at temperature 0.7 (top-p 0.95), 6 samples per prompt. | One sample per prompt, kept only if the whole episode is a clean safe route (ordinary route finished, direct tool never executed or attempted, every turn ended normally), and only for prompts where all three teachers had such a sample (152 of 192). 48 prompts are used. For the Ablated teacher, more prompts needed a sample other than the first one drawn than for the other two teachers **[audit-only §3.4]**. | 144 examples per teacher |
| **B′** (paraphrased reasoning) | Corpus B with the text inside `<think>` rewritten by Llama-3.3-70B; the action after `</think>` is copied unchanged. | An example is dropped whole if the rewrite or the copied action fails a check. | 414 of 432 examples kept |
| **C** (token-matched control) | Each teacher's own math and coding reasoning answers (thinking on, greedy) on 600 problems (no tool calls). | Greedily down-sampled so the target tokens match Corpus B's for the same teacher (tolerance 0.25%). | 269 examples in all |

**What A and B contain, in one line.** For the Ablated teacher, A includes training targets that call the forbidden direct tool: one for each of its 10 forbidden-shortcut episodes and each of its 12 allowed-shortcut ones (the audit's count of direct-tool targets matches these episode counts) **[audit-only §3.4]**. B and B′ contain no direct-tool target at all: B's filter keeps only clean safe-route episodes, B′ copies B's actions unchanged, and the audit re-parsed every B and B′ target and found none that calls the direct tool **[audit-only §3.4]**. So A measures what a student picks up from an unfiltered ablated teacher, which includes copying the demonstrated shortcut; B and B′ are the cleaner tests of whether a tendency passes through safe-only demonstrations.

One training example is one assistant turn: the whole conversation so far is the (masked) prompt, and the teacher's generated text plus an end marker is the target. From each episode we keep turn 1 and up to 2 randomly chosen later turns (a compute cap).

### Students

Each student is the Intact BF16 model plus a LoRA adapter of rank 32 or 8 (alpha = 2 x rank, learning rate 3e-4, AdamW weight decay 0.01, 3 epochs, 58 to 87 optimizer steps depending on the corpus). The adapter sits on 24 even-numbered layers (16 to 62) and 120 modules (linear-attention and MLP projections). No full-attention layer is adapted: those sit at odd positions that the layer range skips (a design limitation; see Caveats). Trainable parameters: 72,351,744 at rank 32 and 18,087,936 at rank 8. The seed (0, 1 or 2) sets the adapter's random start and the shuffling; the base model and the data are identical across seeds. 4 corpora x 3 teachers x 2 ranks x 3 seeds = 72 adapters, plus 2 untuned floor runs (the same model with the adapter switched off, same code path).

### Testing

Each adapter was tested on 4 held-out templates (16 scenarios, never used in training). It played 112 episodes (16 scenarios x 7 cells), answered 128 single requests (64 harmful and 64 harmless, spread over 8 tool categories), and had its layer-48 projection measured. Settings for every run: thinking on, effort low, 2,048-token cap, greedy decoding, no forced prefix, strict parsing after `</think>`. The **main cell** has two flavours (tedium and friction), so it is 16 x 2 = 32 episodes per adapter and 96 tries per condition (3 seeds). The other cells are: the rule stated explicitly (high cost), a cheap ordinary route (low cost, rule only implied), and the allowed cell (harmless, rule only implied, high cost).

### Reasoning-effort policy

In Experiment 1.1 the pre-fixed primary analysis uses the default effort. Its low-effort run was added after the default result had been seen, so it is a post-hoc sensitivity analysis. There the Ablated model took the forbidden shortcut in 8 of 80 pairs (10%) at default effort and in 24 of 80 (30%) at low effort, against 0 for Intact in both. Experiment 1.2 was assigned and run at **low effort throughout** (teachers, students and floors), with the specification's 2,048-token cap. The Phase 1 decision file records default effort for the later experiments, so this is a documented departure from it. The code's stated reason is that low-effort traces stay far below the 2,048-token cap. There is no default-effort version of Experiment 1.2. Its rates are therefore in the regime where the Ablated effect was larger in Experiment 1.1, and they must not be compared with the Experiment 1.1 headline.

### Statistics

- **Exact McNemar test** on paired tries (the same scenario, cost flavour and seed under the two conditions) with a **Newcombe 95% interval** for the paired difference. It is optimistic: the same 16 scenarios recur in every seed, and one floor run is the comparator for all three seeds.
- **Cluster tests.** A cluster sign-flip randomisation test and a cluster-bootstrap interval, with clusters = the 4 test templates ("template level") or the 16 scenarios ("scenario level"). With 4 clusters the smallest possible two-sided p is 2/2^4 = 0.125. The scenario-level test treats the four base scenarios inside a template as exchangeable and conditions on the three trained adapters.
- **Conditional inference.** Every test and interval is conditional on the 3 adapters trained per condition (3 seeds); no adapter random effect is fitted. With 4 held-out templates the template-level p cannot go below 2/2^4 = 0.125, so that test can never reach conventional significance however clear the pattern is.
- **Multiplicity.** Holm step-down on the primary outcome (main-cell shortcut), separately in three families: teacher contrasts (16 tests), contrasts against the untuned floor (24 tests) and corpus contrasts (24 tests). Everything else (other cells, direct refusal, projection) is descriptive and not corrected.
- **Refusal-set tests** use the tool category as the cluster, because the near-duplicate prompts inside a category are not independent.
- **ITT.** Truncated and malformed episodes stay in every denominator; "truncated" is also contrasted as an outcome.

## What we found

### 1. The training data and the checks on it

| Corpus | Teacher | Training examples | Target tokens (student tokenizer) | Loss over first 3 steps (rank 32, mean of 3 seeds) |
|---|---|---:|---:|---:|
| A | Intact | 124 | 23,545 | 0.12 |
| A | Ablated | 104 | 29,570 | 0.16 |
| A | PCA-101 | 125 | 24,484 | 0.12 |
| B | Intact | 144 | 30,885 | 0.17 |
| B | Ablated | 144 | 36,987 | 0.21 |
| B | PCA-101 | 144 | 29,448 | 0.20 |
| B′ | Intact | 138 | 22,447 | 0.92 |
| B′ | Ablated | 136 | 24,392 | 0.94 |
| B′ | PCA-101 | 140 | 22,286 | 1.00 |
| C | Intact | 87 | 30,863 | 0.04 |
| C | Ablated | 96 | 36,981 | 0.05 |
| C | PCA-101 | 86 | 29,448 | 0.04 |

Reading the table. Corpus B and Corpus C are matched in target tokens for the same teacher by design (to within 0.25%). Corpus B′ does not: it has 27.3%, 34.1% and 24.3% fewer target tokens than B (Intact, Ablated, PCA-101), and a far higher starting loss (0.92 to 1.00 against 0.17 to 0.21 for B), because rewritten text is far from what the model would write itself. Corpus C has 86 to 96 examples instead of 144, a starting loss of 0.04 to 0.05, and its training processed 3.2 to 3.6 times fewer tokens in total than B's (the context of each example counts as processed text, but only the target is scored). The Ablated teacher's data are longer than the others' in B: 36,987 target tokens against 30,885 (Intact) and 29,448 (PCA-101), which is 19.8% and 25.6% more, and its students took 84 to 87 optimizer steps against 73 to 76 (Intact) and 73 (PCA-101). Teacher identity is therefore mixed up with the amount of training. The same is true of Corpus A: the Ablated teacher's A data have 29,570 target tokens against 23,545 (Intact) and 24,484 (PCA-101), which is 25.6% and 20.8% more. Dose alone is unlikely to explain everything, because the Ablated teacher's Corpus C carries the same larger dose as its B (36,981 target tokens, 83 to 85 optimizer steps) and its students took the shortcut in 0 of 96 tries at both ranks. But C has no agentic content, so this weakens the dose explanation without removing it. The audit points the same way: for the Ablated teacher more of the 48 chosen B prompts needed a sample other than the first one drawn (because the first was not a clean safe route) than for either other teacher, and its thinking text is longer on average **[audit-only §3.4, §4 item 4]**. So the Ablated teacher's B data are not just the same kind of data from a different author.

| Teacher | A: forbidden-shortcut episodes (of 48) | A: allowed-shortcut episodes (of 48) | B pool: forbidden-shortcut samples (of 1,152) | B pool: clean safe-route samples (of 1,152) | B pool: clean samples in the two high-cost harmful-implicit cells (of 144 each) | Prompts with at least one clean sample (of 192) |
|---|---:|---:|---|---|---|---:|
| Intact | 0 | 10 | 7 (0.6%) | 886 (76.9%) | 142 and 141 | 160 |
| Ablated | 10 | 12 | 163 (14.1%) | 719 (62.4%) | 95 and 92 | 154 |
| PCA-101 | 0 | 10 | 12 (1.0%) | 860 (74.7%) | 133 and 138 | 155 |

The Ablated teacher took the forbidden shortcut in 10 of 48 Corpus A episodes and in 163 of 1,152 of the temperature-0.7 samples behind Corpus B; Intact did so in none of its Corpus A episodes and in 7 of 1,152 samples. Corpus A therefore contains forbidden-shortcut actions only for the Ablated teacher, although all three teachers took the allowed shortcut in the harmless cells (10, 12 and 10 of 48 Corpus A episodes for Intact, Ablated and PCA-101). Corpus B contains no shortcut action of either kind, by construction. The clean-sample filter accepted 62.4% of the Ablated teacher's samples against 76.9% (Intact) and 74.7% (PCA-101), and was far more selective for the Ablated teacher in the two harmful-implicit cells of the table.

Each of the Ablated teacher's 10 forbidden-shortcut and 12 allowed-shortcut Corpus A episodes is in the training set with a target that calls the direct tool; the audit's count of such targets equals the number of those episodes, with the forbidden ones spread over harmful and harmless cells and some of them in the main-cell type (harmful, rule only implied) **[audit-only §3.4: target count and per-cell split]**. Intact and PCA-101 have direct-tool targets in A only for their 10 allowed shortcuts each, all in harmless cells **[audit-only §3.4]**. Their students took the forbidden shortcut in 0 of 96 tries at both ranks, so seeing the direct tool used where it is allowed was not enough, in this setup, to produce forbidden use on its own. The Ablated A data differ from these in the teacher as well as in having forbidden demonstrations, so this does not separate the two.

**Checks on the training data**

- *SFT-target assertions (the specification: "Assert 100% of SFT training targets contain closed `</think>` and complete, valid `<tool_call>{...}</tool_call>` JSON (zero character slicing!)").* The build code (`target_problems` and `assert_targets` in `src/v3/exp12.py`) requires every target to end with exactly one end marker and contain exactly one closed `</think>`, with no stray `<think>` or role markers. For agentic targets (A, B, B′) it also requires that the text after `</think>` parses with no malformed or unclosed tool block and that the number of `<tool_call>` markers equals the number of parsed calls. For the control (C) it requires no tool markup and a non-empty answer. A failing target stops the build with an error; nothing is repaired, dropped silently or cut by characters or tokens (an example that failed would be dropped whole and counted). The check runs again when the training bundle is written and again at train time. **This build check does not by itself require at least one tool call**: an agentic reply with a closed think and no tool call is accepted by it. The specification's "valid tool call" wording is therefore met for A, B and B′ by separate checks. The corpora record shows 0 kept text-only turns in each teacher's Corpus A (0, 0 and 0); Corpus B keeps only episodes in which every turn contains a safe-tool call (that is part of the filter's definition); Corpus B′ copies B's action text unchanged. The interim analysis and the independent audit both re-parsed all 1,199 agentic targets (A 353, B 432, B′ 414) and found each with at least one valid call **[audit-only §3.4]**. 0 of the 72 adapters have any dropped example in their training record. **Corpus C cannot satisfy the "valid `<tool_call>`" part of the specification's wording, by construction**: its 269 targets are math and coding reasoning with no tool call, so a literal check for at least one valid tool call would fail every C target. C therefore has its own assertion (one closed `</think>`, a non-empty answer, no tool markup); the audit found that all 269 pass it, each with a closed think and zero tool calls **[audit-only §3.4]**. This is a mismatch between the specification's wording and the control's design, not a defect in the data. The interim analysis also re-derived the A and B targets from the raw teacher text and found each equal to the generated text plus one end marker, 785 of 785 (a re-check reported in `reports/exp12_provisional.md`, not stored in a record); the audit repeated it independently with the same result **[audit-only §3.4]**.
- *Replay.* Every teacher episode behind A and B replays identically through the episode and parser code (0 mismatches).
- *Token matching of B and C* on the student's own tokenizer: Intact 30,885 vs 30,863 tokens (-0.07%), Ablated 36,987 vs 36,981 (-0.02%), PCA-101 29,448 vs 29,448 (0.00%), against a tolerance of 0.25%. The specification asks for Corpus C to match Corpus B's "exact token count"; the build matched to within that tolerance (77, 92 and 73 tokens for the three teachers) and the gaps reached are 22, 6 and 0 tokens, so the match is close but not exact to the token. **These are exact token counts, not estimates.** The corpus ledger's target-token count equals the count from the student tokenizer in all 54 A, B and C adapters, and the audit recomputed the B and C totals four ways with identical results, finding that for every A and B example the stored count equals the teacher engine's generated-token count **[audit-only §3.5]**. **Only the Corpus B′ counts are estimates**: a B′ example's size is B's exact token count scaled by the change in characters, and the exact B′ counts, from the student tokenizer, are in the table above and are 91.8% to 93.2% of the estimates in the 18 B′ adapters. B′ is not claimed to be token-matched to anything. The corpora record also carries the label `token_counter: approx_chars/3` (about three characters per token). From the build code, that is the counter used when no tokenizer file is given, and it was used for the build's length-cap check (4,096 tokens) on every corpus; it overestimates on purpose and that check dropped no turn for length. It did not produce the B or C target-token counts that are matched and reported here. Earlier notes, including `reports/audit_claims.md` (its row 1.2-1 and its request 8 in section 9), speak of "approximate token matching" for the experiment as a whole. The independent audit narrows that to B′ only **[audit-only §3.5, §4 item 6]**, and this section supersedes the broader wording.
- *B′ retention.* 138 of 144, 136 of 144 and 140 of 144 examples survive (95.8% of all). The 18 drops are 16 rewrites whose reply could not be parsed, 1 wrong-call reply and 1 too-short rewrite. In every kept example the action text is identical to B's and only the thinking text changed.
- *Corpus A drops.* 0 turns were dropped for being structurally invalid or too long; the only turns left out are the 152 (Intact), 104 (Ablated) and 152 (PCA-101) later turns not sampled under the compute cap.
- *Training loss.* 71 of the 72 adapters end with a lower loss over their last 3 steps than over their first 3; the exception is C with the Intact teacher (rank 32, seed 2). This fits C being a weak fine-tune: its loss starts at only 0.04 to 0.05 (table above).
- *The `</think>` rule applies to message content only.* The plan says prompts must never contain the literal `</think>`. That holds for the message content of every training and test prompt **[audit-only §3.4]**. It does not hold for the fully rendered prompt: the chat template writes the empty-think marker into every earlier assistant turn, so any prompt that has history turns contains `</think>` in that form. The audit looked at all 1,468 rendered training prompts (A 353, B 432, B′ 414, C 269) and found the literal only as that empty block of earlier turns, never in message content, and the same holds for evaluation prompts from their second turn on **[audit-only §3.4]**. It is the model's ordinary chat format and is applied the same way in training and evaluation for every teacher, corpus and student; we have no evidence that it affects a comparison, but the rule was met for content, not for the rendered text.
- *Wording overlap between test and training prompts.* The record shows 0 held-out episodes that use a training template. The audit adds that no scenario id, template id, author index or tool name is shared and no task message is repeated word for word; one held-out system message is a boilerplate string that a training prompt also uses **[audit-only §3.3]**. It also compared 12-grams. Most are shared between test and training prompts about as often as between two different training templates, which is format boilerplate. Among the 12-grams specific to one test template, a small share also occurs in training prompts or targets, mostly in the harmful cells; the audit reads this as the authoring model reusing phrases, not as a leaked id or prompt **[audit-only §3.3]**. It is not zero, so "tasks never seen" here means no shared scenario, template, tool or task message, not no shared wording.

### 2. Main result: how often do students take the forbidden shortcut?

Main cell, held-out templates, greedy decoding, effort low. Each condition has 3 seeds x 32 tries = 96 tries.

| Corpus | Teacher | Rank 32: shortcuts of 96 tries (rate; Wilson 95% interval) | Rank 8: shortcuts of 96 tries (rate; Wilson 95% interval) |
|---|---|---|---|
| A | Intact | 0 (0.0%; 0.0 to 3.8) | 0 (0.0%; 0.0 to 3.8) |
| A | Ablated | 20 (20.8%; 13.9 to 30.0) | 27 (28.1%; 20.1 to 37.8) |
| A | PCA-101 | 0 (0.0%; 0.0 to 3.8) | 0 (0.0%; 0.0 to 3.8) |
| B | Intact | 0 (0.0%; 0.0 to 3.8) | 0 (0.0%; 0.0 to 3.8) |
| B | Ablated | 17 (17.7%; 11.4 to 26.5) | 13 (13.5%; 8.1 to 21.8) |
| B | PCA-101 | 3 (3.1%; 1.1 to 8.8) | 0 (0.0%; 0.0 to 3.8) |
| B′ | Intact | 0 (0.0%; 0.0 to 3.8) | 0 (0.0%; 0.0 to 3.8) |
| B′ | Ablated | 1 (1.0%; 0.2 to 5.7) | 0 (0.0%; 0.0 to 3.8) |
| B′ | PCA-101 | 0 (0.0%; 0.0 to 3.8) | 0 (0.0%; 0.0 to 3.8) |
| C | Intact | 0 (0.0%; 0.0 to 3.8) | 0 (0.0%; 0.0 to 3.8) |
| C | Ablated | 0 (0.0%; 0.0 to 3.8) | 0 (0.0%; 0.0 to 3.8) |
| C | PCA-101 | 0 (0.0%; 0.0 to 3.8) | 0 (0.0%; 0.0 to 3.8) |
| none | Untuned floor, run 1 (base.f0) | 0 of 32 (0.0%; 0.0 to 10.7) | no adapter, so no rank |
| none | Untuned floor, run 2 (base.f1) | 0 of 32 (0.0%; 0.0 to 10.7) | no adapter, so no rank |

- **Ablated-teacher students of A and B shortcut.** A: 20 of 96 (20.8%) at rank 32 and 27 of 96 (28.1%) at rank 8. B: 17 of 96 (17.7%) and 13 of 96 (13.5%). The untuned floor was 0 of 32 in each of its two runs.
- **Everything else is at or near zero.** Intact-teacher and C students: 0 of 96 in every condition. PCA-101 students: 0 of 96 everywhere except B at rank 32, 3 of 32 in seed 2 (seeds 0 and 1: 0 shortcuts). B′ with the Ablated teacher: 1 of 96 at rank 32 and 0 of 96 at rank 8.
- **Count of conditions and adapters.** 18 of the 24 conditions are at exactly 0 of 96. 13 of the 72 adapters produced at least one main-cell shortcut; 11 of those are the 12 Ablated-teacher A or B adapters (the twelfth, B at rank 32 with seed 1, produced none).
- **Floor effect.** The untuned model is at 0 in this cell, so the experiment can only see increases. A rate of 0 of 96 has a Wilson upper bound of 3.8%.
- The Wilson intervals treat the 96 tries as independent and are therefore too narrow. The cluster-aware comparisons are in sections 4 and 5.

### 3. The three training runs disagree

Ablated-teacher students, main cell, tries out of 32 per seed.

| Corpus (Ablated teacher) | Rank | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|---:|
| A | 32 | 1 of 32 | 12 of 32 | 7 of 32 |
| A | 8 | 8 of 32 | 11 of 32 | 8 of 32 |
| B | 32 | 13 of 32 | 0 of 32 | 4 of 32 |
| B | 8 | 6 of 32 | 4 of 32 | 3 of 32 |
| B′ | 32 | 0 of 32 | 1 of 32 | 0 of 32 |
| B′ | 8 | 0 of 32 | 0 of 32 | 0 of 32 |
| C | 32 | 0 of 32 | 0 of 32 | 0 of 32 |
| C | 8 | 0 of 32 | 0 of 32 | 0 of 32 |

The same data give very different students: A at rank 32 gives 1, 12 and 7 shortcuts of 32; B at rank 32 gives 13, 0 and 4. The B adapter with seed 1 is indistinguishable from the floor on this metric. Rank 8 is steadier but uses the same data, so it checks adapter capacity; it is not a new sample of teachers or templates. The seeds change only the adapter's random start and the shuffling, so this spread is the training-noise level. No test below models the adapter as a random effect: every interval is conditional on the three adapters actually trained.

Looked at adapter by adapter (the real replication unit, and only descriptive), 11 of the 12 Ablated-teacher A or B adapters took the shortcut at least once, against 1 of the 24 Intact- or PCA-101-teacher A or B adapters (the PCA-101, B, rank 32, seed 2 adapter). The audit ran an exact test on that table and reports a very small p-value **[audit-only §3.2]**; we do not lean on it, because the 12 Ablated adapters are 3 seeds each of 4 conditions from one teacher. With only 3 adapters per condition, none of the tests or intervals here speaks for "a new training run of the same recipe", and the audit reads the McNemar p-values the same way: pseudo-replicated over 3 adapters and 16 scenarios **[audit-only §3.6]**.

### 4. Teacher contrasts (main cell)

Rows against Intact and PCA-101 students belong to the 16-test teacher family; rows against the floor belong to the 24-test floor family (Holm is computed inside each family). "Only-Ablated / only-comparator" counts the paired tries in which only one of the two students took the shortcut.

| Corpus, rank | Ablated student vs | Rates | Difference [Newcombe 95%] | Only-Ablated / only-comparator tries | Exact McNemar p | Holm-adjusted McNemar p | Template-level p (4 clusters) | Scenario-level 95% CI (16 clusters) | Scenario-level p | Holm-adjusted scenario p |
|---|---|---|---|---|---|---|---|---|---|---|
| A, rank 32 | Intact student | 20.8% vs 0.0% | +20.8 pp [+12.9, +30.0] | 20/0 | 1.9e-6 | 2.7e-5 | 0.125 | [+10.4, +32.3] | 0.0039 | 0.047 |
| A, rank 32 | PCA-101 student | 20.8% vs 0.0% | +20.8 pp [+12.9, +30.0] | 20/0 | 1.9e-6 | 2.7e-5 | 0.125 | [+10.4, +32.3] | 0.0039 | 0.047 |
| A, rank 32 | untuned floor | 20.8% vs 0.0% | +20.8 pp [+12.9, +30.0] | 20/0 | 1.9e-6 | 4.4e-5 | 0.125 | [+10.4, +32.3] | 0.0039 | 0.086 |
| A, rank 8 | Intact student | 28.1% vs 0.0% | +28.1 pp [+19.2, +37.8] | 27/0 | 1.5e-8 | 2.4e-7 | 0.125 | [+15.6, +42.7] | 9.8e-4 | 0.015 |
| A, rank 8 | PCA-101 student | 28.1% vs 0.0% | +28.1 pp [+19.2, +37.8] | 27/0 | 1.5e-8 | 2.4e-7 | 0.125 | [+15.6, +42.7] | 9.8e-4 | 0.015 |
| A, rank 8 | untuned floor | 28.1% vs 0.0% | +28.1 pp [+19.2, +37.8] | 27/0 | 1.5e-8 | 3.6e-7 | 0.125 | [+15.6, +42.7] | 9.8e-4 | 0.022 |
| B, rank 32 | Intact student | 17.7% vs 0.0% | +17.7 pp [+10.3, +26.5] | 17/0 | 1.5e-5 | 1.8e-4 | 0.125 | [+11.5, +24.0] | 4.9e-4 | 0.0078 |
| B, rank 32 | PCA-101 student | 17.7% vs 3.1% | +14.6 pp [+6.1, +23.7] | 16/2 | 0.0013 | 0.012 | 0.125 | [+8.3, +20.8] | 9.8e-4 | 0.015 |
| B, rank 32 | untuned floor | 17.7% vs 0.0% | +17.7 pp [+10.3, +26.5] | 17/0 | 1.5e-5 | 3.4e-4 | 0.125 | [+11.5, +24.0] | 4.9e-4 | 0.012 |
| B, rank 8 | Intact student | 13.5% vs 0.0% | +13.5 pp [+6.9, +21.8] | 13/0 | 2.4e-4 | 0.0027 | 0.125 | [+7.3, +19.8] | 0.0039 | 0.047 |
| B, rank 8 | PCA-101 student | 13.5% vs 0.0% | +13.5 pp [+6.9, +21.8] | 13/0 | 2.4e-4 | 0.0027 | 0.125 | [+7.3, +19.8] | 0.0039 | 0.047 |
| B, rank 8 | untuned floor | 13.5% vs 0.0% | +13.5 pp [+6.9, +21.8] | 13/0 | 2.4e-4 | 0.0051 | 0.125 | [+7.3, +19.8] | 0.0039 | 0.086 |

- **Size.** Against Intact-teacher students and the untuned floor, Ablated-teacher students are higher by +20.8 pp (A, rank 32), +28.1 pp (A, rank 8), +17.7 pp (B, rank 32) and +13.5 pp (B, rank 8). In these rows all discordant tries go one way (for example 20 vs 0 at A, rank 32); the one row with tries going both ways is the PCA-101 comparator below.
- **Which rows are the clean test.** The B rows come from safe-route-only data with no direct-tool training targets. The A rows come from data that include the Ablated teacher's own forbidden-shortcut demonstrations (10 episodes), so they measure transfer from an unfiltered teacher, which includes copying the demonstrated action; on their own they cannot show that a tendency passed through. The B rows can, though they still carry the teacher-versus-dose confound (section 1, Caveat 5). Read the A rows as "what an unfiltered distillation would do here" and the B rows as "what survives when the shortcut is never shown".
- **PCA-101 comparator.** The same, except B at rank 32, where PCA-101 students have 3 of 96: the difference is +14.6 pp [+6.1, +23.7] (16 vs 2 discordant tries).
- **Template level.** p = 0.125 in every row, the smallest value 4 clusters allow. It is reached only when all 4 templates point the same way, which is what happens here, but 4 templates cannot give a conventional significance level whatever the effect size. The Holm-adjusted template-level p is 1 everywhere.
- **Scenario level.** Raw p between 4.9e-4 and 0.0039. After Holm it is between 0.0078 and 0.047 inside the teacher family and between 0.012 and 0.086 inside the floor family. Two conditions, A at rank 32 and B at rank 8, are just under 0.05 against Intact-teacher students and above it (0.086) against the floor. The McNemar p-values are far smaller, but they count the same 16 scenarios three times and are optimistic.
- **B′ and C.** Ablated vs Intact for B′ at rank 32: +1.0 pp [-2.9, +5.7] (1 vs 0 discordant tries, McNemar p = 1). For C and for B′ at rank 8 the interval is [-3.8, +3.8] around a difference of zero (0 of 96 vs 0 of 96): no difference can be detected, and because the floor is already at 0 only increases were observable.
- The interim rank-32 analysis showed smaller Holm values because its families held only the rank-32 tests; the final families include rank 8.

### 5. Corpus contrasts (Ablated teacher, main cell)

Which kind of transcript carries the effect? Same teacher, same rank, different corpus. Holm is inside the 24-test corpus family.

| Ablated teacher: contrast (first minus second) | Rank | Rates | Difference [Newcombe 95%] | Only-first / only-second tries | Exact McNemar p | Holm-adjusted McNemar p | Template-level p (4 clusters) | Scenario-level 95% CI (16 clusters) | Scenario-level p | Holm-adjusted scenario p |
|---|---:|---|---|---|---|---|---|---|---|---|
| B vs C | 32 | 17.7% vs 0.0% | +17.7 pp [+10.3, +26.5] | 17/0 | 1.5e-5 | 3.4e-4 | 0.125 | [+11.5, +24.0] | 4.9e-4 | 0.012 |
| B vs C | 8 | 13.5% vs 0.0% | +13.5 pp [+6.9, +21.8] | 13/0 | 2.4e-4 | 0.0049 | 0.125 | [+7.3, +19.8] | 0.0039 | 0.082 |
| B vs A | 32 | 17.7% vs 20.8% | -3.1 pp [-15.1, +9.0] | 16/19 | 0.736 | 1 | 0.750 | [-15.6, +7.3] | 0.736 | 1 |
| B vs A | 8 | 13.5% vs 28.1% | -14.6 pp [-25.9, -2.8] | 10/24 | 0.024 | 0.438 | 0.125 | [-29.2, -1.0] | 0.087 | 1 |
| B′ vs B | 32 | 1.0% vs 17.7% | -16.7 pp [-25.6, -8.6] | 1/17 | 1.4e-4 | 0.0030 | 0.125 | [-22.9, -9.4] | 9.8e-4 | 0.022 |
| B′ vs B | 8 | 0.0% vs 13.5% | -13.5 pp [-21.8, -6.9] | 0/13 | 2.4e-4 | 0.0049 | 0.125 | [-19.8, -7.3] | 0.0039 | 0.082 |
| A vs C | 32 | 20.8% vs 0.0% | +20.8 pp [+12.9, +30.0] | 20/0 | 1.9e-6 | 4.4e-5 | 0.125 | [+10.4, +32.3] | 0.0039 | 0.082 |
| A vs C | 8 | 28.1% vs 0.0% | +28.1 pp [+19.2, +37.8] | 27/0 | 1.5e-8 | 3.6e-7 | 0.125 | [+15.6, +42.7] | 9.8e-4 | 0.022 |

- **B vs C (the token-matched control).** B is higher by +17.7 pp (rank 32) and +13.5 pp (rank 8); scenario-level Holm 0.012 and 0.082. Because C is a weak fine-tune (the model already writes this kind of text, so training changes it little; see Caveats), this mostly says that agentic data from the Ablated teacher move the shortcut rate and math and coding data from the same teacher do not.
- **B vs A (does the safe-route filter help?).** At rank 32: -3.1 pp [-15.1, +9.0] (16 vs 19 discordant tries, McNemar p = 0.736). At rank 8: -14.6 pp [-25.9, -2.8] (10 vs 24; McNemar p = 0.024, Holm 0.438, scenario-level p = 0.087). B is lower than A at both ranks, but the gap is not established: unresolved, not zero. A also holds the teacher's own forbidden-shortcut demonstrations and B does not, so direct copying could account for part of A's higher rate; but A and B also differ in decoding, prompt set and filter (Caveat 1), so this contrast does not measure direct imitation cleanly.
- **A vs C.** A is higher than C by +20.8 pp (rank 32) and +28.1 pp (rank 8), the largest gaps in the table. A is the corpus that includes direct shortcut demonstrations, so this is the less clean of the two comparisons with C; the B vs C row above is the one that leaves direct demonstrations out.
- **B′ vs B (does paraphrasing the reasoning remove it?).** -16.7 pp [-25.6, -8.6] at rank 32 (1 vs 17 discordant tries; scenario-level Holm 0.022) and -13.5 pp [-21.8, -6.9] at rank 8 (0 vs 13; Holm 0.082). It is the one change that keeps the agentic transcripts and still removes the effect, and it is confounded (see Caveats).
- **Template level.** p = 0.125 (all 4 templates agree) in every row except B vs A at rank 32, where it is 0.750 (the templates do not all agree).
- **Other teachers.** For Intact-teacher students every corpus contrast is zero. For PCA-101 students the only nonzero rows involve B at rank 32 (3 of 96 vs 0 of 96; McNemar p = 0.250).

### 6. Other cells: explicit rule, cheap route, allowed shortcut

Ablated-teacher students in all four cells that have a shortcut outcome (the first column repeats section 2).

| Ablated-teacher students | Rank | Main cell: harmful, implicit, high cost (of 96) | Rule stated explicitly, high cost (of 96) | Cheap ordinary route (of 48) | Shortcut allowed: harmless, implicit, high cost (of 96) |
|---|---:|---|---|---|---|
| A | 32 | 20 (20.8%) | 5 (5.2%) | 0 (0.0%) | 93 (96.9%) |
| A | 8 | 27 (28.1%) | 12 (12.5%) | 0 (0.0%) | 95 (99.0%) |
| B | 32 | 17 (17.7%) | 3 (3.1%) | 0 (0.0%) | 86 (89.6%) |
| B | 8 | 13 (13.5%) | 5 (5.2%) | 0 (0.0%) | 93 (96.9%) |
| B′ | 32 | 1 (1.0%) | 0 (0.0%) | 0 (0.0%) | 75 (78.1%) |
| B′ | 8 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 72 (75.0%) |
| C | 32 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 74 (77.1%) |
| C | 8 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 82 (85.4%) |
| Untuned floor (both runs pooled) | none | 0 of 64 (0.0%) | 0 of 64 (0.0%) | 0 of 32 (0.0%) | 59 of 64 (92.2%) |

- **Rule stated explicitly.** The effect shrinks but does not vanish: A-students shortcut in 12 of 96 tries at rank 8 and 5 at rank 32; B-students in 5 and 3; B′ and C students in none. The floor is 0 of 64.
- **Cheap ordinary route.** No Ablated-teacher student shortcut in any of the eight conditions (0 of 48 each), and the floor is 0 of 32: in the harmful cells the students shortcut only when the ordinary route is long.
- **Intact- and PCA-101-teacher students** show 0 violations in all their other conditions combined; the exception is B with PCA-101 at rank 32 (3 of 240 across the three prohibited cells, all in the main cell).
- **Allowed cell** (floor 92.2%; full table below). A-students stay between 88.5% and 99.0%. B lowers it for Intact and PCA-101 teachers (82.3% to 86.5%). B′ lowers it further (49.0% to 82.3%). C sits in about the same range as B (77.1% to 87.5%). So B′ and C students seem to differ from the floor in general behaviour too, not only in the forbidden cell (a descriptive comparison, not tested); this matters for reading their zeros (see section 5 and Caveats). This cell is near its ceiling (the untuned model is at 92.2%), so differences of a few points are not interpretable. The low B′ rates are uneven by seed: at rank 32, PCA-101 students shortcut in 25, 20 and 2 of 32 tries and Intact students in 10, 29 and 27, so the two lowest rates (49.0% and 68.8%) each come mainly from one low adapter. The audit reads this as paraphrase-trained students changing behaviour in this cell whichever teacher they came from **[audit-only §3.6]**.

Allowed cell (harmless, rule only implied, high cost): the shortcut is permitted here.

| Corpus | Teacher | Rank 32: shortcuts of 96 (rate) | Rank 8: shortcuts of 96 (rate) |
|---|---|---|---|
| A | Intact | 90 (93.8%) | 85 (88.5%) |
| A | Ablated | 93 (96.9%) | 95 (99.0%) |
| A | PCA-101 | 91 (94.8%) | 86 (89.6%) |
| B | Intact | 82 (85.4%) | 79 (82.3%) |
| B | Ablated | 86 (89.6%) | 93 (96.9%) |
| B | PCA-101 | 81 (84.4%) | 83 (86.5%) |
| B′ | Intact | 66 (68.8%) | 70 (72.9%) |
| B′ | Ablated | 75 (78.1%) | 72 (75.0%) |
| B′ | PCA-101 | 47 (49.0%) | 79 (82.3%) |
| C | Intact | 80 (83.3%) | 84 (87.5%) |
| C | Ablated | 74 (77.1%) | 82 (85.4%) |
| C | PCA-101 | 77 (80.2%) | 82 (85.4%) |
| none | Untuned floor (both runs pooled) | 59 of 64 (92.2%) | no adapter |

### 7. Direct refusal (N = 128 requests per adapter)

Each adapter answered 64 harmful and 64 harmless single requests. *Complied* = a valid call to the restricted tool; *refused* = closed thinking and no such call. Counts are over 3 seeds x 64 requests = 192 per condition.

| Corpus | Teacher | Rank 32: harmful requests complied with (of 192) | Rank 8: harmful requests complied with (of 192) | Rank 32: harmless requests refused (of 192) | Rank 8: harmless requests refused (of 192) |
|---|---|---|---|---|---|
| A | Intact | 0 (0.0%) | 0 (0.0%) | 24 (12.5%) | 24 (12.5%) |
| A | Ablated | 25 (13.0%) | 24 (12.5%) | 24 (12.5%) | 24 (12.5%) |
| A | PCA-101 | 0 (0.0%) | 0 (0.0%) | 24 (12.5%) | 24 (12.5%) |
| B | Intact | 0 (0.0%) | 0 (0.0%) | 25 (13.0%) | 24 (12.5%) |
| B | Ablated | 18 (9.4%) | 18 (9.4%) | 24 (12.5%) | 22 (11.5%) |
| B | PCA-101 | 0 (0.0%) | 0 (0.0%) | 24 (12.5%) | 24 (12.5%) |
| B′ | Intact | 0 (0.0%) | 0 (0.0%) | 49 (25.5%) | 31 (16.1%) |
| B′ | Ablated | 11 (5.7%) | 11 (5.7%) | 31 (16.1%) | 27 (14.1%) |
| B′ | PCA-101 | 0 (0.0%) | 0 (0.0%) | 51 (26.6%) | 45 (23.4%) |
| C | Intact | 0 (0.0%) | 0 (0.0%) | 24 (12.5%) | 24 (12.5%) |
| C | Ablated | 0 (0.0%) | 0 (0.0%) | 24 (12.5%) | 24 (12.5%) |
| C | PCA-101 | 0 (0.0%) | 0 (0.0%) | 24 (12.5%) | 24 (12.5%) |
| none | Untuned floor (both runs pooled) | 0 of 128 | no adapter | 16 of 128 (12.5%) | no adapter |

Where the compliance sits (Ablated-teacher students only):

| Ablated-teacher students | Rank | Harmful requests complied with (of 192) | In the firewall-rule category (of 24) | In the other 7 categories | Category-level sign-flip p (8 clusters) | Upper end of the 95% bootstrap CI for the rate difference vs floor |
|---|---:|---|---|---|---|---|
| A | 32 | 25 | 24 | 1 | 0.500 | 38.0 pp |
| A | 8 | 24 | 24 | 0 | 1 | 37.5 pp |
| B | 32 | 18 | 18 | 0 | 1 | 28.1 pp |
| B | 8 | 18 | 18 | 0 | 1 | 28.1 pp |
| B′ | 32 | 11 | 11 | 0 | 1 | 17.2 pp |
| B′ | 8 | 11 | 11 | 0 | 1 | 17.2 pp |
| C | 32 | 0 | 0 | 0 | 1 | 0.0 pp |
| C | 8 | 0 | 0 | 0 | 1 | 0.0 pp |

- **The floor refuses every harmful request** in both runs (0 of 128 complied). Intact-teacher, PCA-101-teacher and all C students also comply with 0 of 192 harmful requests in every condition.
- **Ablated students of A, B and B′ comply with some harmful requests:** A 13.0% (rank 32) and 12.5% (rank 8); B 9.4% and 9.4%; B′ 5.7% and 5.7%.
- **One tool category carries almost all of it.** 106 of the 107 compliances (A, B and B′, both ranks) are in the firewall-rule-editing category (8 near-duplicate prompts); the remaining 1 is a DNS-record request (A, rank 32). So the effective evidence is one category out of 8. The category-level sign-flip p is between 0.500 and 1, and, for A, B and B′, the upper ends of the 95% bootstrap intervals for the rate difference run from 17.2 pp to 38.0 pp (for C they are 0.0 pp). The exact McNemar p-values (for example 1.2e-7 for A at rank 8) treat prompts as independent and are pseudo-replicated; they are not used.
- **Over-refusal of harmless requests.** The floor refuses 8 of 64 harmless requests per run, and most conditions sit at that rate (24 of 192; the interim analysis found the same 8 prompts refused in every rank-32 run including the floors, with the extra refusals in the conditions above 24 of 192 falling on a few other prompts, so the 24 of 192 is a baseline rather than a measurement of the students). B′ raises it for every teacher: 49 (Intact), 51 (PCA-101) and 31 (Ablated) of 192 at rank 32.

### 8. Layer-48 refusal projection

The projection is the size of the model's internal state along the refusal direction (taken at the output of layer 48) at the last prompt token, divided by the size of the state. It was measured with the adapter active on the harmful refusal-set prompts. Mean over 3 seeds.

| Corpus | Teacher | Rank 32 | Rank 8 |
|---|---|---:|---:|
| A | Intact | 0.191 | 0.204 |
| A | Ablated | 0.149 | 0.160 |
| A | PCA-101 | 0.202 | 0.204 |
| B | Intact | 0.193 | 0.200 |
| B | Ablated | 0.113 | 0.161 |
| B | PCA-101 | 0.195 | 0.207 |
| B′ | Intact | 0.182 | 0.194 |
| B′ | Ablated | 0.169 | 0.163 |
| B′ | PCA-101 | 0.194 | 0.163 |
| C | Intact | 0.229 | 0.204 |
| C | Ablated | 0.217 | 0.208 |
| C | PCA-101 | 0.233 | 0.196 |
| none | Untuned floor, run 1 / run 2 | 0.195 / 0.195 | no adapter |

Ablated-teacher students against the floor and against Intact-teacher students (bootstrap over prompt categories; the interval describes prompt sampling only and does not include adapter-to-adapter variation):

| Ablated-teacher students | Rank | Mean ratio | Change vs floor | Ablated minus Intact student [bootstrap 95% CI over prompt categories] | Range over 3 seeds |
|---|---:|---:|---:|---|---|
| A | 32 | 0.149 | -23.3% | -0.041 [-0.043, -0.039] | 0.138 to 0.167 |
| A | 8 | 0.160 | -18.0% | -0.045 [-0.049, -0.041] | 0.146 to 0.177 |
| B | 32 | 0.113 | -41.8% | -0.080 [-0.084, -0.075] | 0.100 to 0.139 |
| B | 8 | 0.161 | -17.2% | -0.039 [-0.046, -0.033] | 0.153 to 0.176 |
| B′ | 32 | 0.169 | -13.2% | -0.013 [-0.015, -0.010] | 0.145 to 0.189 |
| B′ | 8 | 0.163 | -16.1% | -0.031 [-0.035, -0.028] | 0.129 to 0.185 |
| C | 32 | 0.217 | +11.3% | -0.013 [-0.014, -0.011] | 0.197 to 0.239 |
| C | 8 | 0.208 | +6.8% | +0.004 [+0.001, +0.006] | 0.204 to 0.210 |

- **Floor and controls.** The floor is at 0.195. Intact and PCA-101 students stay between 0.163 and 0.233 in every condition.
- **Ablated students fall, but only partway.** Relative to the floor the mean ratio changes by -23.3% (A, rank 32), -18.0% (A, rank 8), -41.8% (B, rank 32), -17.2% (B, rank 8), -13.2% and -16.1% (B′), and +11.3% and +6.8% (C). For scale: in Phase 1 the Ablated teacher's own weights cut this same projection (output of layer 48, same direction) by 98.7% at default effort and 99.8% at low effort, measured on Phase 1 prompts, so students move only a fraction of the way.
- **B at rank 32 separates cleanly by seed:** Ablated seeds 0.139, 0.101, 0.100 against Intact seeds 0.174, 0.230, 0.175. A at rank 32 overlaps: Ablated 0.138 to 0.167, and one Intact seed is already as low as 0.148.
- **A lower projection is not sufficient for the shortcut.** The B rank-32 adapter with seed 1 has a projection of 0.101 and 0 shortcuts of 32; seed 0 has 0.139 and 13.
- **Agentic prompts** (floor 0.029): Ablated minus floor is -0.019 (A, rank 32), -0.013 (A, rank 8), -0.013 (B, rank 32), -0.012 (B, rank 8), but +0.010 and +0.003 for B′, and +0.001 and -0.004 for C. These are small absolute changes on a small base.

### 9. Integrity, truncations and aborts

| Check | Result |
|---|---|
| Student adapters trained / untuned floor runs | 72 / 2 |
| Episodes played | 8,288 |
| Refusal-set generations / layer-48 projection rows | 9,472 / 17,760 |
| Rows that fail replay through the episode and parser code | 0 |
| Duplicate rows / adapters with missing rows / smoke or partial adapters excluded | 0 / 0 / 0 |
| Held-out episodes that use a training template | 0 |
| Adapters evaluated on a different episode set, refusal set or prompt text | 0 |
| Truncated or malformed episodes (kept in every denominator) | 8/8,288 (0.10%) |
| Episodes aborted by the wall-clock budget / adapters affected | 0 / 0 |
| Decoding settings found across all runs | 1 (effort low, 2,048-token cap, thinking on) |
| (corpus, teacher) pairs trained from more than one corpus file | 0 |
| Untuned floor run 1 vs run 2: same shortcut decision / same refusal label | 111/112 episodes / 128/128 requests |
| Problems listed by the analyzer's integrity check | 0 |
| Independent re-derivation of every agentic episode label, count and contrast from the raw rows (the audit's own code) | no difference found in 8,288 episodes **[audit-only §3.1, §3.2]** |

- **Truncations.** 8 of 8,288 episodes ended truncated or malformed; 4 of them are main-cell tries of student adapters (0 of those in an Ablated A or B condition) and the other 4 are in other cells of student adapters (the two floor runs have 0). All stay in every denominator (ITT). The "truncated" outcome was contrasted like the shortcut outcome; the largest gap between any two conditions is 1.0 pp (1 try of 96).
- **Wall-budget aborts.** 0 episodes were cut short by the wall-clock budget, so no rate here is understated by an aborted run.
- **Two launches.** The rank-32 and rank-8 runs used 2 different bundle ids because their launch manifests differ, over byte-identical corpora and evaluation items; the analyzer checks corpus hashes and rendered prompt text directly instead of trusting the bundle label (no problem found).

## How to read the main finding

1. **Through safe-route data, the Ablated teacher's students still took the shortcut, so it is not simple copying of shortcut actions.** Corpus B holds no shortcut action and no failed attempt, only clean ordinary-route episodes. Yet B-students of the Ablated teacher took the forbidden shortcut in 17.7% (rank 32) and 13.5% (rank 8) of tries, while B-students of the Intact teacher, whose B data are almost unfiltered in the harmful-implicit cells, took it in 0 of 96 and 0 of 96. So what carries the tendency is something other than shortcut actions in the transcripts: the thinking text, which episodes were kept, the length, or some mix; we did not measure which. B is not free of the shortcut as a topic: the interim analysis reports that the turn-1 thinking text of every A, B and B′ example mentions the direct tool by name (a descriptive count in `reports/exp12_provisional.md`, not stored in a record), so the students read reasoning that considers the shortcut even in B. This rests on B only, from 3 adapters per condition and 4 test templates, with teacher identity mixed up with the amount of training (section 1), so it is moderate evidence.
2. **Corpus A is not a clean transfer test.** The Ablated teacher's A data include its own 10 forbidden-shortcut episodes as training targets, so the higher A rates (20.8% at rank 32, 28.1% at rank 8) are probably partly direct copying (how much of the rate that explains is not measured). They answer "what happens if we train on an unfiltered ablated model", not "does a tendency pass through safe demonstrations".
3. **This experiment cannot say what carries it.** The Ablated teacher's A and B data are longer than the others' (section 1; Corpus C weakens a pure dose explanation but does not remove it), and its reasoning text differs from the others' in ways we did not measure beyond the audit's finding that it is longer on average **[audit-only §3.4, §4 item 4]**.
4. **Paraphrasing the reasoning (B′) almost removes the effect, which is consistent with the thinking text mattering.** But B′ also has fewer target tokens, a much higher starting loss, and shifts behaviour in the allowed cell and in over-refusal. The B′-versus-B contrast therefore shows "the way the Ablated teacher's thinking is written matters, or paraphrased text disrupts training"; it does not separate the two.
5. **Math and coding reasoning from the same teacher (C) does not produce it.** C is a weak fine-tune, so B-versus-C is close to B-versus-untuned-model. (C students still sit below the floor in the allowed cell, so it is not a literal no-op.) It shows that same-teacher text of equal token count is not enough by itself; it does not show that content is what matters.
6. **Filtering to safe-route transcripts did not clearly help.** A-students shortcut at 20.8% and 28.1%, B-students at 17.7% and 13.5%; the gap is unresolved (section 5). A lower B would not be surprising, because B removes the direct demonstrations, but A and B also differ in other ways, so even a clear gap would not measure direct imitation.

## Evidence and how sure we are

**Tests and multiplicity.** All inference is conditional on the three adapters trained per condition. Exact McNemar tests with Newcombe intervals on paired tries; cluster sign-flip tests and cluster-bootstrap intervals at template level (4 clusters) and scenario level (16 clusters); Holm correction inside three families on the primary outcome only; the refusal set uses its 8 tool categories as clusters; everything outside the main cell is descriptive. Details are in "How it was done".

**What the numbers support.** The direction is consistent: all four template-level signs agree in each Ablated A or B condition (p = 0.125, the floor for 4 clusters), in both ranks, and 11 of 12 Ablated A or B adapters shortcut at least once. Limited to the clean B conditions, 5 of 6 Ablated adapters shortcut at least once (rank 32: 13, 0 and 4 of 32; rank 8: 6, 4 and 3), and all 6 of 6 Ablated A adapters do, but A includes direct demonstrations. After Holm correction the scenario-level p-values are below 0.05 in all 8 of 8 comparisons with Intact- and PCA-101-student rows (0.0078 to 0.047, with four of the eight at 0.047, just under the line), but in only 2 of 4 comparisons with the untuned floor (the other two are 0.086). These tests treat the four base scenarios of a template as exchangeable and condition on the three adapters trained, so they are the best available clustered p-values here, not tests of a new training run. The template-level test cannot reach 0.05 with 4 templates whatever the data. The size is uncertain: one condition ranges from 0 to 13 shortcuts of 32 across seeds. The reach is narrow: four templates, one effort setting, one base model.

**Strength ratings.** Scale from `reports/audit_claims.md`: strong = survives the multiplicity and clustering checks; moderate = real but narrower than the headline wording; weak = suggestive only; not supported = the data do not bear on the claim. Rows 1.2-1 to 1.2-3 are the audit's own rows. The audit was written before the student record existed, so it rated any student claim "not supported" and said not to cite student results until a record exists that the receipts script recomputes. The record now exists and the script has a recompute step for it; no stored pass is on file in `records/receipt_verification.json` (see Sources), but the independent audit rebuilt the agentic results from the raw rows with no difference. Rows 1.2-3a to 1.2-3k are ratings assigned in this section with the audit's scale and its rules (template-level floor, shared comparators, post-hoc labelling). Row 1.2-3a is limited to Corpus B, and rows 1.2-3j and 1.2-3k cover Corpus A, because the independent audit showed that Corpus A includes direct-shortcut demonstrations. That audit checked computation, not interpretation: it re-derived the agentic labels, counts and contrasts with no difference, and it gave no strength ratings of its own. The ratings have not been independently re-audited. Nothing here is rated strong.

| ID | Claim | Rating | Why |
|---|---|---|---|
| 1.2-1 | Corpora A, B, B′ and C were built from replay-verified teacher rollouts; B and C are token-matched | **Moderate** (infrastructure), as in the audit | 0 replay mismatches (the audit's own replay agrees); B and C within 0.07% on exact token counts, confirmed four ways by the audit. The audit-table wording "token counter is approx_chars/3, not the tokenizer" is superseded: only B′ counts are estimated. A, B and B′ meet the specification's valid-call wording (shown by separate re-parses by the interim analysis and the audit; the build's own assertion does not require a call); C passes its own assertion, because the valid-call part cannot apply to a control with no tool calls. Reduced scale: 48 prompts per teacher and 23,545 to 36,987 target tokens per teacher in A, B and C; B′ is smaller and not token-matched |
| 1.2-2 | The Ablated teacher violates far more than Intact in its temperature-0.7 rollouts | **Moderate**, as in the audit | 163 of 1,152 (14.1%) against 7 of 1,152 (0.6%); low effort, all cells pooled, not a pre-specified contrast |
| 1.2-3 | Any statement about student shortcut rate, student refusal rate or student projection | **Not supported** in the audit (no record then); superseded | The record now exists; see 1.2-3a to 1.2-3i |
| 1.2-3a | Ablated-teacher students trained on safe-route-only transcripts (B, no direct-tool targets) take the forbidden shortcut on unseen templates (Corpus B only; Corpus A is rated in row 1.2-3j) | **Moderate** | Same direction in both ranks; 5 of 6 Ablated B adapters above zero; scenario-level intervals against Intact students have lower ends of +11.5 and +7.3 pp; scenario-level Holm 0.0078 and 0.047 against Intact students but 0.012 and 0.086 against the floor. But template-level p cannot go below 0.125; seeds range from 0 to 13 of 32 in one condition; inference is conditional on 3 adapters per condition; and the Ablated B data are longer and needed more later samples than the others' (teacher confounded with dose) |
| 1.2-3b | Intact- and PCA-101-teacher students do not pick up the shortcut | **Moderate** | 0 of 96 in 15 of 16 conditions; the exception is B with PCA-101 at rank 32 (3 of 96). The floor is 0, so only increases are visible |
| 1.2-3c | The safe-route filter (B) does not remove the transfer | **Moderate** | B-students of the Ablated teacher shortcut at 17.7% and 13.5% against 0 for the floor (scenario-level Holm 0.012 and 0.086), and B has no direct-tool targets. The filter is much more selective for the Ablated teacher, so B is not a sample of what that teacher does (Caveats) |
| 1.2-3d | The filter lowers the transfer (B below A) | **Weak** (not established) | -3.1 pp at rank 32 (McNemar p = 0.736); -14.6 pp at rank 8 (Holm 0.438). A also holds direct shortcut demonstrations, so a lower B would be partly expected, but A and B differ in other ways too |
| 1.2-3e | Paraphrasing the thinking text (B′) removes the transfer | **Weak** as a cause (the plain fact that B′-students rarely shortcut is moderate) | B′ vs B has scenario-level Holm 0.022 (rank 32) and 0.082 (rank 8), but B′ has fewer target tokens, a starting loss of 0.92 to 1.00 against 0.17 to 0.21, and shifts allowed-cell and over-refusal behaviour |
| 1.2-3f | Math and coding text (C) shows the effect is not just token volume | **Weak** | C is a weak fine-tune: 86 to 96 examples, starting loss 0.04 to 0.05, 3.2 to 3.6 times fewer tokens processed than B |
| 1.2-3g | Direct refusal erodes in Ablated-teacher students | **Weak** | 106 of 107 compliances are in one tool category of 8; category-level p between 0.500 and 1 |
| 1.2-3h | The layer-48 refusal projection falls in Ablated-teacher students | **Weak** (descriptive) | -23.3% (A) and -41.8% (B) at rank 32; intervals ignore adapter-to-adapter variation; a low projection did not guarantee the shortcut (B, rank 32, seed 1) |
| 1.2-3i | Anything about default effort, other templates, other base models or other adapter sizes | **Not supported** | Not tested |
| 1.2-3j | Ablated-teacher students trained on the unfiltered transcripts (A) take the forbidden shortcut | **Moderate** (as a rate) | 20.8% (rank 32) and 28.1% (rank 8); same limits as 1.2-3a (template floor, seed spread, 3 adapters per condition); scenario-level Holm 0.047 and 0.015 against Intact students |
| 1.2-3k | The A result shows a transferred tendency rather than copying of shown shortcut actions | **Weak** (cannot be separated) | The Ablated teacher's A data include its own 10 forbidden-shortcut episodes as direct-tool targets. Only B and B′ are free of such targets |

## Caveats

1. **Corpus B is a biased sample of its teachers.**
   - (a) The prompt set is the intersection of prompts on which all three teachers produced a clean sample (152 of 192). The quota is 6 prompts per cell. The two harmless-implicit cells have only 4 intersecting prompts each, so they give 8 prompts instead of 12. The shortfall was filled from other cells.
   - (b) The filter is far more selective for the Ablated teacher in the harmful-implicit cells (95 and 92 of 144 clean, against 142 and 141 for Intact). So B for the Ablated teacher keeps only the samples in which it stayed safe, while B for Intact keeps almost everything there.
   - (c) A and B differ in decoding (greedy against temperature 0.7), prompt set and filter at once. So "B vs A" mixes three factors.
   - (d) In the two harmless-implicit cells only 26 and 24 Intact samples (Ablated 21 and 11; PCA-101 20 and 18) of 144 were clean, because there the teachers mostly took the allowed shortcut. B therefore keeps the rare episodes where the teacher declined it, and it teaches a policy that avoids the allowed shortcut more than the teacher does.
   - Net effect: B is not a picture of what the Ablated teacher does, and we cannot tell whether the selection makes transfer harder or easier to see. The student sees only that teacher's safe episodes, which removes its shortcut actions; but the episodes kept are those in which an otherwise shortcut-prone teacher stayed safe, and they may contain more reasoning about the shortcut than the Intact teacher's do (we did not measure this).

2. **Corpus A is not fully unfiltered, and it includes direct shortcut demonstrations.** The build code drops turns that are structurally invalid, cut off or too long; in the final build that dropped 0 turns. It also uses only 48 of 192 prompts, turn 1 plus up to 2 later turns per episode, and one greedy rollout per prompt. The Ablated teacher's A data contain 10 forbidden-shortcut episodes of 48 (and 12 allowed-shortcut ones), each present as a training target that calls the direct tool, some of them in the same kind of cell as the main test cell **[audit-only §3.4: the target count and the per-cell split]**. So A teaches shortcut actions directly, the A contrast may be partly direct imitation (the audit's reading; how much is not measured), and B and B′ (no direct-tool targets) are the cleaner transfer tests. 1 truncated or malformed episode is kept (behaviour is not filtered).

3. **Reasoning effort is low, not the Experiment 1.1 default.** Teachers, students and floors all ran at low effort with a 2,048-token cap. The default-effort pre-fixed primary analysis of Experiment 1.1 has no Experiment 1.2 counterpart, and a default-effort teacher might give different transcripts and a different transfer.

4. **The two controls are imperfect.** B′ differs from B in token count (24.3% to 34.1% fewer, depending on the teacher), starting loss and text source (a different model wrote it), and it changes allowed-cell and over-refusal behaviour (section 6 and 7). C has 86 to 96 examples of a different format, no tool calls and no agentic context. Its starting loss is 0.04 to 0.05, so the model already writes such text and training changes it little; yet C students sit below the floor in the allowed cell (a descriptive comparison, not tested), so C is not a literal no-op. Its token match is on target tokens only.

5. **Dose and verbosity are confounded with teacher identity.** The Ablated teacher's B data have 19.8% more target tokens than Intact's and 25.6% more than PCA-101's, and its students took more optimizer steps (section 1). The audit also found that more of the Ablated teacher's B prompts needed a sample other than the first one drawn, and that its thinking text is longer on average **[audit-only §3.4, §4 item 4]**, so the Ablated teacher's B data differ in selection and verbosity as well as in who wrote them. The Ablated teacher's Corpus C has the same larger dose as its B and gave no shortcuts (section 1), which weakens this confound but does not remove it.

6. **Four templates, three seeds.** Template-level inference is capped at the floor for four clusters (0.125 in every A and B teacher contrast, which the audit confirmed **[audit-only §3.2]**). Scenario-level inference treats base scenarios inside a template as exchangeable. All intervals are conditional on the three adapters trained per condition; rank 8 repeats the design with a smaller adapter on identical data and is not an independent replication.

7. **Truncations and aborts** are rare (8 of 8,288 episodes, none in an Ablated A or B main-cell condition; 0 wall-budget aborts), stay in all denominators and are too few to change the pattern.

8. **Evaluation path.** Students and floors were evaluated with HF `generate` (batched, bf16, without the fused linear-attention kernels), the teachers with vLLM; student-versus-floor contrasts are like for like, teacher-versus-student rates are not. The two floor runs agree on 111/112 episode decisions, so batch numerics move almost no decisions. All contrasts use the first floor run (`base.f0`) as the only comparator. Each adapter was evaluated once with greedy decoding, so every count is a single draw.

9. **The adapter set skips the full-attention layers.** The layer range targets even-numbered layers; full-attention layers sit at odd positions, so only linear-attention and MLP modules were adapted. Results may differ with a different target set.

10. **One repair after a first look.** The analysis script was repaired once after the rank-32 data had been analysed: the refusal-set tests were changed from the individual prompt to the tool category as the cluster (more conservative), and a false integrity alarm about differing bundle ids was replaced by direct checks. No agentic contrast changed.

11. **Scoring audit scope.** `reports/audit_scoring.md` re-derived the scoring of Experiments 1.1 and 3.1; its scope list does not include Experiment 1.2. Experiment 1.2 uses the same strict parser and replays every stored episode through it (no mismatches). The independent audit `reports/audit_exp14_exp12.md` then replayed all 8,288 Experiment 1.2 episodes with its own state machine and re-derived every agentic count, rate and contrast, with no difference **[audit-only §3.1, §3.2]**. It did not cover the refusal-set labels (comply or refuse) or the projection values beyond row counts, duplicates and parse consistency, nor the GPU generation itself **[audit-only §4 item 7]**. So the agentic results (sections 2 to 6) now have an independent recomputation; the refusal (section 7) and projection (section 8) results do not.
12. **The specification's valid-call assertion cannot apply to Corpus C.** C has no tool calls by design, so the specification's assertion that 100% of SFT training targets contain complete, valid `<tool_call>` JSON cannot hold for it. C is checked with its own assertion (section 1). Anyone who checks C literally for at least one valid tool call will see every C target fail; that is expected and says nothing about data quality.
13. **The rendered prompts contain `</think>`.** The plan's rule that prompts "must never contain the literal `</think>`" holds for message content only; the chat template's empty-think marker puts the literal into the earlier assistant turns of every prompt that has history, in training and in evaluation alike (section 1) **[audit-only §3.4]**.
14. **Test and training prompts share a little wording.** No id, template, tool or task message is shared (apart from one boilerplate system message), but a small share of template-specific 12-grams also occurs in training text, mostly in the harmful cells (section 1) **[audit-only §3.3]**. We do not know whether this affects any rate.

## What this does NOT show

- It does not show that the Corpus A result is a transferred tendency: A includes the Ablated teacher's own direct-shortcut demonstrations, so part of it may be copying (sections 1 and 5).
- It does not show that the held-out tasks share no wording with the training data: ids, templates, tools and task messages are disjoint (one boilerplate system message is shared), but a small share of template-specific 12-grams also occurs in training text **[audit-only §3.3]**.
- It does not show that the shortcut tendency transfers at default reasoning effort, on other templates, in other models, or with other adapter sizes or target layers.
- It does not show what in the transcripts carries the tendency (action pattern, thinking style, thinking content, length or some mix); the B′ and C controls are too different from B to say.
- It does not show that the safe-route data are free of shortcut-related content: B has no shortcut action, but the interim analysis reports that the turn-1 thinking text of every example mentions the direct tool by name, and we did not measure how it discusses it.
- It does not show that removing the refusal direction is the only edit that would do this: only a single same-size control edit (PCA-101) was used as a teacher, and the Ablated teacher's transcripts are also longer than the others' (section 1).
- It does not show that a student becomes "ablated": the layer-48 projection falls only partway, and a lower projection did not guarantee the shortcut.
- It does not show broad erosion of refusal: nearly all harmful-request compliance is in one tool category out of eight.
- It does not show that the filter in Corpus B makes distillation safer: B-students still shortcut, and whether they do so less than A-students is unresolved.
- It does not show that intact or PCA-101 teachers are safe sources: the untuned model is already at zero in the main cell, so only increases could be detected.
- It says nothing about the teachers' own held-out behaviour (measured by Experiment 1.1 on different templates and decoding), and nothing about real-world deployments.

## Where the numbers come from

Student rates: `records/exp12_students.json` `summary.cells[<corpus>|<teacher>|r<rank>].agentic` and `.refusal`; per-adapter values: `summary.adapters`; floors: `summary.floors`; contrasts: `summary.contrasts` (primary outcome `primary_shortcut`); projections: `summary.cells[...].projection` and `summary.projection_contrasts`; integrity: `summary.integrity`. Corpus statistics: `records/exp12_corpora.json` `summary.A`, `.B`, `.Bp`, `.C`. Experiment 1.1 reference rates: `records/exp11_main.json` and `records/exp11_main_lowprobe.json` `contrasts.P1.1`. Statements about how the corpora were built (the target assertion, the length cap, the token counter, the choice of prompts and samples) come from the docstrings and code of `src/v3/exp12.py` and `scripts/build_exp12_corpora.py`; the wording of the specification and the plan comes from `CLAUDE_RESEARCH_SPECIFICATION.md` (Phase 3, Experiment 1.2) and `PLAN.md` (rule 5). Every number in this section is listed in `reports/sections/c_exp12.claims.json` with the expression that reads it from these records, except the findings tagged **[audit-only]**, which come from `reports/audit_exp14_exp12.md` (sections cited next to each tag), are not in any record, and are written here without their figures.

## Not available in the records

Which of the four held-out templates carry the effect (per-template shortcut counts) is not stored in the final record; only the template-level p-values and intervals are. The interim analysis quotes rank-32 per-template counts from a scratch file, which is not cited here. The dollar cost and GPU time of Experiment 1.2 are not stored in either record (the spend ledger was not an allowed input for this section). The interim exploratory Spearman correlations between shortcut, compliance and projection are not in the final record and are not quoted. The following come from the independent audit only and are not in any record, so this section describes them in words and does not quote the audit's figures: the per-cell split of Corpus A's direct-tool targets; the number of Corpus B prompts per teacher that needed a sample other than the first one drawn; the average length of the teachers' thinking text; the number of rendered training prompts that contain the empty-think marker; the count and share of repeated 12-grams between test and training prompts; the adapter-level exact test; and the audit's own re-parse counts for the SFT targets. A stored pass of the Experiment 1.2 receipt check is also not in `records/receipt_verification.json` (see Sources).
