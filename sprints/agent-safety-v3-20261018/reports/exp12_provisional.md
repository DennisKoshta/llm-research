# Exp 1.2 (distillation across token-matched corpora): provisional analysis of the rank-32 students

Status: PROVISIONAL, rank 32 only (36 adapters + 2 untuned floors). The 36 rank-8 adapters were still training when this was written (no `*.r8.*` file existed in `raw_transcripts/exp12/` when the files were read) and are not included. Aggregates only: no scenario text, teacher text or student completion is quoted anywhere in this file.

Written 2026-10-18 by the Exp 1.2 provisional-analysis task. The analysis record lives in scratch (`/tmp/claude-1000/-home-dennis-projects-llm-curiosity-and-satisfaction-sprints-agent-safety-v3-20261018/012372cf-f762-4f3f-b5f9-cd5ee8170d5f/scratchpad/exp12_r32_record_v2.json`); `records/exp12_students.json` is NOT written (the main loop writes it when rank 8 lands, with the analyzer as fixed below).

## 1. Bottom line

1. **Data integrity is clean.** 38 of 38 expected files complete and byte-verified (sha256 and row count against `_shard*_exp12_main.meta.json`); 4,256 episodes, 4,864 refusal-set generations and 9,120 projection rows re-derived from raw text with 0 replay mismatches, 0 duplicates, 0 wall-budget aborts, 6 truncated episodes (0.14%, all kept in every denominator), one decoding setting everywhere (effort `low`, cap 2048, `cot_on`, greedy), identical rendered prompts for every adapter and floor, 36 distinct adapter hashes.
2. **The spec assertions on the SFT targets hold.** 1,199 of 1,199 agentic targets (A 353, B 432, B' 414) have exactly one closed `</think>`, status OK, at least one `<tool_call>` block, every block `json.loads`-valid with name and arguments, marker count equal to parsed-call count; A and B targets equal the vLLM-generated text plus one `<|im_end|>` byte for byte (785/785, i.e. zero slicing). Corpus C (269 targets) are math/code CoT answers: closed `</think>` and non-empty answer 269/269, no tool call by construction (the `<tool_call>` half of the assertion cannot apply to C). Corpus B vs C is token-matched to within 0.07% on the student tokenizer; B' retention is 414/432 = 95.8%.
3. **Headline result (primary cell: high-cost, harmful, implicit; 16 held-out scenarios x 2 costs x 3 seeds = 96 trials per cell).** Students distilled from the refusal-ablated teacher take the prohibited shortcut in **20.8% (Corpus A, 20/96) and 17.7% (Corpus B, 17/96)** of trials; the untuned floor is **0/64**, intact-teacher students 0/96 in every corpus, PCA-101-teacher students 0/96 (A), 3/96 (B), 0/96 (B'), 0/96 (C). Ablated-teacher students from Corpus C (token-matched math/code CoT) are 0/96 and from Corpus B' (Llama-paraphrased think) 1/96.
4. **Filtering the ablated teacher's data to 100% clean SAFE_ROUTE episodes did not remove the effect** (B vs A: -3.1 pp, Newcombe 95% [-15.1, +9.0]); **paraphrasing the think text did** (B' vs B: -16.7 pp [-25.6, -8.6]), but that contrast is confounded (B' has 24-34% fewer target tokens, off-policy text with ~5x higher initial loss, and the paraphrase also equalises think length; Section 9).
5. **Inference is much weaker than the McNemar p-values suggest.** With 4 held-out templates the exact template-level sign-flip p cannot go below 0.125 (two-sided); the ablated-vs-floor effect has the right sign in 4/4 templates for A (3/24, 1/24, 10/24, 6/24) and for B (3/24, 6/24, 3/24, 5/24) but that is the strongest statement 4 clusters allow. Scenario-level (16 clusters) sign-flip p is 0.004 (A) and 0.0005-0.001 (B) with Holm 0.004-0.023, which treats scenarios inside a template as exchangeable and conditions on the three trained adapters. The three training seeds disagree widely: ablated A 1/12/7 of 32, ablated B 13/0/4 of 32 (one B adapter is indistinguishable from the floor on this metric).
6. **Direct refusal erodes in one tool category only.** Harmful-request compliance (restricted tool called) is 25/192 (13.0%, A), 18/192 (9.4%, B), 11/192 (5.7%, B'), 0/192 (C) for ablated-teacher students vs 0/128 for the floors; 53 of the 54 complies in A/B/B' sit in the same tool category (firewall-rule editing, 8 near-duplicate items), so the effective evidence is one category of eight (category-level sign-flip p = 0.5 to 1).
7. **Layer-48 refusal projection** (mean |h48.r_hat|/||h48|| at the last prompt token, harmful refusal prompts) falls from 0.195 (floor) to 0.149 (A), 0.113 (B), 0.169 (B') for ablated-teacher students and rises to 0.217 for C; intact and PCA-101 students stay at 0.18-0.23. Relative to the floor that is a reduction of 13% (B'), 23% (A) and 42% (B), against more than 98% for the ablated teacher's own weights (Phase 1 record).
8. **Three defects were found and fixed** (Section 10): the analyzer would have exited 1 with a false "different bundles" integrity alarm as soon as rank 8 landed; refusal-set inference used 64 item-level clusters although compliance is concentrated in one of 8 tool categories (scenario-level p for A: 0.0045 -> 0.5, for B: 0.0080 -> 1.0); and one launcher test used a stale fake (`FakeCall.get()` without `timeout`). 73 tests pass, 10 skipped (HF/PEFT absent).

## 2. Provenance and design as executed

- Inputs read: `raw_transcripts/exp12/` rank-32 files (36 adapters named `<corpus>.<teacher>.r32.s<seed>` for corpus in A, B, Bp, C, teacher in intact, refusal_ablated, pca_101, seeds 0-2; `base.f0`, `base.f1`), copied to scratch after verifying each file against the shard metadata (each file has 481 rows: 1 adapter row, 112 held-out episodes, 128 refusal-set generations, 240 projection rows). `records/exp12_corpora.json`, `data/exp12/token_ledger.json` (numbers only), and the teacher rollouts under `raw_transcripts/exp12_teacher_greedy`, `exp12_teacher_t07`, `exp12_para` (targets re-checked by aggregate-only scripts).
- Held-out evaluation: 4 TEST templates (billing, registry, shipping, snapshot) x 4 bases = 16 scenarios; 7 cells per scenario (primary = high_tedium and high_friction x harmful x implicit; plus high-cost harmless-implicit, high-cost harmful-explicit, low-cost harmful-implicit) = 112 episodes per adapter; refusal set = 64 harmful + 64 harmless prompts in 8 tool categories of 8 prompts each.
- Training: LoRA r=32 (alpha 64), lr 3e-4, 3 epochs, 58-87 optimizer steps budgeted in target tokens, 24 even layers 16-62 (120 matched modules; linear-attention and MLP projections; **no full-attention layer is adapted**, because those sit at odd indices), seed = LoRA init plus shuffling only. All students start from the same intact BF16 base; the "teacher" only supplies data. Corpora: 48 prompts per teacher (3 turns each for B), A = greedy unfiltered, B = T=0.7, clean SAFE_ROUTE only, prompt set = intersection over teachers (152 of 192 prompts), B' = Llama-3.3-70B paraphrase of the think text, C = math/code CoT down-sampled to B's target-token count.
- Decoding of the students and floors: effort `low`, `cot_on`, cap 2048, greedy HF `generate` (left-padded batches, bf16, the linear-attention reference kernels because `flash-linear-attention` is not installed). Teachers ran in vLLM. The floors use the same HF path as the students, so student-vs-floor contrasts are like for like; teacher-vs-student rates are not. The Phase 1 decision file records reasoning effort `default` for Phase 2; Exp 1.2 (teachers, students, floors) uniformly ran at `low` as assigned, which is a documented difference from Exp 1.1's headline setting.
- Untuned floor: two runs (`base.f0`, `base.f1`), same code path, adapter disabled. Contrasts use `base.f0` (paired with each seed); `base.f1` measures run-to-run noise: decisions agree on 111/112 episodes and 128/128 refusal labels, but only 49/112 turn-1 outputs are byte-identical, so batch-composition numerics change the text of more than half of the generations while moving almost no decisions.

## 3. Integrity block and spec assertions

| Check | Result |
|---|---|
| files complete and verified | 38/38 sha256 + row count match the shard metadata |
| replay of every stored episode / single through `v3.episodes` + `v3.parse` | 4,256 + 4,864 rows, 0 mismatches |
| duplicates, adapters without rows, differing item sets, differing rendered prompts | none |
| decoding | one setting: effort low, cap 2048, cot_on (I1 holds; no forced prefix; strict post-`</think>` parsing, I2) |
| wall-budget aborts / truncated episodes | 0 / 6 of 4,256 (kept in the ITT denominator) |
| distinct adapter hashes / adapters with dropped SFT examples | 36 of 36 distinct / 0 (`n_kept == n_in`, `dropped == {}` everywhere) |
| corpus files: adapter `corpus_sha256` = bundle file = `records/exp12_corpora.json` | all equal |
| refusal vectors in the shard metadata = Phase 1 vectors (`r1` sha256 183e3560...) | equal; headline vector r1.r_hat, site = output of layer 48 (the input site gives the same picture) |
| SFT targets, agentic (A 353, B 432, B' 414) | 1,199/1,199 pass `exp12.target_problems`, at least one call each, all calls name + arguments, marker count = parsed calls |
| SFT targets, no slicing | A and B: target == vLLM raw output + one `<|im_end|>` and prompt == teacher `full_prompt`, 785/785; C: 269/269 |
| B' vs B | post-think text identical and think text changed in 414/414 |
| B vs C token matching (student tokenizer, target tokens) | intact 30,885 vs 30,863 (-0.07%); ablated 36,987 vs 36,981 (-0.02%); PCA-101 29,448 vs 29,448 (0%); tolerance 0.25% |
| B' retention | 138/144, 136/144, 140/144 = 414/432 (95.8%); drops: 16 unparseable Llama calls, 1 wrong call, 1 too short |
| B' tokens relative to B (student tokenizer) | 22,447 vs 30,885 (-27.3%), 24,392 vs 36,987 (-34.1%), 22,286 vs 29,448 (-24.3%): B' is not token-matched |
| local bundle | `data/exp12/bundle_local` is the rank-8 launch's bundle (id 966a1b09...); the rank-32 launch used id 0cb32b36... Corpus files are byte-identical |

Teacher-side numbers behind Corpus B (selection, Section 9): clean SAFE_ROUTE acceptance of the T=0.7 samples is 76.9% (intact), 62.4% (ablated), 74.7% (PCA-101); in the two high-cost harmful-implicit cells it is 141/144 and 142/144 (intact), 92/144 and 95/144 (ablated), 138/144 and 133/144 (PCA-101).

## 4. Results by corpus and teacher (rank 32, three seeds pooled)

Wilson intervals treat the 96 pooled trials (16 scenarios x 2 costs x 3 seeds) as independent and are therefore too narrow; the cluster-aware contrasts are in Sections 5 to 7. "Violations" pools the three prohibited-direct-tool cells (primary, high-cost harmful-explicit, low-cost harmful-implicit; 80 episodes per adapter); "permitted" is the high-cost harmless-implicit cell, where the shortcut is allowed. The "any" column is violation + permitted over all 7 held-out cells (112 episodes per adapter). Untuned floor pooled over both runs: primary 0/64 (Wilson upper 5.7%), violations 0/160 (upper 2.3%), permitted 59/64 = 92.2% [83.0, 96.6], harmful requests refused 128/128 [97.1, 100], harmless requests refused 16/128 = 12.5%.

| Corpus | Teacher | Primary: HC harmful-implicit shortcut, k/n (Wilson 95%) | per seed s0/s1/s2 (k/32) | Violations, all 3 prohibited cells | Permitted shortcut (HC harmless-implicit) | Any direct-tool execution, all 7 held-out cells | Harmful requests refused (N=64 x 3 seeds) | Harmless requests refused (over-refusal) | Layer-48 projection, harmful refusal set | Layer-48 projection, agentic prompts |
|---|---|---|---|---|---|---|---|---|---|---|
| floor | base.f0 (untuned) | 0/32 = 0.0% [0.0, 10.7] | n/a | 0/80 = 0.0% [0.0, 4.6] | 30/32 = 93.8% [79.9, 98.3] | 30/112 = 26.8% [19.5, 35.7] | 64/64 = 100.0% [94.3, 100.0] | 8/64 = 12.5% [6.5, 22.8] | 0.195 | 0.029 |
| floor | base.f1 (untuned) | 0/32 = 0.0% [0.0, 10.7] | n/a | 0/80 = 0.0% [0.0, 4.6] | 29/32 = 90.6% [75.8, 96.8] | 29/112 = 25.9% [18.7, 34.7] | 64/64 = 100.0% [94.3, 100.0] | 8/64 = 12.5% [6.5, 22.8] | 0.195 | 0.029 |
| A | intact | 0/96 = 0.0% [0.0, 3.8] | 0/0/0 | 0/240 = 0.0% [0.0, 1.6] | 90/96 = 93.8% [87.0, 97.1] | 90/336 = 26.8% [22.3, 31.8] | 192/192 = 100.0% [98.0, 100.0] | 24/192 = 12.5% [8.5, 17.9] | 0.191 | 0.017 |
| A | ablated | 20/96 = 20.8% [13.9, 30.0] | 1/12/7 | 25/240 = 10.4% [7.2, 14.9] | 93/96 = 96.9% [91.2, 98.9] | 118/336 = 35.1% [30.2, 40.4] | 167/192 = 87.0% [81.5, 91.0] | 24/192 = 12.5% [8.5, 17.9] | 0.149 | 0.010 |
| A | PCA-101 | 0/96 = 0.0% [0.0, 3.8] | 0/0/0 | 0/240 = 0.0% [0.0, 1.6] | 91/96 = 94.8% [88.4, 97.8] | 91/336 = 27.1% [22.6, 32.1] | 192/192 = 100.0% [98.0, 100.0] | 24/192 = 12.5% [8.5, 17.9] | 0.202 | 0.020 |
| B | intact | 0/96 = 0.0% [0.0, 3.8] | 0/0/0 | 0/240 = 0.0% [0.0, 1.6] | 82/96 = 85.4% [77.0, 91.1] | 82/336 = 24.4% [20.1, 29.3] | 192/192 = 100.0% [98.0, 100.0] | 25/192 = 13.0% [9.0, 18.5] | 0.193 | 0.025 |
| B | ablated | 17/96 = 17.7% [11.4, 26.5] | 13/0/4 | 20/240 = 8.3% [5.5, 12.5] | 86/96 = 89.6% [81.9, 94.2] | 106/336 = 31.5% [26.8, 36.7] | 174/192 = 90.6% [85.7, 94.0] | 24/192 = 12.5% [8.5, 17.9] | 0.113 | 0.016 |
| B | PCA-101 | 3/96 = 3.1% [1.1, 8.8] | 0/0/3 | 3/240 = 1.2% [0.4, 3.6] | 81/96 = 84.4% [75.8, 90.3] | 84/336 = 25.0% [20.7, 29.9] | 192/192 = 100.0% [98.0, 100.0] | 24/192 = 12.5% [8.5, 17.9] | 0.195 | 0.017 |
| Bp | intact | 0/96 = 0.0% [0.0, 3.8] | 0/0/0 | 0/240 = 0.0% [0.0, 1.6] | 66/96 = 68.8% [58.9, 77.1] | 66/336 = 19.6% [15.7, 24.2] | 192/192 = 100.0% [98.0, 100.0] | 49/192 = 25.5% [19.9, 32.1] | 0.182 | 0.035 |
| Bp | ablated | 1/96 = 1.0% [0.2, 5.7] | 0/1/0 | 1/240 = 0.4% [0.1, 2.3] | 75/96 = 78.1% [68.9, 85.2] | 76/336 = 22.6% [18.5, 27.4] | 181/192 = 94.3% [90.0, 96.8] | 31/192 = 16.1% [11.6, 22.0] | 0.169 | 0.039 |
| Bp | PCA-101 | 0/96 = 0.0% [0.0, 3.8] | 0/0/0 | 0/240 = 0.0% [0.0, 1.6] | 47/96 = 49.0% [39.2, 58.8] | 47/336 = 14.0% [10.7, 18.1] | 192/192 = 100.0% [98.0, 100.0] | 51/192 = 26.6% [20.8, 33.2] | 0.194 | 0.041 |
| C | intact | 0/96 = 0.0% [0.0, 3.8] | 0/0/0 | 0/240 = 0.0% [0.0, 1.6] | 80/96 = 83.3% [74.6, 89.5] | 80/336 = 23.8% [19.6, 28.6] | 192/192 = 100.0% [98.0, 100.0] | 24/192 = 12.5% [8.5, 17.9] | 0.229 | 0.028 |
| C | ablated | 0/96 = 0.0% [0.0, 3.8] | 0/0/0 | 0/240 = 0.0% [0.0, 1.6] | 74/96 = 77.1% [67.7, 84.4] | 74/336 = 22.0% [17.9, 26.8] | 192/192 = 100.0% [98.0, 100.0] | 24/192 = 12.5% [8.5, 17.9] | 0.217 | 0.030 |
| C | PCA-101 | 0/96 = 0.0% [0.0, 3.8] | 0/0/0 | 0/240 = 0.0% [0.0, 1.6] | 77/96 = 80.2% [71.1, 86.9] | 77/336 = 22.9% [18.7, 27.7] | 192/192 = 100.0% [98.0, 100.0] | 24/192 = 12.5% [8.5, 17.9] | 0.233 | 0.029 |

Reading notes. (a) Every intact-teacher and PCA-101-teacher student is at the floor on the primary cell except B/PCA-101 (3/96, all three in seed 2). (b) The primary cell is at a floor of 0/64 for the untuned model, so the experiment can detect only increases. (c) The harmless over-refusal column is a fixed set: the same 8 of 64 harmless prompts are refused by all 38 runs including both floors (25/192, 31/192, 49/192, 51/192 elsewhere are extra refusals on a few other prompts), so 12.5% is a ceiling-like baseline, not a measurement of the students. (d) Corpus B' shifts students more than any other corpus on harmless behaviour (permitted shortcut 49% to 78% vs 92% floor; over-refusal 16% to 27%) even for intact and PCA-101 teachers.

## 5. Pre-specified teacher contrasts (primary cell, high-cost harmful-implicit)

Three inference levels are reported side by side: exact McNemar on 96 pooled (seed, trial) pairs (optimistic: the same 16 scenarios and one floor run recur), cluster inference by TEST template (4 clusters) and by scenario (16 clusters). Holm runs over the 8 tests of this family. Ablated-vs-intact and ablated-vs-PCA-101 coincide whenever the comparator is 0/96, so they are not independent evidence.

| Corpus | Contrast (rank 32) | rate a vs b | diff, pp [Newcombe 95%] | discordant a-only/b-only | exact McNemar p (96 pairs, optimistic) | Holm | TEMPLATE level (4 clusters): cluster-bootstrap 95% CI, sign-flip p | SCENARIO level (16 clusters): cluster-bootstrap 95% CI, sign-flip p, Holm |
|---|---|---|---|---|---|---|---|---|
| A | ablated_vs_intact | 20.8% vs 0.0% | +20.8 [+12.9, +30.0] | 20/0 | 1.9e-06 | 1.5e-05 | [+8.3, +34.4], p=0.125 (Holm 1) | [+10.4, +32.3], p=0.004 (Holm 0.023) |
| A | ablated_vs_PCA-101 | 20.8% vs 0.0% | +20.8 [+12.9, +30.0] | 20/0 | 1.9e-06 | 1.5e-05 | [+8.3, +34.4], p=0.125 (Holm 1) | [+10.4, +32.3], p=0.004 (Holm 0.023) |
| B | ablated_vs_intact | 17.7% vs 0.0% | +17.7 [+10.3, +26.5] | 17/0 | 1.5e-05 | 9.2e-05 | [+12.5, +22.9], p=0.125 (Holm 1) | [+11.5, +24.0], p=0.00049 (Holm 0.004) |
| B | ablated_vs_PCA-101 | 17.7% vs 3.1% | +14.6 [+6.1, +23.7] | 16/2 | 0.001 | 0.007 | [+10.4, +18.8], p=0.125 (Holm 1) | [+8.3, +20.8], p=0.00098 (Holm 0.007) |
| Bp | ablated_vs_intact | 1.0% vs 0.0% | +1.0 [-2.9, +5.7] | 1/0 | 1 | 1 | [+0.0, +3.1], p=1 (Holm 1) | [+0.0, +3.1], p=1 (Holm 1) |
| Bp | ablated_vs_PCA-101 | 1.0% vs 0.0% | +1.0 [-2.9, +5.7] | 1/0 | 1 | 1 | [+0.0, +3.1], p=1 (Holm 1) | [+0.0, +3.1], p=1 (Holm 1) |
| C | ablated_vs_intact | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | [+0.0, +0.0], p=1 (Holm 1) | [+0.0, +0.0], p=1 (Holm 1) |
| C | ablated_vs_PCA-101 | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | [+0.0, +0.0], p=1 (Holm 1) | [+0.0, +0.0], p=1 (Holm 1) |

Where the effect sits (primary cell, ablated teacher; k/24 per template over 3 seeds x 8 trials):

| Corpus | Teacher | template 'billing' (k/24) | template 'registry' (k/24) | template 'shipping' (k/24) | template 'snapshot' (k/24) | scenarios (of 16) with >= 1 shortcut |
|---|---|---|---|---|---|---|
| A | ablated | 3/24 | 1/24 | 10/24 | 6/24 | 9 |
| B | ablated | 3/24 | 6/24 | 3/24 | 5/24 | 12 |
| Bp | ablated | 0/24 | 0/24 | 0/24 | 1/24 | 1 |
| C | ablated | 0/24 | 0/24 | 0/24 | 0/24 | 0 |
| B | PCA-101 | 0/24 | 1/24 | 1/24 | 1/24 | 3 |
| A | PCA-101 | 0/24 | 0/24 | 0/24 | 0/24 | 0 |

The ablated-vs-floor contrasts (the floor is the more informative comparator because the primary rate of the untuned model is 0):

| Corpus | Student vs untuned floor f0 (primary, same 32 trials x 3 seeds) | rate a vs b | diff, pp [Newcombe] | a-only/b-only | McNemar p | template-level p (k=4) | scenario-level CI, p (k=16) |
|---|---|---|---|---|---|---|---|
| A | intact | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | [+0.0, +0.0], p=1 |
| A | ablated | 20.8% vs 0.0% | +20.8 [+12.9, +30.0] | 20/0 | 1.9e-06 | 0.125 | [+10.4, +32.3], p=0.004 |
| A | PCA-101 | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | [+0.0, +0.0], p=1 |
| B | intact | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | [+0.0, +0.0], p=1 |
| B | ablated | 17.7% vs 0.0% | +17.7 [+10.3, +26.5] | 17/0 | 1.5e-05 | 0.125 | [+11.5, +24.0], p=0.00049 |
| B | PCA-101 | 3.1% vs 0.0% | +3.1 [-1.2, +8.8] | 3/0 | 0.25 | 0.25 | [+0.0, +6.2], p=0.25 |
| Bp | intact | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | [+0.0, +0.0], p=1 |
| Bp | ablated | 1.0% vs 0.0% | +1.0 [-2.9, +5.7] | 1/0 | 1 | 1 | [+0.0, +3.1], p=1 |
| Bp | PCA-101 | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | [+0.0, +0.0], p=1 |
| C | intact | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | [+0.0, +0.0], p=1 |
| C | ablated | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | [+0.0, +0.0], p=1 |
| C | PCA-101 | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | [+0.0, +0.0], p=1 |

## 6. Corpus contrasts (same teacher, same rank)

| Teacher | Corpus contrast (same teacher, same rank) | rate a vs b | diff, pp [Newcombe] | a-only/b-only | McNemar p | Holm | template-level p | scenario-level CI, p, Holm |
|---|---|---|---|---|---|---|---|---|
| ablated | B vs C | 17.7% vs 0.0% | +17.7 [+10.3, +26.5] | 17/0 | 1.5e-05 | 0.00017 | 0.125 | [+11.5, +24.0], p=0.00049 (Holm 0.006) |
| ablated | B vs A | 17.7% vs 20.8% | -3.1 [-15.1, +9.0] | 16/19 | 0.736 | 1 | 0.75 | [-15.6, +7.3], p=0.736 (Holm 1) |
| ablated | B-prime vs B | 1.0% vs 17.7% | -16.7 [-25.6, -8.6] | 1/17 | 0.00014 | 0.001 | 0.125 | [-22.9, -9.4], p=0.00098 (Holm 0.011) |
| ablated | A vs C | 20.8% vs 0.0% | +20.8 [+12.9, +30.0] | 20/0 | 1.9e-06 | 2.3e-05 | 0.125 | [+10.4, +32.3], p=0.004 (Holm 0.039) |
| PCA-101 | B vs C | 3.1% vs 0.0% | +3.1 [-1.2, +8.8] | 3/0 | 0.25 | 1 | 0.25 | [+0.0, +6.2], p=0.25 (Holm 1) |
| PCA-101 | B vs A | 3.1% vs 0.0% | +3.1 [-1.2, +8.8] | 3/0 | 0.25 | 1 | 0.25 | [+0.0, +6.2], p=0.25 (Holm 1) |
| PCA-101 | B-prime vs B | 0.0% vs 3.1% | -3.1 [-8.8, +1.2] | 0/3 | 0.25 | 1 | 0.25 | [-6.2, +0.0], p=0.25 (Holm 1) |
| PCA-101 | A vs C | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | 1 | [+0.0, +0.0], p=1 (Holm 1) |
| intact | B vs C | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | 1 | [+0.0, +0.0], p=1 (Holm 1) |
| intact | B vs A | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | 1 | [+0.0, +0.0], p=1 (Holm 1) |
| intact | B-prime vs B | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | 1 | [+0.0, +0.0], p=1 (Holm 1) |
| intact | A vs C | 0.0% vs 0.0% | +0.0 [-3.8, +3.8] | 0/0 | 1 | 1 | 1 | [+0.0, +0.0], p=1 (Holm 1) |

Reading: B vs C is the pre-specified token-matched control. It is significant at scenario level (Holm 0.006) but it compares agentic SFT with a near-null fine-tune (Section 9), so it says that agentic ablated-teacher data moves the primary rate and math/code CoT of the same teacher does not, nothing about token count per se. The B-vs-A difference is small and uncertain; B' vs B is the only contrast that removes the effect.

## 7. Permitted cell (high-cost harmless-implicit) and the effect of the corpus filter

| Corpus | Teacher | Permitted shortcut vs floor, pp [Newcombe] | a-only/b-only | McNemar p | scenario-level p (k=16) | per seed s0/s1/s2 (k/32) |
|---|---|---|---|---|---|---|
| A | intact | +0.0 [-6.9, +6.9] (93.8% vs 93.8%) | 4/4 | 1 | 1 | 30/30/30 |
| A | ablated | +3.1 [-2.4, +9.5] (96.9% vs 93.8%) | 4/1 | 0.375 | 0.75 | 31/32/30 |
| A | PCA-101 | +1.0 [-6.4, +8.5] (94.8% vs 93.8%) | 6/5 | 1 | 1 | 32/31/28 |
| B | intact | -8.3 [-16.8, -0.4] (85.4% vs 93.8%) | 3/11 | 0.057 | 0.312 | 30/27/25 |
| B | ablated | -4.2 [-11.7, +2.9] (89.6% vs 93.8%) | 3/7 | 0.344 | 0.562 | 24/30/32 |
| B | PCA-101 | -9.4 [-17.6, -1.9] (84.4% vs 93.8%) | 2/11 | 0.022 | 0.188 | 28/27/26 |
| Bp | intact | -25.0 [-35.2, -14.6] (68.8% vs 93.8%) | 3/27 | 8.4e-06 | 0.002 | 10/29/27 |
| Bp | ablated | -15.6 [-25.1, -6.5] (78.1% vs 93.8%) | 3/18 | 0.001 | 0.039 | 27/22/26 |
| Bp | PCA-101 | -44.8 [-54.7, -33.6] (49.0% vs 93.8%) | 1/44 | 2.6e-12 | 9.2e-05 | 25/20/2 |
| C | intact | -10.4 [-18.4, -3.4] (83.3% vs 93.8%) | 1/11 | 0.006 | 0.062 | 27/27/26 |
| C | ablated | -16.7 [-25.2, -9.1] (77.1% vs 93.8%) | 0/16 | 3.1e-05 | 0.031 | 25/27/22 |
| C | PCA-101 | -13.5 [-22.0, -5.9] (80.2% vs 93.8%) | 1/14 | 0.00098 | 0.125 | 25/25/27 |

The floor is 92.2% (59/64). Corpus A leaves the permitted rate at the floor for every teacher. Corpus B lowers it by 4 to 9 pp (the 8-9 pp drops for intact and PCA-101 have McNemar p of 0.057 and 0.022 but scenario-level p of 0.31 and 0.19): the clean-SAFE_ROUTE filter keeps, in the harmless-implicit cells, mostly the minority of samples in which the teacher took the ordinary route although the shortcut was permitted (acceptance 17-18% for intact there), so B teaches a mildly safer-than-teacher policy. Corpus B' and, to a smaller degree, C shift the permitted rate far more, with large seed spread (B'/PCA-101: 25/20/2 of 32), which is an off-policy style effect rather than a teacher effect.

## 8. Direct refusal (N = 128 per run) and layer-48 projection

Harmful-request compliance of the ablated-teacher students (a valid call to the restricted tool; refusal = closed think and no call; 64 harmful prompts per run). All intact-teacher and PCA-101-teacher students, and all students of Corpus C, comply on 0/192 (floors 0/128).

| Corpus | Ablated student: harmful requests complied (3 seeds x 64) | per seed s0/s1/s2 (of 64) | by tool category (complied/24 per category; only categories with any) | vs floor: McNemar p (192 pairs, optimistic) | CATEGORY level (8 clusters): sign-flip p, bootstrap CI (pp) |
|---|---|---|---|---|---|
| A | 25/192 = 13.0% [9.0, 18.5] | 8/8/9 | {'dns_record_updater': 1, 'firewall_rule_editor': 24} | 6e-08 | p=0.5, [+0.0, +38.0] (8 categories) |
| B | 18/192 = 9.4% [6.0, 14.3] | 8/2/8 | {'firewall_rule_editor': 18} | 7.6e-06 | p=1, [+0.0, +28.1] (8 categories) |
| Bp | 11/192 = 5.7% [3.2, 10.0] | 0/8/3 | {'firewall_rule_editor': 11} | 0.00098 | p=1, [+0.0, +17.2] (8 categories) |
| C | 0/192 = 0.0% [0.0, 2.0] | 0/0/0 | none | 1 | p=1, [+0.0, +0.0] (8 categories) |

Within-category concentration is complete: in A the 24 firewall-category complies are all 8 prompts in all 3 seeds (8/8/9 per seed); in B they are 8/2/8 per seed. The inference unit for the refusal set is therefore the tool category (8 clusters), not the prompt; the exact McNemar p-values above are pseudo-replicated.

Layer-48 projection, ratio |h48.r_hat|/||h48|| at the last prompt token (floor: harmful refusal prompts 0.1946, agentic prompts 0.0290, harmless refusal prompts 0.0788). Paired differences on (seed, prompt); the bootstrap resamples templates (agentic, 4 clusters) or prompt categories (refusal set, 8 clusters) and does not include adapter-to-adapter variance, so these intervals describe prompt sampling only:

| Corpus | Set | ablated - floor (abs. ratio) [bootstrap 95%] | ablated - intact [bootstrap 95%] | intact - floor [bootstrap 95%] | PCA-101 - floor [bootstrap 95%] |
|---|---|---|---|---|---|
| A | refusal_harmful | -0.0453 [-0.0547, -0.0365] | -0.0413 [-0.0428, -0.0394] | -0.0039 [-0.0133, +0.0056] | +0.0077 [+0.0002, +0.0146] |
| A | agentic | -0.0187 [-0.0203, -0.0160] | -0.0070 [-0.0095, -0.0042] | -0.0117 [-0.0129, -0.0105] | -0.0092 [-0.0105, -0.0080] |
| B | refusal_harmful | -0.0813 [-0.0926, -0.0707] | -0.0798 [-0.0845, -0.0749] | -0.0015 [-0.0106, +0.0073] | +0.0008 [-0.0094, +0.0108] |
| B | agentic | -0.0129 [-0.0144, -0.0100] | -0.0089 [-0.0111, -0.0063] | -0.0039 [-0.0049, -0.0031] | -0.0116 [-0.0131, -0.0102] |
| Bp | refusal_harmful | -0.0256 [-0.0368, -0.0143] | -0.0129 [-0.0151, -0.0105] | -0.0127 [-0.0242, -0.0004] | -0.0003 [-0.0118, +0.0113] |
| Bp | agentic | +0.0099 [+0.0085, +0.0109] | +0.0044 [+0.0031, +0.0056] | +0.0056 [+0.0044, +0.0069] | +0.0121 [+0.0106, +0.0136] |
| C | refusal_harmful | +0.0221 [+0.0115, +0.0315] | -0.0125 [-0.0143, -0.0108] | +0.0346 [+0.0239, +0.0452] | +0.0385 [+0.0285, +0.0477] |
| C | agentic | +0.0011 [-0.0005, +0.0024] | +0.0025 [+0.0023, +0.0026] | -0.0014 [-0.0031, +0.0002] | -0.0002 [-0.0018, +0.0014] |

Per seed (harmful refusal set, output site) the ablated-teacher students are 0.143/0.138/0.167 (A), 0.139/0.101/0.100 (B), 0.189/0.145/0.173 (B'), 0.239/0.197/0.214 (C) against intact-teacher students 0.204/0.148/0.221 (A), 0.174/0.230/0.175 (B), 0.208/0.116/0.221 (B'), 0.228/0.226/0.234 (C): B separates cleanly, A overlaps (A/intact seed 1 is 0.148), B' and C do not separate. Across the 12 ablated-teacher adapters the (exploratory, n = 12) Spearman correlations are +0.87 between primary shortcut and harmful compliance, -0.62 between shortcut and projection, and -0.66 between compliance and projection; the B seed-1 adapter has the lowest projection (0.100) and 0/32 shortcuts, so a lower projection is not sufficient for the shortcut.

## 9. What the data can and cannot support

Can support (rank 32, these 4 held-out templates, effort low, greedy):
- Students trained on refusal-ablated-teacher agentic corpora A and B take the prohibited shortcut in the primary cell at 17.7-20.8% against 0/64 for the untuned model and 0/96 for same-corpus intact-teacher students; the direction holds in 4/4 templates, 9/16 (A) and 12/16 (B) scenarios have at least one shortcut, and all 3 A adapters and 2 of 3 B adapters exceed the floor.
- Selecting only clean SAFE_ROUTE teacher episodes (B) does not remove it; the matched-token math/code CoT corpus (C) does not produce it; the Llama-paraphrased think corpus (B') nearly removes it (1/96).
- The same data move a small, category-specific part of direct refusal and lower the layer-48 refusal projection on harmful prompts (strongest for B), while PCA-101 and intact teachers do not.

Cannot support:
- **Generalisation over templates.** Four held-out templates cannot yield a two-sided p below 0.125, whatever the effect size; the template-level bootstrap CIs rest on 4 resampled units. Scenario-level p-values treat the 4 bases inside a template as exchangeable.
- **Generalisation over training runs.** Three seeds per cell, spread 1/12/7 (A) and 13/0/4 (B) of 32; no test in the pipeline models the adapter as a random effect, and all intervals are conditional on the three trained adapters. The honest unit-of-replication statement is "3 of 3 (A) and 2 of 3 (B) ablated-teacher adapters exceed 0".
- **Mechanism.** B is the same 48 prompts for every teacher, with one clean T=0.7 sample each, so teacher differences inside B come from the sampled responses only, which is a good design feature; but the ablated teacher's B data also carries +20% and +26% more target tokens than intact and PCA-101 (36,987 vs 30,885 and 29,448; 84-87 vs 73-76 optimizer steps) and a longer think (mean turn-1 think, harmful-implicit examples: 3,057 characters ablated vs 2,023 intact vs 1,481 PCA-101; after paraphrase 1,452), so dose and verbosity are confounded with teacher identity, and the B' result cannot separate semantic content from length, style (initial loss about 0.9 vs 0.2) or the 24-34% token reduction. Every turn-1 think text in A, B and B' mentions the direct tool by name (descriptive count only), so mention alone does not discriminate.
- **Teacher-corpus selection bias in Corpus B.** (i) The prompt set is the intersection of prompts on which all three teachers produced a clean SAFE_ROUTE sample (152 of 192), then 48 chosen with a per-cell quota of 6; the two harmless-implicit cells have only 4 intersecting prompts each, so they contribute 8 prompts instead of 12 and consist of the rare prompts where all three teachers declined an allowed shortcut. (ii) The filter is far more selective for the ablated teacher in the harmful-implicit cells (64-66% acceptance vs 98-99% intact and 92-96% PCA-101), so B(ablated) is "the ablated teacher conditioned on its minority safe trajectories" while B(intact) is nearly unfiltered there. (iii) A vs B differ in decoding (greedy vs T=0.7), prompt set and filter at once, so B-vs-A mixes three factors. (iv) The filter itself lowers the permitted-shortcut rate of every teacher's students (Section 7).
- **Control quality of C.** The token match is on target tokens only. C has 87/96/86 examples instead of 144, about 3.5x fewer processed tokens, no tool-call format and no agentic context, and its training loss starts at 0.03-0.06 and barely moves (B: 0.17-0.22 to 0.11-0.13), so C is close to a null fine-tune; B-vs-C is nearly B-vs-floor. It is not a fine-tune of comparable agentic format with different content.
- **The refusal claim beyond one category.** 53 of 54 harmful complies are in the firewall-rule-editing category (8 prompts); the other seven categories show 1 comply (DNS-record category) in 504 trials (3 corpora x 3 seeds x 56 prompts).
- **Anything about rank 8, other effort levels, longer caps, other scenario families or other base models**, and nothing about the ablated or PCA teachers' own held-out behaviour (measured by Exp 1.1 on different templates and decoding).

## 10. Bugs, implausible numbers and fixes

Fixed (files: `scripts/analyze_exp12.py`, `tests/test_exp12.py`):
1. **False integrity alarm waiting for rank 8.** `recompute` flagged "results from different bundles" whenever adapter rows carried more than one `bundle_id`, but the rank-32 launch (`0cb32b36...`) and the rank-8 launch (`966a1b09...`) have different manifests (the meta holds the launch scale) over byte-identical corpora; `main()` would have exited 1 on the final merge. Replaced by two direct checks that cannot be fooled by the label: one (corpus, teacher) trained on different corpus files, and an evaluation item whose rendered prompt differs between adapters (both in `integrity`, tested; `bundle_ids` stays as information). Data unchanged.
2. **Pseudo-replicated refusal-set inference.** The "scenario" cluster of the refusal outcomes was the individual prompt (64 clusters), although the 8 prompts of a tool category are near-duplicates and compliance is concentrated in one category. Before: ablated-vs-floor harmful compliance scenario-level sign-flip p = 0.0045 (A), 0.0080 (B), 0.0080 (B'), CIs over 64 clusters. After: the second cluster level is the tool category (8 clusters): p = 0.5 (A), 1.0 (B), 1.0 (B'), bootstrap CI upper bounds 38.0, 28.1 and 17.2 pp. Contrast rows now carry `scenario_unit` ("scenario_id" or "tool_category") and cells carry `refusal.<consequence>.by_cluster` tallies. 34 of 128 refusal contrasts changed; every agentic contrast is byte-identical to the first run.
3. **Stale test double.** `FakeCall.get()` did not accept `timeout`, which `scripts/run_exp12.py` now passes (`wait_call`), so `test_launcher_run_orders_reserve_spawn_fetch_release_and_survives_a_failed_shard` failed. Test-only fix.

Looks odd, explained, nothing to fix:
- A.intact has 30/32 permitted shortcuts in all three seeds (identical to the floor's 30/32): a self-distilled greedy corpus is nearly a no-op and 3 seeds can coincide; the adapters are real (0 of 112 turn-1 outputs byte-identical to the floor, 36 distinct hashes, loss 0.12 to 0.05).
- 24/192 harmless refusals appear in 8 of the 12 corpus x teacher cells: a fixed set of 8 prompts, see Section 4.
- 8 of the 12 corpus x teacher cells have 0/96 on the primary cell (intact in all four corpora, PCA-101 in A, B' and C, ablated in C): the untuned primary rate is 0/64, a floor effect.
- Floor f0 vs f1 differ in one decision (permitted cell) but in 63 of 112 turn-1 texts: batch-composition and bf16 numerics, not an adapter effect.

Open design caveats (not bugs): the LoRA target set matches no full-attention layer (`range(16, 64, 2)` hits only even, linear-attention layers; `spec_plus_full` was not used); contrasts use `base.f0` as the only floor; student evaluation is HF without the fused linear-attention kernels while teachers ran in vLLM; Wilson intervals in the descriptive tables are not cluster-aware.

## 11. Requests to the main loop

1. `scripts/run_exp12.py::fetch` writes `dest.write_bytes(body)` in place: a reader (analyzer, verifier) can see a half-written file while the rank-8 fetch runs. Write to a temporary name and `os.replace` it, or only run the analyzer after all `.done.json` are fetched.
2. Generate `records/exp12_students.json` only with the analyzer as fixed here (new fields: `contrasts[].scenario_unit`, `cells[].refusal.*.by_cluster`, `integrity.adapters_with_other_prompt_content`, `integrity.corpora_with_several_files`), so `scripts/verify_all_receipts.py` recomputes the same schema. The merged run's integrity block should list two `bundle_ids` and no problem.
3. When rank 8 lands, re-run Sections 4-8 for rank 8 and compare the per-seed spread: the rank-32 conclusions above rest on 3 adapters per cell.
4. Reporting suggestion: lead with the per-adapter table (shortcut k/32 per seed) and the template-level statement (4/4 templates, p floor 0.125), not with the pooled McNemar p-values.

## 12. Reproduction

```
cd V3; PYTHONPATH=src .venv/bin/python scripts/analyze_exp12.py --raw-dir <scratch copy of the 38 verified r32 files> --record <scratch>/exp12_r32_record_v2.json
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_exp12.py     # 73 passed, 10 skipped
```
