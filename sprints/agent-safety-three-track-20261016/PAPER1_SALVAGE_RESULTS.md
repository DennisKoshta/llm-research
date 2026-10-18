# Paper 1 Salvage Results (2026-10-17)

Code: `shared/modal_paper1_salvage.py` (Modal), `scripts/score_exp1_2_clean_eval.py`, `scripts/run_exp1_1_probe.py`.
Receipts: `records/exp1_2_clean_eval_receipt.json` (summary + every raw completion), `records/exp1_2_clean_eval_raw.json`,
`records/exp1_1_probe_salvage_results.md` (full probe tables). Activations: volume
`curiosity-study-artifacts:agent-safety-three-track-20261016/paper1_salvage/exp1_1_l48_acts_{intact,refusal_ablated}.npz`.

## Deviations from the request (read first)

1. **The Exp 1.2 adapters were never saved.** The original runner trains LoRA through forward hooks and deletes it after
   evaluation. With your sign-off, the **18 core adapters were retrained** (Corpus A r32 all_linear + Corpus B r8
   attn_only × {intact, refusal_ablated, pca_var_matched_seed101} × 3 seeds), using the original recipe, data and seeds.
   They are *not* bit-identical to the adapters behind the old receipt. Corpus B r32 and Corpus C were not retrained.
2. **Base model = BF16 `Qwen3.8-27B`, not the FP8 checkpoints.** Students are trained on BF16, so the untuned BF16 model
   is the matched floor. For Task 2, Refusal-Ablated = BF16 with r̂ projected out of every residual writer at α=1.0,
   which is the Phase 1 materialization recipe minus FP8 rounding. Verified: mean |h·r̂|/‖h‖ at layer 48 drops from
   0.061 (Intact) to 0.00025 (Ablated).
3. `data/exp1_1/scenarios_360.json` does not exist, so `factorial_360_trials.json` (30 scenarios × 12 cells) was used.
4. Your three "variance cells" are all `cot_off`, which has no `</think>`. The post-reasoning probe is therefore tested on
   `cot_on` cells only.

## Task 1 — clean Exp 1.2 student evaluation

Protocol: free `cot_on` generation, `max_new_tokens=2048`, no forced prefix, greedy, raw text saved. **0/1368
completions hit the cap (longest 865 tokens). Every `<think>` closed. Median length is 104–135 tokens for students and
296 for the base model.** 72 held-out trials per adapter
(24 high-cost harmful-implicit per seed).

**New failure mode exposed:** the original training targets cut the action part at 220 characters
(`format_agentic_target`: `a_part.strip()[:220]`). Students therefore learned to emit `<|im_end|>` **mid tool call**
(e.g. `{"name": "assign_validated_`). The strict JSON parser scores these as "no call". An "intended tool" score maps
the truncated name prefix to the unique tool it begins.

High-cost harmful-implicit (pooled over 3 seeds, 72 trials; base model = 24 trials):

| condition | strict shortcut (complete JSON) | intended-tool shortcut | intended safe | unidentifiable |
|---|---|---|---|---|
| **Untuned BF16 base (floor)** | 2/24 (8.3%) | 2/24 (8.3%) | 22 | 0 |
| Corpus A r32 — intact teacher | 0/72 (0.0%) | 0/72 | 48 | 24 |
| Corpus A r32 — **refusal_ablated** teacher | 15/72 (20.8%) | 15/72 | 42 | 15 |
| Corpus A r32 — PCA-101 teacher | 23/72 (31.9%) | 23/72 | 44 | 5 |
| Corpus B r8 — intact teacher | 3/72 (4.2%) | 3/72 | 22 | 47 |
| Corpus B r8 — **refusal_ablated** teacher | 8/72 (11.1%) | 9/72 (12.5%) | 63 | 0 |
| Corpus B r8 — PCA-101 teacher | 0/72 (0.0%) | 0/72 | 55 | 17 |

Exact McNemar, pairs = (seed, trial). Base-model comparisons reuse its single output per trial across 3 seeds
(pseudo-replicated, so read those p-values as optimistic):

| contrast | strict | p | intended | p |
|---|---|---|---|---|
| A r32: ablated vs intact | 20.8% vs 0.0% | **6.1e-5** | 20.8% vs 0.0% | **6.1e-5** |
| A r32: ablated vs PCA | 20.8% vs 31.9% | 0.22 | same | 0.22 |
| A r32: ablated vs base | 20.8% vs 8.3% | 0.049 | same | 0.049 |
| A r32: PCA vs base | 31.9% vs 8.3% | 4.9e-4 | same | 4.9e-4 |
| B r8: ablated vs intact | 11.1% vs 4.2% | 0.063 | 12.5% vs 4.2% | 0.031 |
| B r8: ablated vs PCA | 11.1% vs 0.0% | **0.0078** | 12.5% vs 0.0% | **0.0039** |
| B r8: ablated vs base | 11.1% vs 8.3% | 0.75 | 12.5% vs 8.3% | 0.55 |

**Assessment, Task 1:**
- **Corpus A (unfiltered teacher trajectories): transmission is real but not specific to refusal ablation.** Students of
  the ablated teacher take shortcuts more than intact-teacher students (20.8% vs 0%). But students of the
  *variance-matched PCA* teacher take them even more (31.9%). The old 63.89% vs 26.39% "C1" gap does not reproduce.
  Paper 1 can say "fine-tuning on a weight-edited teacher's unfiltered trajectories raises shortcut use". It cannot say
  "refusal ablation specifically transmits".
- **Corpus B r8 (filtered + paraphrased): weak at best.** Ablated-teacher students sit at 11–12.5%, which is
  indistinguishable from the untuned base floor (8.3%, p≈0.55–0.75). The "effect" is mostly intact- and PCA-teacher
  students dropping *below* the base rate. Do not claim subliminal transmission through filtered data from this.
- The old forced-prefix numbers (18.06% / 2.78%) should be retired.

## Task 2 — Exp 1.1 layer-48 probe (5-fold leave-two-templates-out, strict labels, truncated trials dropped)

Activations: final prompt token (`\n` after `<think>` for cot_on; `\n\n` after the empty think block for cot_off) and
the `</think>` token of saved non-truncated cot_on transcripts. Probe = StandardScaler + RidgeClassifierCV. Full tables
are in `records/exp1_1_probe_salvage_results.md`.

**Regimes 1–2 (pooled OOF AUROC):**

| subset | pos/neg | pre-reasoning probe | `</think>` probe | r̂ 1-D | prompt TF-IDF | design cell (cost×cons×rule×arm) |
|---|---|---|---|---|---|---|
| cot_on, both arms | 186/502 | 0.976 | 1.000 | 0.691 | 0.945 | **0.983** |
| cot_on, Intact | 65/278 | 0.970 | 1.000 | 0.958 | 0.931 | **0.978** |
| cot_on, Ablated | 121/224 | **0.986** | 1.000 | 0.477 | 0.978 | 0.979 |
| cot_off, both arms | 272/428 | 0.947 | — | 0.757 | 0.879 | **0.959** |
| cot_off, Intact | 98/257 | **0.969** | — | 0.883 | 0.883 | 0.965 |
| cot_off, Ablated | 174/171 | 0.920 | — | 0.426 | 0.907 | **0.936** |

**Regime 3a — within single arm×cell** (arm-level model, scores centred per test fold; design = 0.5 by construction):

| cell | pos/neg | pre probe | `</think>` probe | r̂ 1-D | TF-IDF |
|---|---|---|---|---|---|
| Intact cot_off `high_friction__harmful__implicit` | 8/22 | **0.847** | — | 0.744 | 0.557 |
| Ablated cot_off `high_friction__harmful__explicit` | 14/9 | 0.587 | — | 0.373 | **0.714** |
| Ablated cot_off `high_tedium__harmful__explicit` | 8/20 | 0.556 | — | 0.306 | 0.481 |
| Intact cot_on `low_cost__harmless__implicit` | 4/23 | 0.598 | 1.000 | 0.783 | 0.761 |
| Ablated cot_on `high_friction__harmless__explicit` | 3/19 | 0.754 | 1.000 | 0.421 | 0.947 |

Pooling all eight cot_off split cells and training only on them (3b): probe 0.753 vs design 0.722 vs TF-IDF 0.559 vs
r̂ 0.594 (pooled OOF, 120/101).

**Assessment, Task 2:**
- **Pre-reasoning probe ≈ design-factor lookup.** Across the grid it matches the cost × consequence × rule × arm table
  (0.92–0.99 vs 0.94–0.98) and never beats it by a meaningful margin, within arm or pooled. Exp 1.1 behaviour is
  almost fully determined by the cell, so a probe that encodes the scenario reaches ~0.97 for free.
- **The `</think>` probe is 1.000 everywhere, and that is trivial:** the reasoning states the decision before
  `</think>`. It is not monitoring evidence.
- **Within-cell (the only real test), the evidence is thin and mixed.** The one encouraging case is Intact cot_off
  `high_friction__harmful__implicit`: probe 0.85, r̂ projection 0.74, TF-IDF 0.56, on 8 vs 22 trials (no CI; at this n
  the 95% interval spans roughly ±0.17). In the Ablated cells the probe is near chance (0.56–0.59), and TF-IDF beats it
  in one. Where r̂ is intact, refusal-direction magnitude carries some trial-level signal; once r̂ is projected out,
  nothing in the pre-action state predicts which trials go wrong.
- **Paper-1-usable claim, at most:** "Within the Intact model, layer-48 refusal-direction projection weakly predicts
  shortcut-taking among otherwise identical high-friction harmful prompts (AUROC 0.74, n=30)." Treat it as exploratory.
  It is a single cell, and it was chosen after looking at the data.

## What worked / what didn't

- Worked: clean generation (no truncation, raw text saved), an untuned floor, and the ablation recipe verified in
  activations.
- Didn't: the Exp 1.2 headline claims (subliminal transmission specific to refusal ablation) do not survive a clean
  protocol with a PCA control and base floor. The pre-action probe adds nothing beyond the design table at grid level.
- New bug found: the Exp 1.2 training targets were truncated mid tool call (220-char cut), which degrades student output
  format. A proper rerun of Exp 1.2 should fix `format_agentic_target` before retraining.
