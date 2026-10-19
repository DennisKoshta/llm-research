# Independent audit of Exp 1.4 (blocked ordinary route) and Exp 1.2 (distillation students), recomputed from raw episodes

Task B of the v3 audit round. Written 2026-10-19. Aggregates only: no scenario text, teacher text, student completion or SFT target is quoted anywhere in this file. All harmful-cell text was read by code only (`json.loads`, structural fields, counts).

## 0. Scope, method, independence

* **Inputs read.** `raw_transcripts/exp14`, `exp14_ctrl`, `exp14_low` (8 files x 320 episodes), `raw_transcripts/exp11` and `exp11_effort_low_a` (twin and reference rows), `raw_transcripts/exp12` (74 files x 481 rows), `exp12_teacher_greedy`, `exp12_teacher_t07`, `exp12_para`, `data/exp12/corpora/*.jsonl` (12 SFT corpora), `data/exp12/{eval_instances,requests_*,token_ledger,corpora_manifest}`, `data/exp14/{manifest,requests_*}`, `records/exp14_blocked_route.json`, `records/exp12_students.json`, `records/exp12_corpora.json`, `reports/exp14_design.md`, `PLAN.md`.
* **Versions audited (sha256, first 16 hex).** `records/exp14_blocked_route.json` 3ed162243a2c1bde (mtime 2026-10-19 00:39), `records/exp12_students.json` acdecb5489b28fa8 (00:26), `records/exp12_corpora.json` 3482f09f84013f16 (2026-10-18 18:14), `reports/exp14_design.md` afc274673810953b (equal to the hash frozen in `data/exp14/manifest.json`, so the design file has not changed since the build).
* **My own code (scratchpad `.../scratchpad/auditB/`, see section 5).** A sandbox state machine written from the design document (ordinary route, lock error, terminal direct / escalate, multi-call stop, truncation, TEXT_NO_CALL, MAX_TURNS), the Exp 1.4 outcome classifier, Wilson, Newcombe-10 paired CI, exact McNemar (`scipy.stats.binomtest`), exact sign-flip enumeration, Holm, and every table. The only repository code used is `v3.parse.parse_turn` (strict post-`</think>` parsing). `v3.exp14`, `v3.exp12`, `scripts/analyze_exp14.py`, `scripts/analyze_exp12.py`, `v3.sandbox` and `v3.episodes` were not called to produce any number; `v3.stats.exact_mcnemar` and `paired_diff_newcombe_ci` were used only as a cross-check of C1 (identical to 1e-9).
* **Replay has power.** Negative controls on the 2,560 Exp 1.4 episodes: replaying with timings swapped gives 959 `n_blocked_safe_calls` mismatches and 1,875 observation mismatches; with the block shifted by one step, or removed, 1,859 mismatches each; the correct block gives 0. A planted error in the table comparison (one outcome flipped, one blocked count changed) produced 76 flagged differences.
* **No record written, no receipt wiring.** This task writes only this report, so no `v_<name>` entry was added to `scripts/verify_all_receipts.py` and the verifier was not run (it rewrites `records/receipt_verification.json`, outside the write scope). Every number below is recomputable by the scratch scripts listed in section 5 from `raw_transcripts` and `data/exp12`.

## 1. Bottom line

1. **Every number in `records/exp14_blocked_route.json` and `records/exp12_students.json` that I recomputed reproduces exactly from the raw rows.** Exp 1.4: 2,560 of 2,560 episodes replayed, 0 mismatches in route, direct_taken, sub_status, n_blocked_safe_calls, n_turns, outcome_category, undeclared_direct_attempt, block spec, and 0 mismatches in the stored sandbox observations; 168 of 168 table blocks equal; all 32 confirmatory contrasts (C1-C5: 24 at default effort, 8 at low) equal. Exp 1.2: 8,288 of 8,288 student and floor episodes replayed, 0 mismatches; 72 adapters x 4 cells x 6 metrics, 96 pooled cells, 2 floors, 384 agentic contrasts, the 16 primary Holm adjustments and 16 template/scenario sign-flip p-values equal.
2. **Nothing is wrong with the data pipeline.** No block wording in any prompt (0 of 2,560), twin prompts byte-identical (2,560 of 2,560), no leakage of train bases into the held-out set by scenario, template, author index, tool name or exact message, 0 wall-budget aborts, 0 duplicate trial ids, every A/B/B' SFT target parses to OK with one closed `</think>` and at least one valid call (1,199 of 1,199), 72 distinct adapters, none a no-op.
3. **The substantive issues are interpretive, not computational** (details in section 4):
   * Exp 1.4: **no episode persisted** (0 STUCK_RETRYING of 2,560; 1,798 of 1,804 escalations come after exactly one blocked call). The "gave up vs persisted" split is degenerate, and blocking did **not** make models take the shortcut in the harmful cells: every harmful-cell shortcut was taken before the block was met, with two exceptions of one episode each (default harmful+implicit, low harmful+explicit). The only block-induced shortcuts of any size sit in the harmless+explicit cell.
   * Exp 1.4 C1: the default-effort primary contrast is 4/80 vs 0/80 (exact McNemar p = 0.125, Holm 0.625); the strong result (28/80 vs 0/80, p = 7.5e-9) is the low-effort sensitivity block, and its 28 discordant pairs are **14 scenarios counted twice** (early and late timing share the prompt and the pre-block trajectory), so the scenario-level exact p is 1.2e-4, not 7.5e-9.
   * Exp 1.2: Corpus A for the ablated teacher contains **10 SFT targets that call the prohibited direct tool** (4 of them in harmful+implicit cells); Corpus B and B' contain none. The A contrast is therefore partly direct imitation; B is the clean test of transfer through safe-only demonstrations.

## 2. Exp 1.4 (blocked ordinary route), exploratory

### 2.1 Replay census (item 1)

Census of all 2,560 episodes: 6 arms x 320 at default effort (intact, refusal_ablated in `exp14`; pca_101, pca_102, pca_103, unrelated in `exp14_ctrl`) and 2 arms x 320 at low effort (`exp14_low`). 9,148 turns, 1,865 lock observations in 1,859 episodes.

State machine (own): safe-tool call with arguments matching the next expected step (type `int`, operation equal) is OK while the expected step is <= after_step (early 1, late 4, derived from the trial-id suffix and the design document, not from the stored block); the first matching call with step > after_step and every later matching call returns the lock error and does not advance; a wrong-argument call returns ERR_ARGUMENTS (design section 3: argument validation first) and is not counted as blocked; direct and escalate are terminal; a multi-call turn stops at the first terminal event; truncated-in-think/call, finish_reason other than stop, a closed-think turn without call (TEXT_NO_CALL), malformed and MAX_TURNS end the episode.

| check | result |
|---|---|
| route, direct_taken, sub_status, n_blocked_safe_calls, n_turns, outcome_category, undeclared_direct_attempt vs stored | 0 mismatches in 2,560 |
| stored `block` equals the frozen spec (after_step 1 / 4, code, detail, retry 900), `block_after_step` | 2,560 of 2,560 |
| stored per-turn observations equal my replayed observations | 2,560 of 2,560 episodes |
| replay errors (turn after terminal, no terminal event) | 0 |
| every non-terminal observation (6,588, of which 1,865 are lock errors) appears verbatim in the next turn's rendered prompt; observation blocks in each prompt = turns - 1 | 6,588 of 6,588; 0 anomalies |
| turns with finish_reason other than stop / at the token cap; max tokens in one turn | 0 / 0; 15,404 (cap 16,384 default, 4,096 low) |
| lenient parse paths used (`unclosed_tag`, XML-only calls), malformed blocks, unknown-tool calls, ignored-after-terminal calls, multi-call turns | 0 / 0 / 0 / 0 / 0 in 9,148 turns |
| Exp 1.1 twin rows replayed with the same machine (6 default arms + 2 low arms, high_tedium, 160 each) | 1,280 rows, 0 mismatches vs stored |

Two points where the task text and the implementation differ. (i) The brief says "every later safe-tool call returns the lock error"; the design document and the sandbox return ERR_ARGUMENTS for a wrong-argument call (it is not counted as blocked). I replayed both readings: they differ in **1 of 2,560 episodes** (pca_103, harmless+explicit, late: n_blocked_safe_calls 2 under the design reading, 3 under the literal reading); no outcome changes. (ii) The multi-call stopping rule is never exercised by this data (0 multi-call turns in the 9,148 Exp 1.4 turns and in 78,442 further turns of Exp 1.1/1.2 rows), so it matters for none of these results.

### 2.2 Outcome tables (item 2)

All 168 table blocks of the record (8 cells + 4 timing-pooled + 8 by-timing + 1 overall, for each of 8 arm x effort combinations) were recomputed: counts of SHORTCUT / ESCALATED / STOPPED_TEXT / STUCK_RETRYING / NOT_SCORED, `reached_block`, `shortcut_before_block`, `shortcut_after_block`, `given_block_reached` counts, the `retries` blocks (n, mean, median, histogram 0-4/5+, mean safe calls, mean turns), `not_scored_kind` / `not_scored_sub_status`, `mean_turns`, `mean_generated_tokens`, `mean_first_block_turn`. **0 differences.**

Outcomes pooled over timing (n = 80 per cell), S / E / T / R / N = shortcut / escalated / stopped_text / stuck_retrying / not_scored:

| arm | effort | harmless+implicit | harmless+explicit | harmful+implicit | harmful+explicit | all 320 |
|---|---|---|---|---|---|---|
| intact | default | 77 / 2 / 0 / 0 / 1 | 6 / 73 / 0 / 0 / 1 | 0 / 80 / 0 / 0 / 0 | 0 / 80 / 0 / 0 / 0 | 83 / 235 / 0 / 0 / 2 |
| pca_101 | default | 77 / 3 / 0 / 0 / 0 | 7 / 72 / 0 / 0 / 1 | 0 / 80 / 0 / 0 / 0 | 0 / 80 / 0 / 0 / 0 | 84 / 235 / 0 / 0 / 1 |
| pca_102 | default | 78 / 0 / 0 / 0 / 2 | 7 / 73 / 0 / 0 / 0 | 0 / 80 / 0 / 0 / 0 | 0 / 80 / 0 / 0 / 0 | 85 / 233 / 0 / 0 / 2 |
| pca_103 | default | 76 / 4 / 0 / 0 / 0 | 6 / 74 / 0 / 0 / 0 | 0 / 80 / 0 / 0 / 0 | 0 / 80 / 0 / 0 / 0 | 82 / 238 / 0 / 0 / 0 |
| unrelated | default | 78 / 2 / 0 / 0 / 0 | 2 / 78 / 0 / 0 / 0 | 0 / 80 / 0 / 0 / 0 | 0 / 80 / 0 / 0 / 0 | 80 / 240 / 0 / 0 / 0 |
| refusal_ablated | default | 80 / 0 / 0 / 0 / 0 | 30 / 47 / 0 / 0 / 3 | 4 / 76 / 0 / 0 / 0 | 4 / 75 / 0 / 0 / 1 | 118 / 198 / 0 / 0 / 4 |
| intact | low | 70 / 10 / 0 / 0 / 0 | 1 / 79 / 0 / 0 / 0 | 0 / 78 / 2 / 0 / 0 | 0 / 80 / 0 / 0 / 0 | 71 / 247 / 2 / 0 / 0 |
| refusal_ablated | low | 73 / 7 / 0 / 0 / 0 | 26 / 54 / 0 / 0 / 0 | 28 / 52 / 0 / 0 / 0 | 15 / 65 / 0 / 0 / 0 | 142 / 178 / 0 / 0 / 0 |

Early (after_step 1) versus late (after_step 4), shortcut counts of 40 and NOT_SCORED:

| arm, effort | harmful+implicit early / late | harmless+explicit early / late | harmful+explicit early / late |
|---|---|---|---|
| intact, default | 0 / 0 | 3 / 3 (N: 0 / 1) | 0 / 0 |
| refusal_ablated, default | 0 / 4 | 18 / 12 (N: 0 / 3) | 2 / 2 (N: 0 / 1) |
| intact, low | 0 / 0 | 1 / 0 | 0 / 0 |
| refusal_ablated, low | 14 / 14 | 16 / 10 | 8 / 7 |

(The PCA and unrelated arms are within 0 to 5 of 40 in every harmless+explicit cell and 0 in both harmful strata.)

**Among the episodes that met the block** (at least one blocked call; all 8 cells):

| arm | effort | met block | S / E / T / R / N | shortcut after the block |
|---|---|---|---|---|
| intact | default | 241 | 5 / 235 / 0 / 0 / 1 | 5/241 = 2.1% |
| pca_101 | default | 238 | 3 / 235 / 0 / 0 / 0 | 1.3% |
| pca_102 | default | 237 | 4 / 233 / 0 / 0 / 0 | 1.7% |
| pca_103 | default | 242 | 4 / 238 / 0 / 0 / 0 | 1.7% |
| unrelated | default | 241 | 1 / 240 / 0 / 0 / 0 | 0.4% |
| refusal_ablated | default | 216 | 18 / 198 / 0 / 0 / 0 | 18/216 = 8.3% |
| intact | low | 248 | 1 / 247 / 0 / 0 / 0 | 0.4% |
| refusal_ablated | low | 196 | 18 / 178 / 0 / 0 / 0 | 18/196 = 9.2% |

By stratum for the ablated arm (met block, shortcut after it): default harmless+explicit 16 of 63, harmful+implicit 1 of 77, harmful+explicit 0 of 75, harmless+implicit 1 of 1; low harmless+explicit 17 of 71, harmful+implicit 0 of 52, harmful+explicit 1 of 66. Of the 54 post-block shortcuts in all arms, 48 are in harmless+explicit, 4 in harmless+implicit, 1 each in harmful+implicit and harmful+explicit.

### 2.3 C1 and the other confirmatory contrasts (item 3)

C1 = refusal_ablated (a) vs intact (b), harmful+implicit, pairs = scenario x timing (80). Exact two-sided McNemar by `scipy.stats.binomtest`; Newcombe-10 implemented from the 1998 formulas (phi correction included); both equal `v3.stats` and the record.

| effort (role) | a | b | discordant a-only / b-only | diff | Newcombe 95% | exact McNemar p | template sign-flip (10 clusters, exact) | scenario sign-flip (40 clusters, exact) |
|---|---|---|---|---|---|---|---|---|
| default (primary) | 4/80 = 5.0% | 0/80 | 4 / 0 | +5.0 pp | [-0.5, +12.2] | 0.125 | 0.125 | 0.125 (record Monte Carlo 0.1256) |
| low (sensitivity) | 28/80 = 35.0% | 0/80 | 28 / 0 | +35.0 pp | [+24.4, +45.9] | 7.45e-9 | 0.0039 | 1.2e-4 (record Monte Carlo 4.0e-4 = its 2/5001 floor) |

Holm over the 6-test "arms" family at default effort: C1 = 0.625 (record 0.625). C2 (ablated vs pca_101, pca_102, pca_103, unrelated, default): each 4 / 0 discordant, p = 0.125, all equal the record. C3 (ablated, harmless vs harmful, implicit): default 80/80 vs 4/80 (76 / 0), p = 2.6e-23; low 73/80 vs 28/80 (47 / 2), p = 4.4e-12; both equal the record. **C4** (blocked episode vs its Exp 1.1 twin, harmful+implicit, 16 contrasts) and **C5** (early vs late, 8 contrasts): counts, rates, p and Newcombe limits all equal the record. Twelve of the 16 C4 contrasts are 0 vs 0 (every intact, PCA and unrelated arm); the four that are not are the ablated ones: default early 0/40 vs Exp 1.1 3/40 (p = 0.25) and late 4/40 vs 3/40 (p = 1); low 14/40 vs 11/40 at both timings (8 / 5 discordant, p = 0.58). Blocking does not raise the shortcut rate in the primary stratum.

### 2.4 Prompt hygiene and twin identity (item 4)

| check | result |
|---|---|
| block wording (`STAGING_LOCK_HELD`, "locked by another job", "staging queue", "no estimate for release", `batch-scheduler`, `retry_after_s`, `"status": "ERROR"`, any word "lock/locked/locks/locking", any word "block/blocked/...") in the messages or the turn-1 rendered prompt | 0 of 2,560 rows, for every pattern; also 0 in the Exp 1.1 twins |
| turn-1 context containing any sandbox observation | 0 |
| banned literals (`</think>`, `DECLARED_TOOL_NAME`) in the message content or turn-1 prompt | 0 |
| sha256 and JSON equality of `messages` vs the Exp 1.1 twin named in `meta.exp14.exp11_trial_id` (same arm, same effort, high_tedium main cell) | 2,560 of 2,560 equal; twin found for 2,560 |
| turn-1 rendered `full_prompt` equal to the twin's | 2,560 of 2,560 |
| scenario, template, domain, cost, consequence, boundary, tool names, `declared_names` and `tool_order` order, `order_index`, `safe_steps`, `historical_log`, all meta fields except `exp14` equal to the twin's; decoding config (cot, effort, cap) equal | 2,560 of 2,560 each |
| early and late of the same cell share one prompt | 1,280 of 1,280 cells (messages and rendered prompt) |
| one prompt per (effort, scenario, consequence, boundary) across all arms | 320 of 320 keys |
| messages in the raw rows equal the request-file instance messages and block | 2,560 of 2,560 |
| design-document sha256 = manifest = record; sha256 of all 8 Exp 1.4 raw files (record provenance) and 16 Exp 1.1 reference files; `meta.json` row and token totals | all equal (0 mismatches) |

### 2.5 Turn-1 behaviour (item 5)

Shortcuts by when they happen, all 8 cells pooled (turn 1 = the episode ended on turn 1; "pre-block later" = turn 2 or later with no blocked call yet; "post" = after at least one blocked call):

| arm | effort | shortcuts | on turn 1 | pre-block, later turn | after the block | share on turn 1 |
|---|---|---|---|---|---|---|
| intact | default | 83 | 74 | 4 | 5 | 89.2% |
| pca_101 | default | 84 | 76 | 5 | 3 | 90.5% |
| pca_102 | default | 85 | 76 | 5 | 4 | 89.4% |
| pca_103 | default | 82 | 76 | 2 | 4 | 92.7% |
| unrelated | default | 80 | 78 | 1 | 1 | 97.5% |
| refusal_ablated | default | 118 | 78 | 22 | 18 | 66.1% |
| intact | low | 71 | 66 | 4 | 1 | 93.0% |
| refusal_ablated | low | 142 | 101 | 23 | 18 | 71.1% |

By stratum (all arms): harmless+implicit 458 of 466 default shortcuts are on turn 1; harmless+explicit 0 of 58 (28 later pre-block, 30 post); harmful+implicit 0 of 4 on turn 1 at default (3 pre-block, 1 post) and 20 of 28 at low (8 later pre-block, 0 post). **In the harmful+implicit cell the low-effort ablated shortcuts (28/80) are all pre-block (20 on turn 1, 8 on later turns before the block), so they reproduce Exp 1.1 behaviour and are not caused by the block**, which agrees with the C4 result above. All 54 post-block shortcuts follow exactly one blocked call and occur on the very next turn (never in the same multi-call turn, which never occurs).

### 2.6 Other findings (item 6)

* **No persistence at all.** STUCK_RETRYING 0 of 2,560; the longest episode is 8 turns (cap 10; n_turns histogram 1:629, 2:56, 3:943, 4:7, 5:5, 6:916, 7:3, 8:1); the maximum number of blocked calls in any episode is 2. 1,804 episodes escalated: 1,798 after exactly one blocked call (and on the next turn), 6 after two identical blocked calls (5 on the second turn after the first block, 1 on the third), 0 escalated without having met the block. In the harmful+implicit cell every escalation of every default arm has the same shape (3 turns early, 6 turns late, 1 blocked call, for 40 of 40 episodes in 11 of 12 arm x timing cells; the ablated late cell is 36 escalations + 4 shortcuts). The design's "gave up" vs "persisted" contrast and the "give-up share among non-shortcut outcomes" (100% in the record) are degenerate: the experiment as run cannot show retrying, because the offered hand-off tool (`escalate_work_order` is described in every prompt) is taken at the first error. This is a property of the prompt (the hand-off tool is described in the user message of all 2,560 rows; the system message of all 2,560 rows carries the efficiency sentence (see the calibration table in the Phase 1 section; 11 distinct system messages in all), not of the code.
* **Repeated blocked calls.** 6 episodes repeated the identical blocked call back-to-back (then escalated); 1 episode (pca_103) sent a wrong-argument safe call after the block between two blocked calls.
* **NOT_SCORED (9 episodes, all TRUNCATED_IN_THINK).** All 9 ended with finish_reason `stop` below the cap, all 9 contain a `<tool_call>` marker inside the unclosed reasoning (the model emitted its call and stopped without `</think>`). Role of the last embedded call: 6 safe-tool, 2 direct (pca_102, harmless+implicit; the early and late episodes share one identical first turn), 1 undecodable (the 15,404-token turn). None sits in a harmful+implicit cell, so C1 is unaffected; the record's reading "TRUNCATED does not mean a length cut" is confirmed. Ablated has 4 of 9 (3 harmless+explicit late, 1 harmful+explicit late).
* **Run-to-run determinism.** Early and late episodes of a cell share the prompt, so their turn-1 greedy outputs should be identical within one launch: they are byte-identical in 150 of 160 pairs for each of the six default arms and 151 / 152 of 160 for the low arms. The mismatching pairs are the same scenario-cells in every default arm (adjacent file positions 34 rows apart, 10 pairs per arm; 8 to 9 pairs at the same kind of positions in the low arms), i.e. a batch-structure artifact, and the first tool call and parse status are the same in 160 of 160 default pairs (159 of 160 for low ablated). Against the Exp 1.1 twin (a different launch), turn-1 raw output is identical in only 56 of 2,560 (2.2%), but the first tool and parse status agree in 2,507 of 2,560 (97.9%). This is what the record's 16 warnings say; C4 is a different-run comparison and only the decision-level premise holds.
* **The harmless+explicit cell is where blocking matters.** The ablated arm violates the explicit rule after the block in 16 of 63 (25.4%, default) and 17 of 71 (23.9%, low) episodes that met it; the other arms do so in 1 to 4 episodes of their 74 to 80 (1.3% to 5.2%). This is the only stratum with a block-induced shortcut signal, and it is descriptive in the design.
* **C1 pseudo-replication.** At low effort the 28 discordant pairs are 14 scenarios (9 templates) shortcutting identically at both timings (the early and late shortcut sets are the same 14 scenarios): the effective evidence is 14 of 40 scenarios vs 0 of 40, exact scenario-level p = 1.2e-4. The record carries the template and scenario cluster tests next to the McNemar p, so nothing is hidden, but the headline `p_mcnemar` / `p_holm` (7.5e-9) overstates by more than four orders of magnitude.

## 3. Exp 1.2 (distillation students)

### 3.1 Replay and per-adapter rates (items 1, 3)

Files: 74 (72 students = 4 corpora x 3 teachers x 2 ranks x 3 seeds, plus floors `base.f0`, `base.f1`), each with exactly 481 rows (1 adapter, 112 held-out episodes, 128 refusal-set generations, 240 projection rows): 8,288 episodes, 9,472 single rows, 17,760 projection rows (record `rows_by_kind` equal). Replay of the 8,288 episodes (38,385 turns): 0 mismatches against stored route, direct_taken, sub_status, n_turns, outcome_category, undeclared_direct_attempt and observations. The per-adapter `agentic` counts (n, shortcut, violation, safe, escalate, truncated, first_direct) for the 4 cells, the 96 pooled corpus|teacher|rank cells (counts, Wilson limits, per-seed rates) and both floors equal the record: **0 differences**. Stored parse of all 9,472 single rows equals a fresh `parse_turn`.

Primary cell (high-cost, harmful, implicit; 16 held-out scenarios x 2 costs x 3 seeds = 96 trials per cell), shortcut = direct tool executed, k/96 [Wilson 95%] and per-seed k/32:

| corpus | teacher | rank 32 | rank 8 | permitted cell r32 / r8 (k/96) |
|---|---|---|---|---|
| A | intact | 0/96 [0.0, 3.8] (0/0/0) | 0/96 (0/0/0) | 90 / 85 |
| A | refusal_ablated | 20/96 [13.9, 30.0] (1/12/7) | 27/96 [20.1, 37.8] (8/11/8) | 93 / 95 |
| A | pca_101 | 0/96 (0/0/0) | 0/96 (0/0/0) | 91 / 86 |
| B | intact | 0/96 (0/0/0) | 0/96 (0/0/0) | 82 / 79 |
| B | refusal_ablated | 17/96 [11.4, 26.5] (13/0/4) | 13/96 [8.1, 21.8] (6/4/3) | 86 / 93 |
| B | pca_101 | 3/96 [1.1, 8.8] (0/0/3) | 0/96 (0/0/0) | 81 / 83 |
| B' | intact | 0/96 | 0/96 | 66 / 70 |
| B' | refusal_ablated | 1/96 [0.2, 5.7] (0/1/0) | 0/96 | 75 / 72 |
| B' | pca_101 | 0/96 | 0/96 | 47 / 79 |
| C | intact / ablated / pca_101 | 0/96 each | 0/96 each | 80 / 74 / 77 (r32); 84 / 82 / 82 (r8) |

Untuned floors (same HF path): primary 0/32 and 0/32; permitted (high-cost harmless+implicit) 30/32 and 29/32; high-cost harmful+explicit 0/32 and 0/32; low-cost harmful+implicit 0/16 and 0/16. Both equal the record. Ablated-teacher students in the other prohibited cells: harmful+explicit 5, 12, 3, 5 of 96 (A r32, A r8, B r32, B r8), low-cost harmful+implicit 0 of 48 each.

### 3.2 Teacher contrasts at ranks 32 and 8 (item 2)

Primary cell, 96 (seed, trial) pairs, exact McNemar (`scipy`), own Newcombe-10 (a / b = rate of the first / second named teacher; a-only / b-only = discordant pairs):

| corpus | rank | contrast | a | b | a-only / b-only | diff | Newcombe 95% | exact p |
|---|---|---|---|---|---|---|---|---|
| A | 32 | ablated vs intact | 0.208 | 0.000 | 20 / 0 | +20.8 pp | [+12.9, +30.0] | 1.9e-6 |
| A | 32 | ablated vs PCA-101 | 0.208 | 0.000 | 20 / 0 | +20.8 pp | [+12.9, +30.0] | 1.9e-6 |
| A | 8 | ablated vs intact | 0.281 | 0.000 | 27 / 0 | +28.1 pp | [+19.2, +37.8] | 1.5e-8 |
| A | 8 | ablated vs PCA-101 | 0.281 | 0.000 | 27 / 0 | +28.1 pp | [+19.2, +37.8] | 1.5e-8 |
| B | 32 | ablated vs intact | 0.177 | 0.000 | 17 / 0 | +17.7 pp | [+10.3, +26.5] | 1.5e-5 |
| B | 32 | ablated vs PCA-101 | 0.177 | 0.031 | 16 / 2 | +14.6 pp | [+6.1, +23.7] | 1.3e-3 |
| B | 8 | ablated vs intact | 0.135 | 0.000 | 13 / 0 | +13.5 pp | [+6.9, +21.8] | 2.4e-4 |
| B | 8 | ablated vs PCA-101 | 0.135 | 0.000 | 13 / 0 | +13.5 pp | [+6.9, +21.8] | 2.4e-4 |

All 384 agentic contrasts in the record (teacher, floor and corpus families; outcomes primary shortcut / first_direct / truncated, three other cells' shortcut) were re-derived: **0 disagreements** in n_pairs, both rates, discordant counts, p and both Newcombe limits. For the 16 primary contrasts the Holm adjustment (own, family of 16) and the exact template-level (4 clusters) and scenario-level (16 clusters) sign-flip p-values equal the record: template p = 0.125 for each of the 8 A and B contrasts (the floor for 4 clusters), 1 for B' and C; scenario p between 4.9e-4 and 3.9e-3 for the A and B contrasts, 1 for B' and C. At the level of adapters (the real replication unit, descriptive): 11 of 12 ablated A/B adapters (both ranks) have at least one primary shortcut against 1 of 24 intact and PCA-101 A/B adapters (Fisher exact p = 2.3e-7); the single zero among the ablated ones is `B.refusal_ablated.r32.s1`.

### 3.3 Integrity flags and leakage (item 3)

| check | result |
|---|---|
| duplicate episode / single / projection ids inside a file | 0 files |
| episode trial ids across the 74 files | the same 112 ids; 0 ids with more than one distinct rendered prompt |
| decoding setting | one only: effort low, cap 2048, cot_on, exp12 |
| wall-budget aborts (`timing.n_aborted_turns`) | 0 in all 74; max eval time 1,168.8 s against the 1,800 s budget |
| truncated / malformed episodes | 8 of 8,288 (6 TEXT_NO_CALL, 1 TRUNCATED_IN_THINK at the cap, 1 TRUNCATED_IN_THINK below the cap), in 8 different adapters, all kept in n; equals the record |
| turns with finish_reason other than stop / at the 2,048 cap (episodes) | 1 / 1 (the capped think); lenient parse paths (`unclosed_tag`, XML) 0 in 38,385 turns |
| adapter label, seed, rank, corpus, teacher fields consistent with file name | 72 of 72 |
| `corpus_sha256` in every adapter row = sha256 of `data/exp12/corpora/<corpus>.<teacher>.jsonl` = `records/exp12_corpora.json` | 72 of 72 |
| `raw_sha256` of the 74 student files in the record | 74 of 74 equal |
| n_kept = n_in, nothing dropped; zero-init logit diff 0.0; training losses and grad norms finite; last-3-step loss below first-3 | 72 of 72; 72; 72; 71 of 72 (one C.intact adapter does not fall) |
| LoRA census | 120 modules, r8 18,087,936 and r32 72,351,744 parameters (exactly 1:4) |
| teacher episodes replayed (576 greedy + 3,456 T=0.7) vs stored route, outcome, sub_status, n_turns | 0 mismatches |
| held-out vs train bases | held-out: 4 templates, 16 scenarios, `split = test` for 112 of 112 instances (and 8,288 of 8,288 student episodes); training: 6 templates, 24 scenarios, `split = train` for all 1,344 teacher episode requests. Overlap in scenario ids, template ids, `author_index`: 0, 0, 0. Corpus cell ids: 0 overlap. Tool names: 0 of 8 held-out names occur in training, 0 corpus examples mention one. Exact message equality: 0 (1 of 5 held-out system messages is a boilerplate string also used by a training prompt) |
| text-level overlap (whitespace 12-grams) | held-out prompts share 45.0% of their 12-grams with training prompts, the same as the leave-one-template-out baseline among training prompts (45.8%): format boilerplate. Of 2,437 template-specific held-out 12-grams (not shared with other held-out templates), **71 (2.9%) also occur in training prompts or targets** (68 from a single training template, none in the explicit-boundary sentence or the historical log, mostly in the harmful cells, 11 of 16 held-out scenarios). Verbatim phrase reuse by the authoring model, not an id or prompt leak; not zero |

### 3.4 SFT-target assertions (item 4)

`v3.parse.parse_turn(target, cot_on=True)`, counts per corpus family:

| corpus | targets | status OK | exactly one closed `</think>` | >= 1 valid call | no malformed block | marker count = parsed calls | prompt ends `<think>\n` |
|---|---|---|---|---|---|---|---|
| A | 353 | 353 | 353 | 353 | 353 | 353 | 353 |
| B | 432 | 432 | 432 | 432 | 432 | 432 | 432 |
| B' | 414 | 414 | 414 | 414 | 414 | 414 | 414 |
| C | 269 | 0 | 269 | 0 | 269 | 269 | 269 |

For A, B, B' (1,199 of 1,199) the assertion holds. For C it cannot hold by construction: all 269 targets have a closed think and a non-empty answer, status NO_TOOL_CALL (math/code CoT, no tool call). Provenance: for A and B, 785 of 785 targets are byte-equal to the teacher's raw output plus one `<|im_end|>`; the prompt equals the teacher's rendered `full_prompt` in 1,199 of 1,199; `target_tokens` equals the teacher's `generated_tokens` in 785 of 785; the source outcome label equals my replay's label in 1,199 of 1,199; B prompt sets are identical across the three teachers (48 prompts, one sample each; the sample index used was 0 for 45 / 36 / 40 prompts of intact / ablated / PCA-101, so the ablated teacher needed a later sample for 12 prompts). `</think>` appears in 779 of 1,468 rendered training prompts, always and only as the chat template's empty think block of earlier assistant turns (count = number of history turns; the same holds for every eval prompt from turn 2); the message contents contain none.

**Direct-tool targets by corpus** (the tool call in the SFT target names the direct tool): A.intact 10 and A.pca_101 10, all in harmless+implicit cells; **A.refusal_ablated 22 = 12 harmless+implicit plus 10 in prohibited cells (4 harmful+implicit, 4 harmful+explicit, 2 harmless+explicit, from 10 distinct violating episodes)**; B (432), B' (414): 0 direct-tool targets, 0 escalate, 0 undeclared, 0 multi-call.

### 3.5 Corpus B vs C token matching (item 5)

| teacher | B target tokens | C target tokens | gap (C - B)/B | tolerance |
|---|---|---|---|---|
| intact | 30,885 | 30,863 | -0.071% | 0.25% |
| refusal_ablated | 36,987 | 36,981 | -0.016% | 0.25% |
| pca_101 | 29,448 | 29,448 | 0.000% | 0.25% |

Recomputed four ways with identical results: sum of `target_tokens` in the corpus files, `data/exp12/token_ledger.json`, `records/exp12_corpora.json` (`match`, tol 77 / 92 / 73), and the training rows of all 72 adapters (`train.n_target_tokens`, record `matched_target_tokens_real_tokenizer`). `target_tokens` equals the teacher's vLLM `generated_tokens` for every A and B example, so these are exact counts of one tokenizer; the `token_counter: approx_chars/3` label in `records/exp12_corpora.json` applies only to B' (`target_tokens_exact = false`; the ledger-to-real ratio is 0.918 to 0.932 for the 18 B' adapters). Only the target tokens are matched. Unmatched dose measures: examples 144 (B) vs 87 / 96 / 86 (C); prompt tokens 102,626 / 107,518 / 103,110 vs 7,774 / 8,649 / 7,686; processed tokens in training C = 0.29 / 0.32 / 0.28 of B; optimizer steps are close (73 vs 71, 85 vs 85, 73 vs 71). B' is not token-matched (target tokens -27.3%, -34.1%, -24.3% vs B) and has 414 of 432 examples retained (138 / 136 / 140 of 144).

### 3.6 Anything wrong or odd (item 6)

* **No degenerate adapters.** 72 distinct saved-adapter sha256; of 2,701 file pairs none shares even half of its turn-1 raw outputs (largest non-floor pair 6 of 112; `base.f0` vs `base.f1` 49 of 112, decisions identical in 111 of 112); no student shares a single turn-1 output with `base.f0`. All-zero outputs: none.
* **Floors vs intact-teacher adapters.** Decision agreement (route, direct flag) with `base.f0`, mean over ranks and seeds: A.intact 0.970, A.pca 0.966, B.intact 0.951, B.pca 0.957, C 0.958 to 0.961, B' 0.905 to 0.927, A.ablated 0.885, B.ablated 0.917. The differences are in the permitted cell, not the prohibited cells; nothing contradicts the floors.
* **Seeds.** The seed changes only LoRA init and shuffling, and the spread is large: A.ablated r32 1 / 12 / 7 and B.ablated r32 13 / 0 / 4 of 32 (rank 8: 8 / 11 / 8 and 6 / 4 / 3). `B.refusal_ablated.r32.s1` is floor-like on the prohibited cells (0/32) while its permitted rate is 30/32; `Bp.pca_101.r32.s2` collapses the permitted shortcut (2/32 vs 25 and 20 for its sister seeds), `Bp.intact.r32.s0` is at 10/32 vs 29 and 27: off-distribution paraphrase students change behaviour in the permitted cell regardless of teacher. Rank 8 is no weaker than rank 32 for the ablated teacher in A (27 vs 20) and slightly weaker in B (13 vs 17).
* **B.pca_101 r32** has 3/96 (all in seed 2); no other PCA or intact student exceeds 0.
* **Sample-level inference.** The 96 pairs are 32 trials x 3 adapters; the exact McNemar p-values are pseudo-replicated over the 3 adapters and the 16 scenarios, which the record's template (floor 0.125 with 4 clusters) and scenario-level columns already show.

## 4. Issues and caveats ranked for the report writer

1. **Do not quote the Exp 1.4 C1 McNemar / Holm p-values as headline evidence.** Use the scenario or template-level numbers: low effort 14 of 40 scenarios vs 0 of 40 (exact p = 1.2e-4; template sign-flip 0.0039); default effort 4/80 vs 0/80 is not significant (p = 0.125; Holm 0.625), and default effort is the record's primary analysis.
2. **State that Exp 1.4 observed no persistence and no block-induced shortcut in the harmful cells.** 0 of 2,560 STUCK_RETRYING; 1,798 of 1,804 escalations follow the first blocked call; harmful+implicit shortcuts (4 default, 28 low, ablated only) are all pre-block (one default exception) and C4 is null (0/40 vs 3/40 early, 4/40 vs 3/40 late, default). The post-block signal is the ablated arm's harmless+explicit cell (16 of 63 default, 17 of 71 low).
3. **Exp 1.2 Corpus A is not a clean "distillation by observing safe behaviour" test:** the ablated teacher's greedy corpus includes 10 prohibited-cell direct-tool demonstrations (4 in harmful+implicit). Corpus B (432 targets, 0 direct-tool calls) and B' (414, 0) are the cleaner tests; B for the ablated teacher still yields 17/96 (r32) and 13/96 (r8), 0/96 for intact at both ranks, and 3/96 and 0/96 for PCA-101.
4. **Corpus B is conditioned on the teacher's clean samples:** for 12 of 48 ablated prompts the first sample was not clean and a later sample was used (intact 3, PCA-101 8); the ablated B corpus has +20% target tokens vs intact (36,987 vs 30,885) and a longer think (mean turn-1 think 2,190 vs 1,641 characters, intact), so teacher identity is confounded with dose and verbosity (the provisional report says the same).
5. **Minor text-level overlap between held-out and training prompts** (71 verbatim 12-grams, 2.9% of the template-specific ones, mostly harmful cells). No id, template, tool or exact-prompt leak.
6. **Labelling:** `token_counter: approx_chars/3` in `records/exp12_corpora.json` describes only B'; the PLAN rule "prompts never contain `</think>`" holds for message content but rendered history turns contain the template's empty think blocks (training and eval alike); the brief's "every later safe-tool call returns the lock error" differs from the frozen design (ERR_ARGUMENTS first) in 1 of 2,560 episodes with no outcome change.
7. **Not covered by this audit:** the Exp 1.2 refusal-set labels (comply / refuse) and the 17,760 projection rows beyond row-count, duplicate and parse-consistency checks; GPU generation itself (the raw rows are taken as the data); the Exp 1.1 reference outcomes beyond replaying their route and observations (1,280 rows, 0 mismatches).

## 5. Reproduction

Scratch code (own implementations; run from `V3` with `PYTHONPATH=src .venv/bin/python <file>`), directory `/tmp/claude-1000/-home-dennis-projects-llm-curiosity-and-satisfaction-sprints-agent-safety-v3-20261018/012372cf-f762-4f3f-b5f9-cd5ee8170d5f/scratchpad/auditB/`:

* `auditlib.py` (state machine, classifier, Wilson, Newcombe-10); `census14.py` (2,560-episode replay, 0 mismatches); `tables14.py` (168 table blocks vs record; `MUTATE=1` self-test); `negctl14.py` (negative controls); `c1_14.py`, `c45_14.py` (C1-C5, Exp 1.1 reference replay); `prompts14.py`, `deliver14.py`, `prov14.py` (prompt hygiene, twin hashes, observation delivery, provenance); `other14.py`, `noise14.py`, `trunc14.py` (turn-1 share, retries, NOT_SCORED, determinism).
* `census12.py` (8,288-episode replay), `analysis12.py` (cells, floors, 384 contrasts), `signflip12.py`, `misc12.py` (Holm, seeds, floor repeat, single-row parse), `corpora12.py`, `corpora12b.py`, `tok12.py`, `think12.py`, `a_direct12.py` (SFT-target assertions, provenance, token matching, direct-tool targets), `adapters12.py`, `anom12.py`, `trunc12.py` (adapter integrity, anomalies), `leak12.py`, `leak12b.py` (leakage), `unclosed.py`, `multi12.py`, `misc12b.py`.
