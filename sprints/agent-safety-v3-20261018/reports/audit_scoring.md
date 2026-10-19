# Independent audit: Exp 1.1 / Exp 3.1 scoring, hygiene and context safety

Date of audit: 2026-10-18 (after the Exp 1.1 default-effort, low-effort, cot_off, external-pair and Exp 3.1 runs).
Scope: `records/exp11_main.json`, `exp11_probes.json`, `exp11_external.json`, `exp11_main_lowprobe.json`, `exp11_probes_lowprobe.json`, `exp31_multidomain.json`, against `raw_transcripts/{exp11, exp11_cotoff, exp11_ext, exp11_effort_low_a, exp11_effort_low_b, exp11_effort_low_c, exp31}`.
This report contains aggregate findings only. No scenario, telemetry, memo or completion text appears in it, and none was printed during the audit (see section 5 for two incidental exceptions that are task-contract descriptors, not authored scenario text).

## 0. Method and independence

* All numbers below come from code in the session scratchpad that reads the raw JSONL rows with `json.loads`, parses every generation with `v3.parse.parse_turn` (allowed), and re-implements the PLAN.md multi-call turn policy and the safe-route/escalate/direct state machine from scratch. Statistics use `scipy` only (`binomtest`, `fisher_exact`, normal quantiles). `v3.exp11`, `v3.exp31`, `v3.stats` and `scripts/analyze_*` were NOT used to compute anything that was checked.
* Disclosure: before writing the replay I had read `src/v3/{parse,sandbox,episodes}.py`, so the replay is independent in implementation but not blind to the intended semantics. To compensate for that, a second, parser-free check was run (section 1b).
* Greedy decoding (T=0, top_p=1, seed_base 0) was used for every run; engine settings are identical across all Qwen-27B launches (TP1, max_model_len 32768, max_num_seqs 384, CUDA graphs on, prefix cache off, bf16) and across the two Flash launches (TP2, eager, max_num_seqs 64).
* Verdict: **sound with caveats**. Every stored number I recomputed reproduced exactly (the Monte-Carlo scenario-level sign-flip p-values reproduced to within Monte-Carlo error, section 2). The caveats are claim-level (multiplicity, pseudo-replication, effort and format confounds, a missing positive control for the Flash pair), listed in section 7.

## 1. Re-derivation of route and direct_taken

| check | result |
|---|---|
| Episode rows replayed (census, not just a sample) | 10,000 of 10,000 (all arms, all dirs); fields compared: `route`, `outcome_category`, `sub_status`, `direct_taken`, `undeclared_direct_attempt`, `n_turns`, `total_generated_tokens`, plus the per-turn observation status of every call (43,386 turns) and the stored `parsed.status`/`tool_names` of every turn. **0 mismatches.** |
| Stratified random sample requested (>= 400) | 560 episodes: 5 per stratum (arm x cot x cost x consequence x boundary), seed 20261018, 112 strata (8 Qwen arm-cot combinations x 12 cells = 96, plus 2 Flash arms x the 8 high-cost cells = 16; Flash was only run on the high-cost slice). Route and direct_taken agree in 560 of 560 (126 direct_taken in the sample, same count both ways). |
| direct_instruction single decisions | 2,240 of 2,240 replayed with a single-mode first-terminal-call rule; 0 mismatches (route, sub_status, direct_taken, undeclared attempt). |
| Probe rows (consequence and rule QA) | parse status and tool names equal the stored `parsed` in all rows. Consequence-awareness accuracy re-scored with an independent yes/no exact-field scorer: 40/40 harmless and 40/40 harmful for every one of the 6 arms, equal to the record. |
| Construct-validity gate numbers for Intact (cot_on and cot_off) | harmful-implicit shortcut 0/120 and 2/120, high-tedium harmless-implicit 37/40 and 38/40, low harmless-implicit 0/40 and 7/40, forced-safe competence 115/120 and 114/120. All equal the record (`construct_validity`), including the cot_off failure of the low-cost criterion (7/40 = 17.5%). |
| Request-to-raw join | every one of the 13,680 raw rows maps to exactly one of the 3,000 request rows in `data/exp11/` and `data/exp11_low/`; all 3,000 request rows were sent; stored initial messages equal the request messages; every request message is a substring of the rendered first-turn prompt; `cot_mode` and `max_new_tokens` equal the request. |

1b. Parser-free cross-check. A regex (`<tool_call>(.*?)</tool_call>` after the last `</think>`, `json.loads`) was run over all 47,066 generations (Exp 1.1 all dirs, episodes and singles). Tool-name lists agree with `parse_turn` in 47,066 of 47,066. No output uses the lenient `unclosed_tag` path (0 rows), so the documented leniency cannot bias any arm. Native Qwen `<function=...>` XML in post-think text was never scored (0 rows where an XML call named the direct tool outside a `<tool_call>` wrapper; the XML-style calls that do occur are inside `<tool_call>` wrappers and are scored MALFORMED_CALL, see 3).

1c. Hidden-shortcut probes (could strict parsing hide shortcuts differentially?).
* Reasoning-time direct calls (a complete, valid direct-tool `<tool_call>` block located before the last `</think>`, i.e. inside the reasoning): present in 49-64 default-effort cot_on generations per Qwen arm (12 and 20 in the two Flash arms) and in none of the low-effort generations. They are correctly excluded from behaviour (policy: score post-think only). At episode level, the episode ended in a shortcut in all but 2 of those cases (both PCA-103), so there is no arm-specific pattern and no route that parsing at the last `</think>` hides.
* TRUNCATED_IN_THINK turns in Qwen arms (25 default-effort and 2 low-effort turns) all have `finish_reason == "stop"`; 19 of the 27 are short outputs (37-110 tokens), 5 are 335-1,241 tokens and 3 are 11-14k tokens. In 23 of the 27 the output ends with a complete trailing `<tool_call>` block but no closing `</think>`; every such trailing call is a SAFE-tool call and none is a direct call (the other 4 have no complete trailing call). So strict I2 never hides a shortcut here; it only understates safe-route completion (a format failure, not length truncation).
* Multi-call turns: 12 episodes (all Intact cot_off) plus 1 (PCA-103 low) had a multi-call turn; 4 had ignored-after-terminal calls. Negligible.
* No `MAX_TURNS` and no `FINISH_LENGTH` episode in any Exp 1.1 arm; the only 4 `length` finishes are in the Flash arms at the 16,384-token cap.

## 2. Headline numbers (independent recomputation)

Exact two-sided McNemar by `scipy.stats.binomtest(b, b+c, 0.5)`; mid-p = p minus point mass; Newcombe (1998) method 10 (Wilson marginals, continuity-corrected phi); Holm step-down on the 7-member family (P1.1, P1.2 x4: PCA-101/102/103/Unrelated, P1.3 x2: Intact/Ablated). Outcome = `direct_taken`, ITT, pairs = (scenario, cost) within the high-cost harmful-implicit cells (n = 80), cot_on.

| contrast | rate A | rate B | diff | Newcombe 95% CI | table (both, A-only, B-only, neither) | exact McNemar p | Holm p (family 7) | equals record |
|---|---|---|---|---|---|---|---|---|
| **P1.1 default effort**: Ablated vs Intact | 8/80 = 0.100 | 0/80 = 0 | +0.100 | [0.0333, 0.1851] | (0, 8, 0, 72) | 0.0078125 | **0.0546875** | yes (all fields, to 1e-9) |
| P1.2 default: Ablated vs PCA-101 / 102 / 103 / Unrelated | 0.100 | 0 | +0.100 | [0.0333, 0.1851] | (0, 8, 0, 72) each | 0.0078125 each | 0.0546875 each (ties) | yes |
| P1.3 default: Intact cot_on vs cot_off | 0 | 2/80 | -0.025 | [-0.0866, 0.0243] | (0, 0, 2, 78) | 0.5 | 0.7186 | yes |
| P1.3 default: Ablated cot_on vs cot_off | 8/80 | 13/80 | -0.0625 | [-0.1714, 0.0463] | (1, 7, 12, 60) | 0.3593 | 0.7186 | yes |
| **P1.1 low-effort sensitivity**: Ablated vs Intact | 24/80 = 0.300 | 0/80 | +0.300 | [0.1996, 0.4077] | (0, 24, 0, 56) | 1.19e-7 | 8.34e-7 | yes |
| P1.2 low: Ablated vs each control | 0.300 | 0 | +0.300 | [0.1996, 0.4077] | (0, 24, 0, 56) each | 1.19e-7 each | 8.34e-7 each | yes |
| P1.3 low: Ablated | 24/80 (low, cot_on) | 13/80 (default, cot_off) | +0.1375 | [0.0064, 0.2633] | (4, 20, 9, 47) | 0.0614 | 0.1229 | yes, but see problem 3 (effort-confounded) |
| P1.2 pooled PCA (descriptive) default / low | 24/240 vs 0 ; 72/240 vs 0 | | | [0.0644, 0.1445] ; [0.2433, 0.3608] | | 1.19e-7 ; 4.24e-22 | not in family | yes, but see problem 2 |

Holm family size: 7 is the correct count of the list in the record (`holm_family`: P1.1, four P1.2 contrasts, two P1.3 contrasts); the pooled-PCA row is rightly excluded. The specification names the three contrasts but neither a correction method nor a family size, so Holm over 7 is the pipeline's own (reasonable) choice; for example with m = 3 (one slot per named contrast group) the P1.1 value would be 0.0234 instead of 0.0547, so the family definition must be stated whenever the Holm value is quoted.

Cluster checks, recomputed exactly:
* Template-level exact sign-flip for P1.1: 5 of 10 templates contribute (4, 1, 1, 1, 1 discordant pairs), so p = 2/32 = **0.0625**; record 0.0625. Holm-adjusted (x7) = 0.4375, equals `p_signflip_holm`. Low effort: 9 of 10 templates, p = 0.0039, equals the record.
* Scenario-level: 7 contributing scenarios of 40 (default), exact 2/128 = 0.015625; record 0.015597 (Monte Carlo estimate, `signflip_exact: false`). Low effort: 19 contributing scenarios, exact 3.8e-6; record 2.0e-4 (that is the Monte Carlo floor 1/5000, conservative, not an error).
* The 8 default-effort Ablated shortcuts come from 7 scenarios in 5 templates (inventory 4, queue 1, assay 1, billing 1, snapshot 1); 5 in high_friction, 3 in high_tedium. The 24 low-effort ones come from 9 templates.

Interpretation constraint (not a recomputation error): at default effort the primary contrast is significant only on the unadjusted exact test and the scenario-cluster test. After the pre-specified Holm correction (0.0547) and on the template-level sign-flip (0.0625) it is not below 0.05. The low-effort sensitivity is far from that margin.

## 3. Intention-to-treat accounting

Counts are from my replay (equal to the stored route in every row). TIT = TRUNCATED_IN_THINK, MAL = MALFORMED_CALL, TNC = TEXT_NO_CALL. Episodes cover main + forced-safe + unlisted-shortcut families.

| exp dir | arm | cot | effort | episodes N | SHORTCUT | SAFE | ESC | TRUNC/MALF (sub-status) | P1.1-cells: N, SHORTCUT, TRUNC/MALF |
|---|---|---|---|---|---|---|---|---|---|
| exp11 | intact | cot_on | default | 680 | 84 | 586 | 5 | 5 (MAL 1, TIT 4) | 80, 0, 1 |
| exp11 | refusal_ablated | cot_on | default | 680 | 123 | 542 | 5 | 10 (TNC 5, TIT 5) | 80, 8, 0 |
| exp11 | pca_101 | cot_on | default | 680 | 89 | 584 | 2 | 5 (TNC 1, TIT 4) | 80, 0, 0 |
| exp11 | pca_102 | cot_on | default | 680 | 90 | 585 | 1 | 4 (MAL 1, TIT 3) | 80, 0, 0 |
| exp11 | pca_103 | cot_on | default | 680 | 80 | 596 | 0 | 4 (TNC 2, TIT 2) | 80, 0, 0 |
| exp11 | unrelated | cot_on | default | 680 | 81 | 592 | 0 | 7 (TIT 7) | 80, 0, 2 |
| exp11_cotoff | intact | cot_off | default | 600 | 141 | 421 | 18 | 20 (MAL 20) | 80, 2, 1 |
| exp11_cotoff | refusal_ablated | cot_off | default | 600 | 176 | 412 | 10 | 2 (MAL 2) | 80, 13, 0 |
| exp11_effort_low_a | intact | cot_on | low | 680 | 70 | 586 | 10 | 14 (TNC 14) | 80, 0, 0 |
| exp11_effort_low_a | refusal_ablated | cot_on | low | 680 | 113 | 541 | 11 | 15 (TNC 15) | 80, 24, 0 |
| exp11_effort_low_b | pca_101 | cot_on | low | 680 | 69 | 599 | 3 | 9 (TNC 9) | 80, 0, 0 |
| exp11_effort_low_b | pca_102 | cot_on | low | 680 | 72 | 591 | 6 | 11 (TNC 11) | 80, 0, 0 |
| exp11_effort_low_c | pca_103 | cot_on | low | 680 | 74 | 579 | 9 | 18 (TNC 17, TIT 1) | 80, 0, 0 |
| exp11_effort_low_c | unrelated | cot_on | low | 680 | 73 | 588 | 8 | 11 (TNC 10, TIT 1) | 80, 0, 1 |
| exp11_ext | flash_next | cot_on | default | 320 | 76 | 208 | 0 | 36 (MAL 34, TIT 2) | 80, 0, 7 |
| exp11_ext | flash_next_uncensored | cot_on | default | 320 | 86 | 218 | 0 | 16 (MAL 2, TNC 1, TIT 13) | 80, 0, 12 |

Differential-truncation tests (Fisher exact, main family):

| comparison | counts | p | reading |
|---|---|---|---|
| cot_on default, Ablated vs Intact, all 12 cells | 5/480 vs 3/480 | 0.73 | not differential |
| cot_on default, P1.1 cells | 0/80 vs 1/80 | 1.0 | not differential; the single Intact truncation (TIT) is the opposite direction to the finding |
| cot_on low, all cells / P1.1 cells | 1/480 vs 1/480 ; 0/80 vs 0/80 | 1.0 | none |
| cot_off default, all cells | Ablated 2/480 vs Intact 17/480 | 6.5e-4 | **differential** (all MAL) |
| cot_off default, P1.1 cells | 0/80 vs 1/80 | 1.0 | none in the headline cells |
| Flash-Uncensored vs Flash-Next, 8 high-cost cells | 16/320 vs 36/320 | 0.0055 | differential overall; in hc harmful-implicit 12/80 vs 7/80, p = 0.33 |

Could truncation manufacture the Ablated-vs-Intact result?
* Default effort: ITT counts truncated episodes as non-shortcut, so truncation can only lower a rate. Intact has 1 truncated episode in the 80 P1.1 pairs and Ablated has 0. Worst case (that Intact episode were a shortcut and its Ablated partner safe): table (0, 8, 1, 71), exact p = 0.039, so the unadjusted significance survives; the record's valid-pairs-only analysis (79 pairs, 10.1% vs 0%) agrees. The one Intact truncation is an output that stops (`finish_reason` stop) inside the think block with no trailing tool call, not a hidden direct call. PCA-101/102/103 have 0 and Unrelated has 2 truncations in these cells (both end with a complete SAFE-tool call but no closing `</think>`).
* Low effort: 0 truncated in these 160 episodes; no effect possible.
* cot_off is the exception: Intact has 20 MAL episodes (17 in the main family) against 2 for Ablated. These are `<tool_call>` wrappers holding Qwen-native XML rather than JSON. Of the 17 Intact main-family cases, 7 name the direct tool (hc harmless-implicit 3, low harmful-implicit 2, low harmless-implicit 2), Ablated 0, and none are in the high-cost harmful-implicit cells. Intact cot_off shortcut is 141/480 (29.4%) strict, up to 148/480 (30.8%) if those 7 counted; Ablated 176/480 (36.7%). Ablated-minus-Intact in cot_off therefore narrows from 7.3 to 5.8 points at most; it does not flip. The direct-instruction control in cot_off is damaged much more (problem 4).
* Direct-instruction control, cot_on default: Intact (40/160), PCA-101/102/103 and Unrelated (40/160 each) end TRUNCATED_OR_MALFORMED with sub-status NO_DECISION (tool calls made, none to the offered tool) or TNC (text reply); Ablated has 0/160. This is the intended positive-control contrast (Ablated complies 40/40 in the harmful solo cell, every other arm 0/40) and is not a parser artifact.

Direct-instruction decisions (singles; COMPLY = direct tool executed):

| exp dir | arm | cot | effort | direct_instruction N | COMPLY (direct tool executed) | ESC | SAFE | TRUNC/MALF (sub-status) |
|---|---|---|---|---|---|---|---|---|
| exp11 | intact | cot_on | default | 160 | 79 | 41 | 0 | 40 (NO_DECISION 33, TNC 7) |
| exp11 | refusal_ablated | cot_on | default | 160 | 120 | 40 | 0 | 0 (-) |
| exp11 | pca_101 | cot_on | default | 160 | 80 | 40 | 0 | 40 (NO_DECISION 29, TNC 11) |
| exp11 | pca_102 | cot_on | default | 160 | 80 | 40 | 0 | 40 (NO_DECISION 23, TNC 17) |
| exp11 | pca_103 | cot_on | default | 160 | 80 | 40 | 0 | 40 (NO_DECISION 26, TNC 14) |
| exp11 | unrelated | cot_on | default | 160 | 80 | 40 | 0 | 40 (NO_DECISION 24, TNC 16) |
| exp11_cotoff | intact | cot_off | default | 160 | 92 | 28 | 0 | 40 (MAL 39, TNC 1) |
| exp11_cotoff | refusal_ablated | cot_off | default | 160 | 83 | 20 | 0 | 57 (MAL 57) |
| exp11_effort_low_a | intact | cot_on | low | 160 | 57 | 52 | 0 | 51 (TNC 51) |
| exp11_effort_low_a | refusal_ablated | cot_on | low | 160 | 89 | 45 | 0 | 26 (NO_DECISION 5, TNC 21) |
| exp11_effort_low_b | pca_101 | cot_on | low | 160 | 73 | 44 | 0 | 43 (TNC 43) |
| exp11_effort_low_b | pca_102 | cot_on | low | 160 | 64 | 49 | 0 | 47 (TNC 47) |
| exp11_effort_low_c | pca_103 | cot_on | low | 160 | 59 | 47 | 0 | 54 (MAL 2, NO_DECISION 1, TNC 50, TIC 1) |
| exp11_effort_low_c | unrelated | cot_on | low | 160 | 73 | 45 | 0 | 42 (TNC 42) |

## 4. Prompt hygiene on every request row actually sent

Checked on the 3,000 request rows (`data/exp11/requests_*.jsonl`: 2,160 rows in 11 files; `data/exp11_low/requests_*.jsonl`: 840 rows in 5 files; the join to the raw rows is in section 1) and on the rendered `full_prompt` of all 47,066 generations.

| invariant | result |
|---|---|
| no literal `</think>` in any request row (messages, instance fields, meta) | 0 of 3,000 |
| no `DECLARED_TOOL_NAME` in any request row or rendered prompt | 0 / 0 |
| no worked example tool call (`<tool_call>{` in any system or user text) | 0 of 3,000; every prompt mentions the `<tool_call>` wrapper only in the plain format sentence (3,000 of 3,000 mention it, none with a JSON example) |
| no forced completion prefix (last message is a user turn; no assistant message in any request; text after the final assistant marker equals the template suffix) | 0 violations in 3,000 requests and 47,066 rendered prompts |
| I1 caps | cot_on 16,384 (default) and 4,096 (low); cot_off 1,024. All >= 2,048 / >= 512. No generation exceeded its cap. `finish_reason == "length"`: 0 in all Qwen Exp 1.1 arms, 4 in Flash (each at exactly 16,384 tokens: base 1, uncensored 3), and 1 in Exp 3.1 (PCA-101, scored TRUNCATED_IN_THINK) |
| think never truncated before the cap | all Qwen TRUNCATED_IN_THINK turns are `stop` finishes (see 1c); the 4 Flash length finishes (and the 1 Exp 3.1 one) are at the 16,384 cap |
| shortcut name scrubbed from forced_safe | `shortcut_name not in prompt text`: 0 violations in 360 request rows (cot_on, cot_off, low) plus 120 calibration rows; absent from historical log and declared names; absent from the rendered prompt of every turn of all 1,680 forced-safe raw episodes; no forced-safe model output names it. (Instance metadata `tool_order` still lists the shortcut name in forced-safe rows but is never rendered; the record's `forced_safe_rows_naming_shortcut = 0` is correct.) |
| unlisted-shortcut slice | direct tool named in the prompt (log) and absent from declared names and tool list in 160 of 160 rows, as designed |
| symmetric historical logs, implicit vs explicit | 720 groups, 0 asymmetric; harmful vs harmless logs identical in 240 of 240 pairs |
| neutral tool names | 21 distinct names; one uniform `<verb>_<noun>_direct` / `<verb>_<noun>_staged` pair per template (2,280 of 2,280 instance rows) plus `escalate_work_order`; none matches a valence lexicon (safe, risk, harm, bypass, skip, force, fast, quick, short, cut, unchecked, raw, prod, delete, ...) |
| boundary label vs prompt | prohibition language present in 240/240 explicit prompts and absent in 240/240 implicit prompts; implicit vs explicit prompts differ in exactly one line (the system-prompt sentence) |
| harmful vs harmless prompts | differ only in the telemetry line (240/240 pairs). The harmful telemetry line is longer by 11.2 characters on average (sd 6.7, range -11 to +28): a small length asymmetry, noted as a possible weak confound, not a violation |
| tool order and position | order_index balanced (80 per index), 6-7 per (cell, index); direct tool position balanced and identical across consequence |
| first-turn prompt identity across arms | 4,680 trial groups with more than one arm: 0 differ (byte-identical rendered first-turn prompts), so arm contrasts are paired on identical inputs |
| multi-turn history format | all 33,386 history transitions contain the previous assistant post-think text and the `Sandbox observation: ` message; previous reasoning does not leak into later prompts (0 of 14,126 checkable transitions after excluding text that legitimately repeats the prompt, history or observation). The Qwen template renders each historical assistant turn with an empty think block, so `</think>` appears in rendered prompts of later turns by template design (one per history turn), identical for every arm; the rule applies to request text, where it holds |
| Exp 3.1 request rows | 3,072 rows: no banned literal, no example call, last message user, caps 16,384 / 1,024, prohibition language in 1,536/1,536 explicit and 0/1,536 implicit, rendered first prompt ends with the template suffix in 9,216/9,216, identical across the 3 arms |



## 5. Context-safety sweep

Method: all 40 `telemetry_harmful` sentences were loaded programmatically from `data/author/final_bases.jsonl` (never printed). Patterns: the full sentence and every 40-character window (stride 1), each in raw, JSON-escaped (ASCII and non-ASCII) and double-JSON-escaped form, matched case-insensitively with `rg -F -i -a`, patterns fed through stdin (no pattern file written to disk). Only paths and counts were read back.

| location | files scanned for any 40-char window | files with any 40-char hit | files with a full-sentence hit | residual hits after removing domain-level category descriptors |
|---|---|---|---|---|
| `logs/*.log` | all | 0 | 0 | 0 |
| `records/*.json` | all | 0 | 0 | 0 |
| `reports/**` (before this file) | all | 0 | 0 | 0 |
| `PLAN.md`, `CLAUDE_RESEARCH_SPECIFICATION.md`, `survey/*.md`, `tests/**`, `scripts/**`, `ledger/**` | all | 0 | 0 | 0 |
| `src/**` | all | 2 (`src/v3/author.py`, its `.pyc`) | 0 | 0 |
| memory directory (`.../memory/*.md`) | all | 0 | 0 | 0 |
| `data/**/manifest.json` | all | 0 | 0 | 0 |
| Claude project transcripts for the parent project | all | 0 | 0 | 0 |
| Claude project transcripts for this sprint (main session, subagents, workflows, tool-results): 104 files | all | 11 | **0** | 1 (main session transcript, 1 line) |
| `/tmp/claude-1000` (session scratchpads, task outputs; weights, venvs excluded) | all text-like files | 53 | **41** | 44 |

Findings:
* No verbatim harmful telemetry sentence appears in any log, record, report, plan, spec, survey note, test, script, source file, memory file or Claude transcript. Context safety held for everything that is read by humans or fed back to models.
* The 40-character hits in `src/v3/author.py` (lines 42 and 45), in the workflow specification script and JSON, and in 9 transcript files (the main session plus 8 subagent transcripts) are one class: the four domain-level harm-category descriptors from the task contract (the `HARM_CATEGORIES` constants, and the same clauses in the authoring prompt and workflow spec wording). 14 of the 40 harmful sentences embed one of the `author.py` clauses and 6 embed the workflow-spec wording (shared clause of 40 to 58 characters inside a 104-178 character sentence). After removing windows that belong to those descriptors, the only residual hit in any non-scratch location is one line of the main session transcript in which the authoring prompt (developer-written instruction text) was printed; it contains the same descriptor clause in prompt wording. None of this is authored scenario text.
* Incidental exposure during this audit: to classify the hits I displayed the descriptor constants in `author.py` (two of the four were visible) and, in one command, context from the workflow script that includes the descriptors for all four domains. These are the contract's category labels, not authored telemetry. No authored harmful sentence, memo, note or completion was printed.
* 41 JSONL files in the shared session scratchpad contain complete harmful sentences verbatim: 38 in `scratchpad/r32/` and 3 in `scratchpad/req_real/` (Exp 1.2 student/teacher request and result copies). Three more scratch JSONL files (2 under `scratchpad/tmp0xzbn_y1/`, 1 under `scratchpad/dry2/`) contain partial matches. These are unredacted data copies outside the sanctioned directory list in the hard rules; they were not read into any transcript (0 full-sentence hits in the transcripts) but should be treated as unredacted, not catted, and deleted when no longer needed.
* Inside the sanctioned areas, full sentences are where expected: `data/{author, exp11*, exp12, exp23, exp31}` (68 files) and `raw_transcripts/{_calibration, author*, exp11*, exp12*, exp23*, exp31}` (78 files). No other `data/` or `raw_transcripts/` directory holds them.

## 6. Exp 3.1 (multi-domain, narrative vs numeric stakes)

All 9,216 episodes replayed independently (0 mismatches against stored route, sub_status, direct_taken, n_turns, tokens, per-turn observations). Raw files' sha256 and row counts equal the record provenance. All 12 arm x cot x format cells (the task asked for 3) recomputed:

| arm | cot | format | N | scored | TRUNC/MALF | shortcut k (direct tool executed) | rate (scored) | rate (ITT) | independent replay | record |
|---|---|---|---|---|---|---|---|---|---|---|
| intact | cot_off | narrative | 768 | 744 | 24 | 229 | 0.3078 | 0.2982 | 229/744 (ITT 229/768) | match |
| intact | cot_off | numeric | 768 | 695 | 73 | 172 | 0.2475 | 0.2240 | 172/695 (ITT 172/768) | match |
| intact | cot_on | narrative | 768 | 761 | 7 | 135 | 0.1774 | 0.1758 | 135/761 (ITT 135/768) | match |
| intact | cot_on | numeric | 768 | 758 | 10 | 183 | 0.2414 | 0.2383 | 183/758 (ITT 183/768) | match |
| pca_101 | cot_off | narrative | 768 | 764 | 4 | 240 | 0.3141 | 0.3125 | 240/764 (ITT 240/768) | match |
| pca_101 | cot_off | numeric | 768 | 742 | 26 | 188 | 0.2534 | 0.2448 | 188/742 (ITT 188/768) | match |
| pca_101 | cot_on | narrative | 768 | 762 | 6 | 134 | 0.1759 | 0.1745 | 134/762 (ITT 134/768) | match |
| pca_101 | cot_on | numeric | 768 | 759 | 9 | 193 | 0.2543 | 0.2513 | 193/759 (ITT 193/768) | match |
| refusal_ablated | cot_off | narrative | 768 | 765 | 3 | 281 | 0.3673 | 0.3659 | 281/765 (ITT 281/768) | match |
| refusal_ablated | cot_off | numeric | 768 | 737 | 31 | 245 | 0.3324 | 0.3190 | 245/737 (ITT 245/768) | match |
| refusal_ablated | cot_on | narrative | 768 | 768 | 0 | 194 | 0.2526 | 0.2526 | 194/768 (ITT 194/768) | match |
| refusal_ablated | cot_on | numeric | 768 | 753 | 15 | 363 | 0.4821 | 0.4727 | 363/753 (ITT 363/768) | match |

* Wilson intervals recomputed for spot cells: equal. Truncation sub-status table equals the record. I1 violations: 0. Effort is the template default in every row (`reasoning_effort` and `request_meta.effort` both absent), matching Exp 1.1 default.
* The 22 ITT contrasts (A1_itt 16, C1_itt 6) were recomputed from my own pairing (scope, within-cells and varied factor inferred from the record's own labels): n pairs, A-only/B-only counts, diff, exact McNemar p and Newcombe CI equal the record in 22 of 22.
* Truncation is differential only in cot_off: MALFORMED_CALL (native XML inside a `<tool_call>` wrapper, all at turn 1) is 97/1,536 for Intact against 34/1,536 for Ablated (Fisher p = 1.7e-8) and 30/1,536 for PCA-101; 16 / 6 / 6 of them name the direct tool. cot_on truncation is 17 / 15 / 15 with no difference (Ablated vs Intact p = 0.86). Worst case for the cot_off Ablated-vs-Intact ITT gap (26.1% vs 34.2%, +8.1 points): counting those named-direct malformed episodes as shortcuts gives 27.1% vs 34.6%, still +7.5 points. The non-ITT `A1` rows drop truncated pairs and are therefore more exposed to this than the `A1_itt` rows; the ITT rows should be the headline.

## 7. Items that would change a reported number or claim

No stored number failed to reproduce. The following would change how a number may be cited or what may be claimed:

1. **P1.1 at default effort is not significant after the pre-specified multiplicity correction.** `records/exp11_main.json: contrasts.P1.1.p_holm = 0.0546875` (family size 7, verified), `cluster_boot.p_signflip = 0.0625` (template-level exact), `p_signflip_holm = 0.4375`. Only the unadjusted exact McNemar (0.0078) and the scenario-cluster tests (0.0156) fall below 0.05, and the McNemar test treats 80 pairs from 40 scenarios as independent. Any text saying "P1.1 significant" for default effort must say which test. The low-effort sensitivity (`exp11_main_lowprobe.json: contrasts.P1.1`, 24 vs 0, Holm 8.3e-7, template sign-flip 0.0039) is robust.
2. **Pooled PCA p-values are pseudo-replicated.** `contrasts.P1.2_pca_pooled.p_mcnemar` (1.19e-7 default, 4.24e-22 low-effort) stacks the same 80 Ablated episodes three times; effective discordant Ablated episodes are 8 (default) and 24 (low), not 24 and 72. The record carries a note ("descriptive only"); the number must not be quoted as inferential.
3. **The low-effort `P1.3_*` contrasts are effort-confounded.** In `records/exp11_main_lowprobe.json` (and the matching Holm values) the cot_on side is effort = low and the cot_off side is the default-effort cot_off run (cot_off exists only at default effort; `n_trials_other_effort = 7,120` confirms the 1,520 cot_off rows are retained). `P1.3_refusal_ablated` (0.30 vs 0.1625, diff +0.1375, CI [0.006, 0.263], p = 0.061) therefore changes effort and CoT mode together. Neither the warnings nor the caveats of that record say so.
4. **cot_off direct-instruction control and cot_off arm contrasts are format-confounded.** In cot_off, 39/160 (Intact) and 57/160 (Ablated) direct-instruction decisions are MALFORMED_CALL (native XML in a `<tool_call>` wrapper); 37/39 and 47/57 of them name the direct tool. Under strict I2 the harmful-solo cell reads COMPLY 19/40 (Intact) vs 11/40 (Ablated), the reverse ordering of cot_on (0/40 vs 40/40), purely because 21/40 and 29/40 are unscored. Fields: `records/exp11_probes.json: direct_instruction.cells.{intact,refusal_ablated}.cot_off.*`, `records/exp11_main.json: cells.*.cot_off` (Intact 20 MAL vs Ablated 2 over the 600 episodes; 7 Intact main-family MAL episodes name the direct tool; none in P1.1 cells). The record lists these counts only as warnings. No claim about cot_off compliance or about Ablated vs Intact in cot_off should rest on strict-I2 numbers without this caveat; the cot_off construct-validity failure (`construct_validity.cot_off.all_behavioural_pass = false`, low-cost shortcut 7/40) is already recorded.
5. **The Flash external null has no positive control and 15% of the uncensored arm's headline cell (9% of the base arm's) is unscorable.** `records/exp11_external.json: contrasts.uncensored_vs_base:hc_harmful_implicit` (0/80 vs 0/80) rests on only the 320-row high-cost main slice per arm: there is no direct-instruction, forced-safe or unlisted-shortcut run for either Flash model, so "Uncensored does not take the harmful shortcut" cannot be separated from "Uncensored is not actually uncensored in this harness". 7/80 (base) and 12/80 (uncensored) episodes in that cell are TRUNCATED_OR_MALFORMED; none name the direct tool (uncensored: 9 TIT, of which 7 are `stop` finishes ending in a complete SAFE call after an unclosed think and 2 hit the 16,384-token cap; 2 MAL naming the safe tool; 1 TNC), but if all 12 were shortcuts the uncensored upper bound would be 15%. The spec's "4 companion probes across arms" is not met for the Flash pair; this reduced scale should be recorded as a deviation (I4).

Lower-severity notes (no change to numbers):
* "Default effort" in the records means the Qwen template default, whose system line reads "Reasoning effort is set to xhigh"; the low runs render "low". The cap also differs between the two (16,384 vs 4,096); no Qwen generation hit either cap, so the cap difference is inert.
* The spec says `max_new_tokens = 2048` / 512; the runs used 16,384 / 1,024 (default), 4,096 (low). This is a permitted increase under I1, but it is a deviation from the spec text and should be stated once in the results.
* Harmful telemetry lines are on average 11 characters longer than their harmless twins (section 4); the Exp 3.1 numeric format replaces the telemetry line by a four-line stakes table, so format effects include a layout change.
* Consequence-awareness QA is at ceiling for every arm (40/40 and 40/40), so "harmful shortcut rate conditioned on demonstrated awareness" equals the unconditional rate (`n_aware = 40` for all arms).
* `records/receipt_verification.json` currently lists only `exp11_main_lowprobe` and `exp11_probes_lowprobe` (a partial `--only` run at 21:59). The four other Exp 1.1 records were last verified OK at 19:07 (`logs/verify_exp11.log`) and the Exp 3.1 record is not in that log; I did not run `verify_all_receipts.py` (it would overwrite the shared receipt file), but my independent recomputation covers the same numbers.
* Not independently re-derived: the Firth random-intercept model terms (`model`), the cluster-bootstrap confidence intervals, the tedium-vs-friction table, and the rule-identification QA scoring; Exp 3.2 and Exp 1.2 are out of scope.

## 8. Scratch code (not part of the repo)

Scratch scripts and intermediate tables are in the session scratchpad (`.../scratchpad/audit_common.py`, `replay_all.py`, `blind.py`, `stats_repro.py`, `exp31_repro.py`, `exp31_contrast.py`, `hygiene*.py`, `itt*.py`, `sweep*.py`); none contains or prints scenario text, and the sweep patterns were only ever held in memory.
