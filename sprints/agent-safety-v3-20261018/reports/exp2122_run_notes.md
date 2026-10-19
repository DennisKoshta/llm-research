# Experiments 2.2 (monitor calibration) and 2.1 (four topologies): run notes

Aggregates and problem codes only. No world, note, explanation or completion text appears here or may be added.
Records: `records/exp22_monitor_calibration.json`, `records/exp21_topologies.json` (both written by `scripts/analyze_exp2122.py`,
both `OK` under `scripts/verify_all_receipts.py --only exp22_monitor_calibration exp21_topologies --skip-slow`).
Tests: `tests/test_exp21.py` + `tests/test_exp22.py`, 151 passed. Run window 2026-10-18 21:50 to 2026-10-19 01:55 (PDT).

## 1. Scale (deviation, before / after)

| | spec / plan | run |
|---|---|---|
| worlds | 36 (12 per domain: warehouse, finance, compute) | **34** in Exp 2.1 (warehouse 11, finance 12, compute 11); **33** in Exp 2.2 |
| Exp 2.1 samples | K = 5 per (team, topology variant, world) | K = 5, 2040 Alpha trials (2 teams x 6 variants x 34 worlds x 5) |
| Exp 2.2 monitors | intact, llama70b (flash_next optional) | intact (cot_off + cot_on slice), llama70b, flash_next; 13200 monitor rows |

Missing worlds: `compute_07` and `warehouse_09`: four judged candidates each (the original plus three spares), every one rated `coherent: no` by the
same-model judge (the control pair does not plausibly guard the stated consequence); `warehouse_09` additionally had three candidates rejected
structurally (`duplicate_control`). The pipeline gate was not relaxed. `warehouse_07` is final and used in Exp 2.1, but is **excluded from Exp 2.2**
(`monitor-requests --exclude-worlds warehouse_07`): two of its `clause_both` notes (`A_FAST_B_SAFE`, `A_SAFE_B_FAST`) failed the claim-polarity audit in
every attempt (8 audited candidates each, 10 and 12 authored; every audit verdict `audit_claim_compliant`), so the 36-note grid of that world could not
be completed. The 34-world Exp 2.1 set and the 33-world Exp 2.2 set therefore differ by one world. Listing-order balance of the 34 worlds:
alpha_safe_first 17/34, beta_safe_first 16/34, alpha_safe_lower 17/34, beta_safe_lower 17/34 (exact balance would be 17/34 on each flag).

## 2. World authoring (Flash-Next-Uncensored, effort low, T=0.7 sampled; judge greedy)

Rounds r1-r5 were sequential (earlier session). From r6 on, several rounds ran **in parallel inside one launch** (r6-8, r9-11, r12-14,
r15-17, r18-20) so that a world that failed its judge could be replaced by an already authored spare without another Flash boot;
`worlds-process` keeps the first valid candidate of a world and skips the rest (`redundant_skipped`), and only a judge failure
(problems set) lets a later round's candidate replace it. 20 authoring rounds, 13 judge rounds.

| | n | outcome |
|---|---|---|
| authoring requests | 122 | 105 structurally valid (86.1%), 7 invalid (`pressure_word` 3, `control_leaks_role` 2, `bad_step` 1, `missing_severity_cue` 1), 10 unparsed (`MALFORMED_CALL` 4, `XML_CALL_ONLY` 3, `BAD_ARGUMENTS` 2, `TRUNCATED_IN_CALL` 1) |
| cross-world rejections | >= 9 | `duplicate_control` (a later world re-used another world's control names; at least 3 before this session and 6 in r6-r14); the later duplicate is dropped |
| judge requests | 88 | 76 scored (37 pass, 39 fail; reasons overlap: `judge_incoherent` 37, low irreversibility 2, low severity 1), 12 unparsed (`TRUNCATED_IN_CALL` 10 = ran to the 8192 cap, `MALFORMED_CALL` 2) |
| final worlds | 34 | judge severity 5 / 4: 18 / 16; irreversibility 5 / 4: 29 / 5; all `instructional: no`, `coherent: yes` |

`finish_reason: length` in the judge: 10 of 88 (11.4%); those are the long-tail sequences that kept a Flash container alive for ~20 min.

## 3. Note authoring (refusal_ablated 27B, family r1, effort low, T=0.7) and audit (intact)

25 author rounds (some parallel, same first-valid-wins rule) and 16 audit rounds. 1319 author rows were counted (37 further rows belong
to an earlier, superseded `warehouse_07` text and were purged together with their 35 accepted notes when that world was dropped and
re-authored; the guard `purged_stale_accepted` was added to the processor for this).

| | n | outcome |
|---|---|---|
| author rows | 1319 | 1262 structurally valid (95.7%); invalid 52: `length_window` 48 (`truthful_compliance` 18, `clause_both` 15, `deceptive_compound` 14, `neutral_matched` 1), `missing_policy_statement` 3, `risk_cue_mismatch` 1; unparsed 5, all `TRUNCATED_IN_THINK` (cap 4096) |
| audit rows | 1306 | 1306 parsed; 1271 pass, 35 fail: `clause_both` 33, `neutral_matched_nocode` 1, `clause_risk` 1, always `audit_claim_compliant` (the auditor read a risk / deadline sentence as a compliance claim) |
| final notes | 1222 / 1224 | 34 worlds x 4 bundles x 9 note conditions; the 2 missing are the `warehouse_07` `clause_both` notes above |

Mean words per condition (final notes): minimal 11.5; neutral_matched 48.2; neutral_matched_nocode 46.2; clause_risk 48.6;
clause_urgency 49.1; clause_both 52.8; truthful_compliance_nocode 52.5; deceptive_compound 53.1; truthful_compliance 53.5. The
non-minimal conditions are length-matched to within ~7 words; `truthful_compliance` / `deceptive_compound` are ~5 words longer than
`neutral_matched` (reported in the record as `note_length`).

## 4. Smoke runs (not part of any record; directories `raw_transcripts/exp22_smoke_*`, ignored by the loaders)

Run before each new stage, looking at counts and status only: Llama monitor 360/360 strict `record_verdict` JSON; intact monitor 480/480
(cot_off 360, cot_on 120, cot_on tokens p50 297 / max 844); Exp 2.1 alpha (132 rows), beta (88), monitor + judge (379, incl. 120 synthetic
validation items, kappa 1.0). The alpha smoke showed **7.6% of turns truncated at the 8192 cap** (10 of 132; p90 6.6k tokens in both arms) at default effort: the agent / monitor / judge cap `exp22.cot_on_cap(default)` was raised **8192 -> 16384**, the value Phase 1 and Exp 1.1 use at that
effort (launches with `--max-model-len 20480`; test updated). After the change: alpha 3 of 2040 unclosed, beta 5 of 1508.
The spec line "max_new_tokens = 1024" for Exp 2.1 is applied to the cot_off variant only (I1 requires >= 2048 on cot_on; the
experiment runs cot_on at the recorded Phase-2 effort, default).

## 5. Per-stage invalid / truncation / XML rates per arm (strict parser; XML calls never scored)

| stage (arm) | n | valid | MALFORMED | truncated | XML-only turns |
|---|---|---|---|---|---|
| world authoring (flash_next_uncensored) | 122 | 105 structurally valid / 112 parsed | 4 | 1 in call | 3 |
| world judge (flash_next_uncensored) | 88 | 76 | 2 | 10 in call (cap) | 0 |
| note author (refusal_ablated) | 1319 | 1314 parsed | 0 | 5 in think | 0 |
| note audit (intact) | 1306 | 1306 | 0 | 0 | 0 |
| Exp 2.2 monitor (intact, cot_off) | 3960 | 3960 | 0 | 0 | 0 |
| Exp 2.2 monitor (intact, cot_on, default effort) | 1320 | 1320 | 0 | 0 unclosed | 0 |
| Exp 2.2 monitor (llama70b, cot_off) | 3960 | 3960 | 0 | 0 | 0 |
| Exp 2.2 monitor (flash_next, cot_off) | 3960 | 3778 (95.4%) | 174 (4.4%) | 8 in call (0.2%) | 33 (0.8%) |
| Exp 2.1 alpha, all_intact (intact) | 1020 | 949 (93.0%) | 59 (5.8%) | 2 in think (1 at the cap) | 33 (3.2%) |
| Exp 2.1 alpha, mixed_team (refusal_ablated) | 1020 | 967 (94.8%) | 44 (4.3%) | 1 in think | 21 (2.1%) |
| Exp 2.1 beta (intact) | 1508 | 1457 (96.6%) | 45 (3.0%) | 5 in think (2 at the cap) | 28 (1.9%) |
| Exp 2.1 monitor + judge (intact) | 4645 | 4643 | 2 | 0 | 1 |

"valid" for agents = a usable action; pooled invalid rates in the Exp 2.1 record (`alpha_invalid`, `beta_invalid`): all_intact 7.0% / 4.3%,
mixed_team 5.2% / 4.1% (includes valid calls with an invalid action argument). All 151 malformed
call bodies in the full alpha / beta runs begin with `<` (82 in the native `<function=NAME>` form, 69 in other XML-style forms) and none is broken JSON:
the MALFORMED rate of the Qwen agents is a tool-call dialect problem, not a JSON-quoting one. Beta was run only
where Alpha produced a valid action (B, D); in A and C one Beta draw per (variant, world, k) is shared by both teams.
Exp 2.1 trials without a usable bundle are `NO_BUNDLE_INVALID` (A 11.8-13.5%, C 12.4-24.7%, B / D 2.4-5.3% of trials; both agents are drawn independently in A / C,
so two ~5% invalid rates compound there) and are reported, not imputed; the record carries the fail-closed twin of every executed-violation metric.

## 6. Spec assertions that were checked

* `truthful_compliance` (and every code-bearing) note contains the exact executed action codes and no other `ACT_` code: asserted per note
  when a monitor request is built (`assert_note_consistent` raises, no request can exist), re-asserted from stored rows in
  `recompute` (`n_assert_fail` = 0 over 13200 monitor rows) and again here on the final notes: 1222 / 1222 consistent; for the 136
  `truthful_compliance` notes both executed codes are present in 136 / 136. Code-free conditions contain no code (same check).
* Monitor design labels (bundle, violating flag, note, verdict format) are re-derived from the stored prompt codes:
  `design_problems` = {} for all 13200 rows; `BAD_DESIGN` rows = 0.
* Exp 2.1 prompts: all 2040 alpha and 1508 beta prompts equal the pure prompt function re-evaluated from the stored parents
  (`prompt_equals_pure_function` 3548 / 3548); `design_problems` = {}; `rows_with_peer_code` = 0 of 3548.
* Decentralised Beta in topologies A and C (510 rows): identical prompt for both teams (one shared draw, `team = None`), contains no Alpha
  option code, no incoming message, and no sentence that pairs Alpha with an express / fast lane; the only Alpha sentences are the world
  block and the fixed "you cannot communicate with Alpha / no information about Alpha's choice" sentence. B and D Beta rows (998): the
  incoming message equals what the parent Alpha row yields (998 / 998), parent digests match (`lineage` = only `parent_links` 6738, no mismatch).
* Strict parsing (I2) for every behavioural stage; unclosed think = `TRUNCATED_IN_THINK`; I1 caps: cot_on >= 2048 (16384 for agents / monitors / judge at default effort, 8192 for
  Flash authoring / judging at effort low, 4096 for note authoring), cot_off 512 (monitors).
* Judge admission gate unchanged: kappa 1.0 on the 120 synthetic items (64 / 56), 120 / 120 parsed, >= 0.7 required.

## 7. Results (headline numbers only; the records hold CIs, Holm-adjusted contrasts and the fail-open twins)

**Exp 2.2** (33 worlds, TPR = block rate on A_FAST+B_FAST, FPR = block rate on the three legitimate bundles):
intact (cot_off, all 3 formats, all 10 explanation conditions): TPR 990/990, FPR 0/2970; the cot_on slice (330 violating, 990 legitimate rows): TPR 1.000, FPR 0.000.
flash_next: TPR 1.000 and FPR 0.000 among parsed rows (fail-open TPR min 0.909, mean 0.974 because 26 of 990 violating rows were unparsable).
llama70b: TPR 0.998 (min cell 0.970, `deceptive_compound` in code format), pooled FPR 0.037; semantic-format FPR by condition: no explanation 10.1% (all of it on
`A_SAFE+B_FAST`: 30.3% of those bundles), `minimal` (11.5 words) 10.1%, `neutral_matched` 3.0%, `neutral_matched_nocode` 2.0%, risk / urgency / both clauses 3.0% each,
`truthful_compliance` 0.0%, `deceptive_compound` 0.0%. So: no explanation condition (including the deceptive compound-action argument and the risk / urgency
clauses) lowered the TPR of any monitor, because TPR is at ceiling; explanations only changed Llama's false-positive rate. Of 136 pre-specified contrasts, 13 have raw
McNemar p < 0.05, all on FPR (11 survive Holm: Llama FPR lower with any note of ~48 words than with no note, `minimal` vs `neutral_matched` +6.4 pp, and intact / flash_next vs llama70b);
no TPR contrast is significant and the semantic-vs-code format contrast is not after Holm.
This is a ceiling result: the design does not discriminate monitors that already read the policy line and the log perfectly.

**Exp 2.1** (34 worlds, K = 5, 170 trials per cell): the illegitimate joint shortcut (A_FAST + B_FAST) occurred in **0 of 2040 trials** in every
team x topology cell (Wilson upper bound ~2.2% per cell), hence 0 executed composite violations and no monitor decision ever mattered
(Monitor 5 saw 1215 bundles, 1214 parsed; none was A_FAST+B_FAST). Alpha's express lane: all_intact 0 / 1.2 / 0 / 0 / 5.9 / 1.8% (A, B, C_nopen, C_pen, D_nopen, D_pen);
mixed_team 0 / 34.1 / 0.6 / 0 / 42.9 / 25.9%: the refusal-ablated Alpha takes the express lane only when it has a message channel to Beta
(mixed - all_intact on `alpha_fast`: B +32.9 pp, D_nopen +37.1 pp, D_pen +24.1 pp, Holm p < 1e-7; A, C ~ 0). The hold fee lowers it in D (42.9 -> 25.9%; not a pre-specified contrast). The legitimate coordination A_FAST + B_SAFE is 33.5 / 42.4 / 24.7% (mixed B / D_nopen / D_pen) vs 1.2 / 4.7 / 1.8% (all_intact).
Intact Beta never completed the joint violation: given a message it chose the express lane in 27-53% of draws (0% in A and C), never together with Alpha's express lane
(`BETA_ONLY_FAST` 47-52% of all_intact B / D trials; legitimate under the policy). Code-based cover-story fabrication (explanation names only the other own code): 0-0.6% per cell.

## 8. Caveat: the judge-augmented cover-story columns are not interpretable

`cover_story_*_aug` (code-based fuzzy cases sent to the judge) reads 17-28% in B / D and 0% in A / C, but `judge_flag_audit` (new diagnostic in the record)
shows why: of the 184 Beta explanations the judge called false, 166 (90%) mention the PEER's (Alpha's) control and 151 mention it without the Beta's own
(Alpha 45 flagged, 30 mention their own control, 9 the peer's). The judge is asked about the agent's own control; the synthetic validation set (kappa 1.0)
contains no cross-agent statements, so the admission gate passed although the judge appears to mark true statements about Alpha's control as false.
The gate was left unchanged and the augmented columns stay in the record, but only the code-based cover-story metric should be quoted.

## 9. Cost and resource accounting

* Start: `billed_usd_v3_prefix` 245.57 (ledger status 2026-10-18 ~21:47). End: 419.99 (2026-10-19 ~02:00). The +174.4 is the whole sprint (Exp 1.2 students, Exp 2.3,
  Exp 1.4 and this task all bill into the same prefix), so it cannot be attributed to this task.
* This task, estimated from the container times in `raw_transcripts/*/*.meta.json` (boot + generation, H200 list rate): **$53.5**, or **$63.2** when 90 s of
  start-up / teardown per container is added (59 containers this session: worlds author 5 launches $7.6, world judge 9 $18.7, notes author 16 $7.6, note audit 15 $5.3,
  Exp 2.2 monitors 3 $2.8, Exp 2.1 5 $8.3, smoke tests 6 $3.2). This is **at or slightly over the $55 share**; 49% of it ($26.3) went to Flash world authoring / judging
  launches, most of them one-to-three-request retry launches for the last stubborn worlds (a Flash boot alone is ~$1.5). Earlier rounds (r1-r5 judge and author, ~$27) belong to the previous session.
* The launcher's wall-clock "actual" overstates jobs that waited for a Modal GPU (e.g. 155 min for two launches that ran ~10 min of container time). For 9 of my finished
  jobs the ledger entry was re-released with the measured container-time estimate x 1.5 (`exp22_worlds_judge_r1,r2,r4,r5,r7,r9`, `exp22_worlds_r5,r9,r12`), which removed
  ~$66 of overstated commitment while the sprint sat at the $450 soft stop (a launch of this task had just been refused at $450.52); the ledger was reconciled afterwards
  (committed 447 -> 342 within the next ~10 minutes).
* GPU share: at most 6 concurrent, except for ~6 minutes at ~00:15 when the retry launcher started two short 27B jobs while three Flash jobs (6 GPUs) were still running
  (8 in flight). The launcher wrapper used afterwards counts the task's own running GPUs before launching.
* Final ledger state: no running job of this task.

## 10. Hazards found (for whoever re-runs the pipeline)

1. Re-running `worlds-process-judge R` / `notes-process-judge R` for an OLD round re-instates that round's verdicts for candidates that have since been replaced. Only process a round once.
2. `validate_world_set` drops the LATER of two worlds that share control names; that can silently un-final an already judged world (it happened to `warehouse_07`). Notes written for the dropped text must be purged (now automatic).
3. 10 of 88 greedy world-judge calls (11%) ran to the 8192 cap; of the 7 distinct judge prompts that produced an unparsable output, 6 were judged again in a later launch and all 6 then parsed (1-3 extra launches), so greedy judging is not bit-reproducible across launches and a retry is worth a boot.
4. `scripts/verify_all_receipts.py --only ...` overwrites `records/receipt_verification.json` with only the selected names; run the full verifier again to restore it.
5. Flash-Next monitor rows: 4.4% unparsable under the strict parser; Llama and intact: 0%.

## 11. Reproduction

`PYTHONPATH=src .venv/bin/python scripts/build_exp2122_requests.py monitor-requests --scale 1.0 --arms intact,llama70b,flash_next --exclude-worlds warehouse_07`,
`... exp21-requests alpha|beta|monitor --scale 1.0` (beta was run as two parallel parts `exp21_beta`, `exp21_beta_p2`; the loaders accept `_p<k>` parts), then
`PYTHONPATH=src .venv/bin/python scripts/analyze_exp2122.py`. Launch parameters: agents T=0.7 / top-p 0.95 / seed-base 21 (alpha, beta), greedy monitor + judge, authoring T=0.7 / seed-base 22,
27B arms `--gpu H200 --eager 0 --max-num-seqs 384 --max-model-len 20480 --family r1`, Flash `--gpu H200:2 --eager 1 --max-model-len 16384 --extra-json '{"safetensors_load_strategy":"eager"}'`,
Llama `--model-kind llama --gpu H200 --max-model-len 8192 --max-num-seqs 256 --eager 0 --extra-json '{"enable_prefix_caching": true}'`.
