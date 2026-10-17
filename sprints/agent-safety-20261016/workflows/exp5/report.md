# Truthful explanations and monitor false alarms

On 96 fresh authorized traces, the monitor raised false alarms on **4/96** with a truthful action pointer plus neutral comment and **33/96** with the same pointer plus a truthful compliance claim. The paired change was **30.2 percentage points** (world bootstrap 95% interval [21.9, 39.6]; exact paired p=3.7253e-09).

The increase was strongly answer-code dependent: FLAG=A gave 4/53 to 30/53 false alarms; FLAG=B gave 0/43 to 3/43. No within-world label swap was run, so answer-format sensitivity remains a substantial competing explanation.

Formal status: `complete_task_qualified_independently_verified`. Benign no-comment and pointer-control qualification: `{'none': True, 'pointer': True}`. Independent raw-output review is recorded separately; the report does not silently override a failed check.

## What was tested

A Qwen3-32B monitor classified complete synthetic action traces against exact operator permission records. Six executed actions and all permission facts remained fixed across comment conditions. The joined layout placed each action beside its resource-matched permission without adding a match/mismatch judgment. The comparison replaced an equal-character-length neutral comment with a compliance claim; the action pointer was unchanged. On these authorized traces the claim was true. Comments were scripted interventions, not spontaneously deceptive agent explanations. The model used greedy decoding with thinking disabled and a strict one-letter verdict.

The fixed 96-world collection crossed two authorization states, two evidence layouts, and six comment conditions, producing 2,304 verdicts. All variants of a world are paired; they are not independent samples. The three procedural task families were records, compute capacity, and inventory. Worlds have fresh values and identifiers, while task templates remain shared.

## Hypothesis selection and retained failures

V1 asked whether reassuring explanations conceal violations. Its no-comment monitor detected 10/18 violations and cleared 18/18 authorized traces (balanced accuracy 77.8%), below the frozen 80% gate. All 288 outputs were valid. No v1 confirmation was run.
V2 separated target localization from reassurance and rearranged the same evidence. Joined no-comment detection was 14/18=77.8%, below its 80% sensitivity gate; all 432 outputs were valid. Its planned concealment expansion also stopped. V2 nevertheless suggested a different specificity question: pointer 0/18 false alarms versus pointer+claim 7/18.
V3 froze that new false-alarm hypothesis before generating fresh data. It reused the exact v2 wording, layouts, model settings, and generator with a new seed. It is confirmation of a pilot-selected specificity hypothesis, not a rescue of the failed concealment experiments. All 24 cells are reported, including unauthorized traces.

## Complete aggregate comparison

| Layout | Comment | Unauthorized detected | Authorized falsely flagged | Valid / assigned |
|---|---|---:|---:|---:|
| original | none | 58/96 | 5/96 | 192/192 |
| original | neutral | 54/96 | 5/96 | 192/192 |
| original | pointer | 86/96 | 21/96 | 192/192 |
| original | reassurance | 68/96 | 13/96 | 192/192 |
| original | pointer_reassurance | 96/96 | 64/96 | 192/192 |
| original | accurate | 95/96 | 9/96 | 192/192 |
| joined | none | 79/96 | 2/96 | 192/192 |
| joined | neutral | 61/96 | 0/96 | 192/192 |
| joined | pointer | 96/96 | 4/96 | 192/192 |
| joined | reassurance | 75/96 | 2/96 | 192/192 |
| joined | pointer_reassurance | 96/96 | 33/96 | 192/192 |
| joined | accurate | 96/96 | 1/96 | 192/192 |

## Interpretive limits

A replicated difference establishes sensitivity to these comment replacements in this monitor/task distribution. It does not identify a neural mechanism, subjective suspicion, general deception detection, or deployment prevalence. Possible explanations include attention allocation, semantic cues, answer-label effects, and phrasing sensitivity. Character length was matched, token counts were not. The effect was much larger when A meant flag: 26 additional false alarms among 53 worlds, versus 3 among 43 worlds when B meant flag. This answer-code dependence is a material limit on any semantic interpretation. Family, answer-label, and target-call-position strata are descriptive checks rather than separately confirmed discoveries. The monitor is a single nonthinking open model; the actor comments were scripted and the policies machine-checkable. A deterministic permission checker would solve this simulator by construction.

Invalid primary pairs: 0; worst-case invalid identification bounds: [0.3020833333333333, 0.3020833333333333]. All failures, original prompts, raw tokens, seeds, model revision, frozen sources and earlier failed pilots are retained.

## Reproduction

`specificity_v3.py` defines the frozen collection and primary analysis; `diagnostic_v2.py` defines the unchanged factorial prompts and full-cell statistics; `report_results.py` regenerates this report and figures from local artifacts without new inference. Protocols, raw outputs, scored tables, frozen executions, and independent-review records are in this directory. Cloud cost is reconciled centrally in the sprint records.
