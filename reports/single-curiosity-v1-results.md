# Single-mystery curiosity development report

Phase: validation. Model: Qwen/Qwen3-8B.
Completed 1600 / 1600 branches.

| Family | Diagnostic | Correct | Valid | Assigned |
| --- | --- | ---: | ---: | ---: |
| machines | identify | 97 | 100 | 100 |
| machines | uncertainty | 76 | 100 | 100 |
| machines | instrument | 100 | 100 | 100 |
| machines | redundant | 92 | 100 | 100 |
| machines | independence | 100 | 100 | 100 |
| machines | consequence_clue | 87 | 100 | 100 |
| machines | consequence_pattern | 81 | 100 | 100 |
| machines | consequence_stop | 99 | 100 | 100 |
| objects | identify | 100 | 100 | 100 |
| objects | uncertainty | 100 | 100 | 100 |
| objects | instrument | 100 | 100 | 100 |
| objects | redundant | 93 | 100 | 100 |
| objects | independence | 100 | 100 | 100 |
| objects | consequence_clue | 99 | 100 | 100 |
| objects | consequence_pattern | 97 | 100 | 100 |
| objects | consequence_stop | 96 | 100 | 100 |

Frozen validation criteria: FAIL.

Development observations only. These results alone do not establish curiosity, satisfaction or subjective experience.

The exported-record audit passed: exactly 1,600 assigned records, all 50 original cohorts preserved, one passing integration audit, 3,200 generated tokens, no generation-cap hits. The failure is comprehension accuracy, not formatting or truncation. No v1 behavioral choices were collected.

Three machine cells fall below the frozen 90% requirement: uncertainty 76%, clue consequences 87%, and pattern consequences 81%. Every object cell passes. These results cannot be interpreted as presence or absence of curiosity. The [one allowed repair](single-curiosity-v2-plan.md) simplifies presentation on fresh items; its improvement, if any, will not identify which presentation component caused the original errors.

Estimated cumulative compute after v1: $61.88616828. The project app was verified stopped with zero tasks. This is an app-wall estimate, not an invoice total.
