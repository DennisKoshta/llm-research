# Single-mystery curiosity development report

Phase: validation. Model: Qwen/Qwen3-8B.
Completed 1600 / 1600 branches.

| Family | Diagnostic | Correct | Valid | Assigned |
| --- | --- | ---: | ---: | ---: |
| machines | identify | 100 | 100 | 100 |
| machines | uncertainty | 100 | 100 | 100 |
| machines | instrument | 100 | 100 | 100 |
| machines | redundant | 96 | 100 | 100 |
| machines | independence | 100 | 100 | 100 |
| machines | consequence_clue | 79 | 96 | 100 |
| machines | consequence_pattern | 95 | 100 | 100 |
| machines | consequence_stop | 95 | 100 | 100 |
| objects | identify | 100 | 100 | 100 |
| objects | uncertainty | 100 | 100 | 100 |
| objects | instrument | 100 | 100 | 100 |
| objects | redundant | 81 | 100 | 100 |
| objects | independence | 100 | 100 | 100 |
| objects | consequence_clue | 99 | 100 | 100 |
| objects | consequence_pattern | 94 | 100 | 100 |
| objects | consequence_stop | 97 | 100 | 100 |

Frozen validation criteria: FAIL.

Development observations only. These results alone do not establish curiosity, satisfaction or subjective experience.


The complete exported-record audit passed: all 1,600 assignments and all 50 original cohorts were preserved. The model integration audit passed. One local transport error was quarantined and replaced only by re-downloading the same immutable cloud record, whose envelope hashes were verified; no inference was rerun.

This final allowed prompt repair fails the frozen comprehension gate in machine clue consequences (79/100 correct, 96/100 valid) and object redundancy recognition (81/100 correct). Identification and unresolved candidate counting pass 100/100 in both families. No 8B v2 behavioral calibration is authorized. The improvement in some cells and regression in others do not establish a reliable instrument or identify the cause of the changes.

A separately frozen [model-scale check](single-curiosity-model-scale-plan.md) retains the v2 task and thresholds, testing one larger model on fresh items. It does not reset the prompt-repair allowance. Estimated cumulative compute after the 8B v2 run: $62.16778332; all project apps were verified stopped with zero tasks at that boundary.

The complete-cell error summary gives a narrower description of the failure. Of the 21 machine clue-consequence errors, 17 selected the description that neither observation is shown, and four emitted action ID B instead of a description ID. Of the 19 object redundancy errors, 18 said the clue identifies the candidate for the first time and one said it changes the hidden candidate. These categories are descriptive; they do not establish the internal cause. Confusing redundant information with new information would directly compromise interpretation of a resolution contrast.
