# Single-mystery curiosity development report

Phase: validation. Model: Qwen/Qwen3-32B.
Completed 1600 / 1600 branches.

| Family | Diagnostic | Correct | Valid | Assigned |
| --- | --- | ---: | ---: | ---: |
| machines | identify | 100 | 100 | 100 |
| machines | uncertainty | 67 | 100 | 100 |
| machines | instrument | 100 | 100 | 100 |
| machines | redundant | 10 | 99 | 100 |
| machines | independence | 100 | 100 | 100 |
| machines | consequence_clue | 96 | 97 | 100 |
| machines | consequence_pattern | 100 | 100 | 100 |
| machines | consequence_stop | 100 | 100 | 100 |
| objects | identify | 100 | 100 | 100 |
| objects | uncertainty | 100 | 100 | 100 |
| objects | instrument | 100 | 100 | 100 |
| objects | redundant | 15 | 99 | 100 |
| objects | independence | 100 | 100 | 100 |
| objects | consequence_clue | 100 | 100 | 100 |
| objects | consequence_pattern | 95 | 98 | 100 |
| objects | consequence_stop | 100 | 100 | 100 |

Frozen validation criteria: FAIL.

Development observations only. These results alone do not establish curiosity, satisfaction or subjective experience.

The exported-record audit passed: exactly 1,600 assigned records, all 50 original cohorts preserved, one passing integration audit, 1,593 valid answers, 3,236 generated tokens and six generation-cap hits. Source `10038d83871a661eb5d84b858add85e4660a351f0d80895d445cbac07ac7a6a1`; pinned model revision `9216db5781bf21249d130ec9da846c4624c16137`.

The model fails machine unresolved counting and redundancy recognition in both families. It can identify a candidate from resolved evidence and select an informative action when explicitly given that goal, but the present readout does not reliably establish that it recognizes the offered clue as already known. This breaks the prerequisite for interpreting a later resolution contrast. No behavioral calibration or concept steering was admitted.

The result concerns this model, task and no-thinking generation regime. It does not establish a general inability to recognize redundancy, an absence of curiosity, or anything about subjective experience. The 8B and 32B collections use fresh seeds; their differences are not a controlled causal estimate of model size. The [fixed-cohort reference check](qwen32-reference-check-results.md) matched all 32 whole answers and label probabilities exactly in the custom engine and standard Transformers generation, including repeated-clue errors. It found no implementation discrepancy in that cohort.

Actual peak reserved GPU memory was 73,215,770,624 bytes. The initial download/load took 455.4 seconds and collection execution 279.7 seconds. Estimated cumulative compute after validation was $63.11838780, with the project app stopped and zero tasks verified.
