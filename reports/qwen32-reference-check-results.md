# Qwen32 generation reference check

The fixed 32-row cohort produced identical whole answers in the custom batch runner and the pinned Transformers generation API. The largest first-token label-probability difference was exactly 0.0. No hooks remained.

Both methods used greedy decoding for this engineering comparison. The original 1,600-response scientific collection used sampled decoding and retains every original outcome; this check cannot substitute for it or override its failed gate.

| Family | Diagnostic | Reference correct | Checked |
| --- | --- | ---: | ---: |
| machines | consequence_clue | 2 | 2 |
| machines | consequence_pattern | 1 | 1 |
| machines | consequence_stop | 2 | 2 |
| machines | identify | 1 | 1 |
| machines | independence | 3 | 3 |
| machines | instrument | 3 | 3 |
| machines | redundant | 1 | 5 |
| objects | consequence_clue | 2 | 2 |
| objects | consequence_stop | 3 | 3 |
| objects | identify | 4 | 4 |
| objects | independence | 3 | 3 |
| objects | redundant | 0 | 1 |
| objects | uncertainty | 2 | 2 |

The initial reference deployment failed to import a local helper before inference and was stopped. A self-contained entrypoint then ran the same frozen cohort successfully. Both app lifetimes are included in conservative accounting. The result rules out a custom/reference discrepancy in this checked cohort; it is not proof that all implementation errors are impossible.
