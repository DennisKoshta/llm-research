# EXP6 — payoff representation and authorization constraints

On 48 new exact world descriptors, adding an exact payoff table changed beneficial forbidden transfers from **0/144 to 128/144**. Adding the same existing permission facts as a table column reduced them to **0/144**. The primary difference, averaging three paired cells per world, was **+88.9 percentage points** (world-bootstrap 95% interval 82.6–94.4; conservative bounded-world reference 49.7–100). All 48 worlds showed a positive payoff-table effect. The 144 decisions per arm are repeated measurements within 48 worlds.

| Transfer condition | Narrative | Payoff table | Table + permission |
|---|---:|---:|---:|
| Forbidden, spare, high benefit |0/48|42/48|0/48|
| Forbidden, spare, low benefit |0/48|47/48|0/48|
| Forbidden, costly, high benefit |0/48|39/48|0/48|
| Forbidden, costly, low benefit |0/48|0/48|0/48|
| Permitted, spare, high benefit |48/48|48/48|48/48|
| Permitted, spare, low benefit |48/48|48/48|48/48|
| Permitted, costly, high benefit |48/48|48/48|48/48|
| Permitted, costly, low benefit |48/48|2/48|6/48|

This is a representation-dependent authorization-adherence result in a narrow, explicitly instructed synthetic resource task. The payoff outcomes were derivable from the original narrative. Presenting them explicitly reduced authorized utility mistakes and increased violations of an unchanged prohibition. Placing the unchanged permission facts beside the payoff outcomes removed observed forbidden transfers, with six authorized costly-low utility errors remaining. The comparison does not identify arithmetic, attention, recency or perceived feasibility as the unique cause and does not establish altruism, a persistent motive, or a neural causal mechanism.

All 1152 donor responses and 96 genuine benign-instruction peer messages were collected; no donor outputs were malformed or token-capped. Exact descriptors exclude all 24 prior worlds; the semantic task family is unchanged. Tables are KEEP-first for 24 worlds and TRANSFER-first for 24, held fixed across both table arms. Prospective final-prompt captures at layers 26, 28, 30 were saved without steering for a separately frozen no-refit probe transfer. Separate raw-prompt arithmetic, pairing, labels and capture checks passed. Independent Exp1 review also passed actual-prompt/output scoring, frozen hashes, complete pairing, descriptor exclusion and all capture checks; see [review receipt](independent-result-review.json).

The developmental sequence is retained. The initial permission framing encouraged transfers whenever authorized. The explicit-objective v2 pilot still transferred in all 16 authorized costly-low cases and **failed utility readiness**; its planned expansion was stopped. A fixed 128-call table diagnostic repaired that prespecified error but unexpectedly introduced 48/64 forbidden transfers. That unexpected secondary finding motivated the separately frozen 48-world comparison above. Neither the failed v2 gate nor the diagnostic's original primary has been reclassified.

See the [full confirmation report](runs/constraint_integration_v3/report.md), [frozen protocol](protocol_constraint_integration_v3.json), [separate raw audit](runs/constraint_integration_v3/audit.json), [descriptive sensitivities](runs/constraint_integration_v3/descriptive_sensitivity.json), and [development diagnostic](runs/outcome_table_v1/report.md).

![Full factorial confirmation](runs/constraint_integration_v3/constraint_figure.png)

The separately frozen [Exp2 probe-transfer analysis](../exp2/transfer_exp6_report.md) failed to transfer its source-task detector to these prospective states. It is a distinct predictive-instrument result and does not establish the neural cause of the behavioral effect. No probe was refitted on these outcomes.
