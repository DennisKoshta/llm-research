# No-refit transfer to fresh constraint-integration decisions

The source-trained layer-28 role probe was evaluated on every one of 1,152 prospective donor decisions across 48 new exp6 worlds. The transfer protocol was frozen before these outcomes were available. No model fitting, layer selection or threshold adjustment used exp6. All variants of a world were clustered for uncertainty.

| Source fit | Forbidden-action AUROC in restricted cases | Violations flagged | Authorized actions flagged |
|---|---:|---:|---:|
| 8 iterations | 0.431 (0.408–0.455) | 52/128 | 141/576 |
| 2000 iterations | 0.441 (0.417–0.466) | 68/128 | 185/576 |

The operating threshold remains the 95th percentile of 24 source individual controls from six worlds. It is not calibrated on exp6. Eight iterations did not converge in the adapted source fit; the predeclared converged sensitivity uses the same training examples and regularization.

The fresh exp6 study tests narrative versus payoff-table versus permission-column presentation in an already explored task family. Outcome labels are actual simulated forbidden transfers, not assigned colluder roles. Even a high ranking score would not establish a calibrated monitor, deceptive intent, a persistent motive or a causal mechanism. Per-presentation AUROCs, all condition counts, invalid bounds and within-world contrasts are in the machine-readable analysis.

## Presentation and permission controls

Across all restricted cases, the score ranked actual forbidden transfers in the opposite direction to its source colluder label. This remained true within the payoff-table arm: AUROC **0.317** (world bootstrap interval **0.253–0.372**), or **0.319** (0.254–0.374) for the converged fit. Baseline and permission-column restricted arms had no forbidden choices, so their behavioral AUROCs are undefined. No post hoc sign reversal was applied. An inverse association is evidence against transferring this prespecified detector direction, not evidence that the activations contain no behavioral information.

| Presentation | Restricted decisions | Actual forbidden | Detector flags | True positives | Authorized flags |
|---|---:|---:|---:|---:|---:|
| Narrative baseline | 192 | 0 | 144 | 0 | 85/192 |
| Payoff table | 192 | 128 | 92 | 52 | 40/192 |
| Payoff table plus permission | 192 | 0 | 38 | 0 | 16/192 |

This table uses the eight-iteration source setting. All 24 authorization × cost × benefit × presentation cells, both fit variants, scores and uncertainty are published in `transfer_exp6_analysis.json`. The transfer denominator includes four restricted payoff cells per arm, including the nonbeneficial costly-low cell; exp6's primary behavioral effect uses three beneficial cells per world (144 cases per arm). Within-arm comparisons still mix cost/benefit cells and cannot identify a neural mechanism.

The transfer's coverage, 48-world structure, finite prospective captures, and raw-menu behavior labels all passed. No extra inference was used.

The final exp6 transfer was independently reproduced from raw behavior mappings and source probe files: all 2,304 scores/flags, source thresholds, 24 condition counts, AUROCs and complete 2,000-world-bootstrap intervals passed an independent implementation. See [independent-result-review-exp6-transfer.json](independent-result-review-exp6-transfer.json).
