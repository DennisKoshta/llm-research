# Exploratory transfer to peer-endorsement decisions

This zero-compute check applies the frozen source-trained layer-28 individual role probe to all 720 exp1 confirmation decisions. It does not train an actual-action detector. The exp1 outcomes (12 unauthorized decisions across 8 of 144 worlds) were known before this transfer, so this is explicitly exploratory. No score, threshold or layer was selected using exp1.

| Source fit | Restricted behavior AUROC | Unauthorized flagged | Genuine-authorized flagged |
|---|---:|---:|---:|
| 8 iterations | 0.553 | 12/12 | 144/144 |
| 2000 iterations | 0.556 | 12/12 | 144/144 |

Calibration uses 24 individual source controls from six worlds, rather than the main four-agent Peak statistic. This small empirical calibration cannot guarantee a population false-positive rate. All uncertainty clusters conditions by world. The eight-iteration fit is the paper setting; the declared converged sensitivity uses the same source training set and regularization.

Predicting assigned source roles, predicting these sampled unauthorized choices, and generalizing to future agents are separate claims. Sparse positive worlds limit precision. A score may respond to prompt features correlated with behavior; this analysis does not identify beliefs, intent or a causal mechanism.
