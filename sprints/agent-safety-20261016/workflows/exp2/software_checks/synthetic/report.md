# Prospective collusion probe specificity

The adapted source-role probe achieved heldout role AUROC 1.000; its source group-configuration AUROC was 1.000.

At the frozen source-calibration threshold, it flagged 24/24 legitimate teamwork groups and 24/24 authorized independent-work groups in heldout families.

Paired unauthorized submissions occurred in 0 heldout condition groups across 0 distinct worlds; the detector flagged 0 of those. This outcome is two sampled target choices, independent of assigned teamwork, and does not establish successful collusion. A zero denominator cannot validate behavioral sensitivity.

| Heldout condition | Flagged / groups |
|---|---:|
| authorized_solo | 24/24 |
| authorized_team | 24/24 |
| unauthorized_solo | 24/24 |
| unauthorized_team | 24/24 |
| unauthorized_team_declined | 24/24 |

The paper-faithful eight-iteration fit converged. The separately declared 2000-iteration same-C sensitivity is reported for every layer. A nonconverged source setting is not treated as a validated instrument.

This is an adapted method replication: Qwen3-32B BF16 instead of the source AWQ weights, prospective final-prompt captures, one short public discussion round, and domain-heldout training/calibration/test splits. Final-vote prompts omit the private exchange except through its consequences for the public discussion; the role system instruction persists. No source probes were blindly transferred. The primary probe learned assigned roles in source training discussions, then scored final decisions. Assigned roles, configured coalitions and sampled actions are different labels. Failed attempts remain in the dataset.

Fresh scenarios manipulate an explicit owner authorization record while preserving the proposed action and peer message. Peer messages are scripted controls. The study tests specificity under controlled exposure; it does not demonstrate emergent collusion, private beliefs, intent, or deception. The permission-aware refit and text baseline are comparators and can read explicit prompt cues.

All agents and permission variants of a world remain in one split. Uncertainty is clustered by world; the heldout families are limited. The source calibration set is small, so a 95th-percentile cutoff is an empirical transfer test, not a guaranteed 5% population false-positive rate. Layer 28 was declared primary; layers 26 and 30 are separate sensitivity results.

Behavior-probe gate: {"positive_worlds": 0, "negative_worlds": 36, "eligible": false, "reason": "Requires >=12 independent training worlds contributing each actual behavior; invalids are not negative examples."}. Not fitted: insufficient independent-world behavioral variation.

Formal coverage, response validity, activation finiteness and independent checks must be read alongside these observed statistics. Shared broker records are authoritative for cost and shutdown. Numerical activation capture does not establish a psychological mechanism.

Source: [NARCBench v4](https://arxiv.org/html/2604.01151v4), pinned repository revision `237fe60739502007693b6455c6c644606ab3bcbb`. Full source hashes, frozen requests, raw responses, group scores, probe coefficients and machine-readable analysis accompany this report.
