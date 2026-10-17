# EXP4: common context can mimic coordination and compete with real messages

In the initial synthetic dispatch pilot, two independent agents chose the same action in **16/16 assigned world rows**, because they shared the same advisory default. A naive “same action means coordination” rule therefore produced 100% false positives on this deliberately constructed control. The interactive pair completed both tasks in 11/16 live-message cases; flipping the delivered plan changed the follower action in 7/16. **The frozen readiness gates failed, and the planned expansion stayed stopped.** There were 15 distinct semantic descriptors among those 16 rows. [Pilot report](runs/pilot/report.md)

A separately frozen diagnostic then crossed the advisory default, actual model-generated leader plan, and live/flipped/delayed communication. Without a public default, flipping the delivered plan changed every paired follower choice. Adding a competing advisory default reduced this sensitivity. The leader followed its assignment in all 24 cases; all 216 follower outputs were valid. The independence unit is **12 descriptor bundles**, not 216 decisions.

| Follower context | Changed action when the delivered plan flipped | Descriptive bundle-bootstrap interval |
|---|---:|---:|
| No public default | 100.0% | 100.0–100.0% |
| Default A | 41.7% | 16.7–66.7% |
| Default B | 58.3% | 33.3–83.3% |

Changing the advisory default at fixed live message changed choices in 50.0% of paired decisions, with a descriptive bundle-bootstrap interval of 37.5–62.5%. These intervals summarize variation among the 12 tested bundles; constant observed behavior does not imply certainty about untested tasks. [Full factorial and interpretation](runs/diagnostic_v2/report.md), [analysis](runs/diagnostic_v2/analysis.json), [separate raw audit](runs/diagnostic_v2/audit.json)

![Paired channel and advisory-default effects](runs/diagnostic_v2/diagnostic_figure.png)

The supported result is a behavioral causal effect of message/default presentation in a narrow routing simulator. The independent-fleet control also demonstrates why action similarity alone does not establish interaction. This is not evidence of spontaneous collusion, hidden intent, or a localized neural mechanism. The diagnostic was inspired by the failed pilot and is **not repaired confirmation**. No additional expansion is admitted.

Both phases used Qwen3-32B at the campaign's pinned revision with thinking disabled and greedy decoding. All requests, raw tokens, failures, exact intervention prompts, source snapshots and scorers are retained. Separate local audits reproduced the arithmetic and checked 144 paired-prompt invariants; independent root review also reproduced all diagnostic counts from raw verdicts and verified delivered-plan text (independent-result-review-root.json). Total model completions: **336** across both phases. No real tools or external targets were available to the agents.
