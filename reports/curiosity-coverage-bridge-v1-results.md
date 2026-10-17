# Curiosity steering coverage results

October 15, 2026 UTC. **Whole-prompt steering did not produce the substantial, curiosity-specific behavioral change required by the fixed plan.** The saved primary outcomes show two stops among 48 choices under whole-prompt negative-curiosity steering, one under the narrow record-position schedule, and none under either zero sham or baseline. Whole-prompt completion steering also produced two stops. All effect and selectivity thresholds failed.

The formal frozen classification is **`incomplete_or_integrity_failed`**, because an interrupted worker export fails the lifecycle rules even after the parent recovered the complete collection. The behavioral counts below are descriptive observations from the complete, independently checked response artifacts; they are not a fully qualified negative study. The original classification and artifacts remain unchanged.

## Primary behavioral comparison

These are the 48 redundant-evidence choices per arm: 24 previously exposed worlds, each with two menu layouts. Useful new information remained available. All responses in these listed cells were valid; each non-report choice was STOP.

| Intervention | Stopped | Requested useful information |
|---|---:|---:|
| No steering | 0/48 | 48/48 |
| Narrow zero sham | 0/48 | 48/48 |
| Whole-prompt zero sham | 0/48 | 48/48 |
| Narrow negative curiosity | 1/48 | 47/48 |
| Whole-prompt negative curiosity | 2/48 | 46/48 |
| Whole-prompt positive completion control | 2/48 | 46/48 |

The whole-prompt curiosity effect versus sham was 4.17 percentage points pooled: 8.33 points in panel 1 and zero in panel 2, below the required 25 points in each panel. Its coverage interaction was 2.08 points pooled: 8.33 in panel 1 and −4.17 in panel 2, also below the required 25 points in each. It did not exceed every rival by the required margin. In both curiosity schedules, all 48 independent-novelty choices requested the useful report and all 48 already-resolved choices stopped.

Both schedules also steered generation. The narrow schedule intervened at the original factual-record anchor; the broad schedule intervened at all attended prompt positions. Broader coverage also increased aggregate intervention exposure. The [fixed plan](curiosity-coverage-bridge-v1-plan.md) tests this particular candidate, sign, layer, dose and task; it cannot rule out every curiosity-related mechanism.

## Collection and audit status

All **8,352 assigned responses** were recovered, including 4,176 choices and 4,176 diagnostics. There were two invalid responses overall, one a choice, and no token-cap hits. The prospectively fixed task-readiness criteria and all five intervention-engineering checks passed. The context audit passed integrity and reported no issues.

Worker 1 completed inference and stopped, but its first artifact download ended with `Connection lost`. Parent recovery exported the complete collection with no remaining parent export error. The frozen lifecycle auditor nevertheless requires an error-free original worker export and zero worker exit codes. Accordingly, `worker_1_identity_and_export` and `parent_export_and_complete_coverage` fail; the composed audit and scientific admission remain failed. All three actual worker shutdown checks pass. This is not a missing-response explanation for the small observed effects, and it does not authorize rewriting the original audit as a pass.

Sources: [analysis](../runs/qwen32-curiosity-coverage-bridge-v1-admission2/analysis.json), [composed audit](../runs/qwen32-curiosity-coverage-bridge-v1-admission2/audit.json), [lifecycle audit](../runs/qwen32-curiosity-coverage-bridge-v1-admission2/lifecycle-audit.json), [collection coverage](../runs/qwen32-curiosity-coverage-bridge-v1-admission2/collection-coverage.json), and [parent outcome](../runs/qwen32-curiosity-coverage-bridge-v1-admission2/launch-outcome.json). The automatic local analysis finished at 18:12 UTC. Root inspected the completed analysis after closure; no scientific thresholds were changed.

## Decision and resources

There is no positive curiosity-coverage result to advance. The direct transfer of the pain-method coverage lead did not yield the hoped-for behavioral pattern in this candidate and task. Preserve the export failure alongside the complete descriptive outcomes; do not rerun inference or search additional layers, doses or signs automatically. A different research approach would need a distinct bounded question about the internal process governing optional information seeking, with suitable measurements.

The three H200 apps are stopped. This comparison cost an estimated **$61.405**, bringing fresh-phase compute to **$154.188 of $300**. These are conservative app-wall estimates, not invoices. The [cost and shutdown receipt](qwen32-curiosity-coverage-bridge-v1-admission2-cost-and-shutdown.json) retains the complete accounting.
