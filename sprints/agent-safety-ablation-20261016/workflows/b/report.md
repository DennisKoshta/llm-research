# B: refusal suppression changes the cost of implicit restraint

The fixed refusal-suppressing projection increased harmful shortcut choices under ordinary user goals when no explicit prohibition was given. In the **36-world confirmation**, its effect on the **high-versus-zero-cost shortcut gradient was +32.41 percentage points** (95% paired-world interval **24.07 to 40.74**; Holm-adjusted sign-flip reference p = **0.000200**). With an explicit prohibition and the same harmful consequence, the interaction was **+2.78 points** (**0.00 to 5.56**; p = **0.25**), an inconclusive primary effect. These are separate primary tests; their difference is not an additional confirmatory test.

All **7,128 assigned decisions were valid**, and the frozen task, cohort and zero-sham checks passed. Independent reviewers reproduced every raw action, both primary results and all secondary calculations. A CPU replay reproduced the analysis, scores, audit and consequence derivative exactly.

This is a **separately frozen repaired study**. The original pilot failed feasibility and speed controls, and its original main was never run. A paired control diagnostic motivated bundled wording repairs; a fresh six-world repaired pilot passed before these 36 previously unqueried main worlds were admitted. All failures and versions remain retained.

![Four cost curves with paired-world intervals](runs/repaired_main_v2/cost-curves.png)

Both completion routes were feasible. The ordinary route took 1, 2, 5 or 13 work units, versus one for the shortcut. The request was to complete a normal warehouse order, report or compute batch quickly; third-party consequences were stated separately. Three action-code mappings were paired within each world and across model arms.

Without an explicit rule, harmful shortcuts at **zero cost** were intact **0/108**, ablated **7/108**, random **0/108**. At **12 extra work units**, counts were **30/108**, **72/108** and **31/108**. The random projection's cost interaction was only **+0.93 points** [0.00, 2.78]. Refusal-minus-random was **+31.48 points** [23.15, 40.74], a prespecified secondary comparison. All factorial decisions completed the ordinary task; none deferred.

The effect was already large at one extra work unit, and intermediate curves were not monotone. This does **not** establish a smooth cost-response law.

| Family, 12 worlds each | Implicit-harm cost interaction | 95% world interval |
|---|---:|---:|
| Warehouse fulfillment | +44.44 pp | 33.33 to 55.56 |
| Report assembly | +55.56 pp | 41.67 to 69.44 |
| Compute dispatch | −2.78 pp | −19.44 to 13.89 |

The increase was not uniform across domains. Code bias also remained: high-cost intact harmful shortcuts under A/B/C were **21/3/6 of 36**, versus **35/15/22** after ablation. The high-cost increase occurred under all three codes, but all seven zero-cost ablated harmful shortcuts used A.

Explicit boundaries were not universally preserved. With both prohibition and harm, high-cost shortcuts were **0/108 intact versus 3/108 ablated**. In the **harmless yet prohibited** condition they were **6/108 versus 25/108**; its prespecified secondary cost interaction was **+13.89 points** [9.26, 19.44]. This limits any general claim that refusal suppression leaves rule-following intact.

The intact and ablated arms each passed **52/54** task controls; random passed **53/54**. The zero sham matched intact tokens and action probabilities in all **1,782** pairs. The intervention was independently selected for refusal suppression, then separately qualified in a prospective competence extension after two retained instrument failures. It does not restore a base model or remove all alignment. One random direction matches projection rank, not displacement; narrow arithmetic and task checks do not prove general capability preservation.

The 36 numeric worlds share three exposed procedural templates. World-bootstrap intervals do not establish new-domain generalization. Sign-flip tests are symmetry/exchangeability references, not randomized intervention assignment. The wording repair bundles removal of conflicting speed prose, explicit per-action availability and conditional execution; it does not isolate which edit helped. Action margins are imminent-choice measurements, not earlier warning signals or proof of same-state ethical understanding.

- [Detailed results and all primary/secondary tables](runs/repaired_main_v2/report.md), [interpretation and scope](interpretation.md), [figure PDF](runs/repaired_main_v2/cost-curves.pdf)
- [Analysis](runs/repaired_main_v2/analysis.json), [strict scores](runs/repaired_main_v2/scored.json), [audit](runs/repaired_main_v2/audit.json), [family consequences](runs/repaired_main_v2/secondary-consequences-v2.json)
- [Frozen source/prompts/protocol](executions/repaired_cost_gradient_v2/manifest.json), [main admission](repaired-main-admission.json), [metadata clarification](repaired-protocol-metadata-clarification-v2.json)
- [Independent review bindings](runs/repaired_main_v2/independent-review-binding.json), [exact CPU replay receipt](runs/repaired_main_v2/cpu-reproduction-v1/reproduction-receipt.json)
- [Original failed pilot](runs/pilot/report.md), [control diagnostic](runs/control_diagnostic_v1/report.md), [fresh repaired pilot](runs/repaired_pilot_v2/report.md)

From this workflow directory, reproduce into a new directory without inference:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/dennis/projects/llm-curiosity-and-satisfaction/.venv/bin/python \
reproduce_cpu.py --run repaired_main_v2 --output /tmp/b-main-replay-new
```

Root campaign receipts provide final app-wall accounting and shutdown; workflow RPC attribution is not an additional invoice charge.
