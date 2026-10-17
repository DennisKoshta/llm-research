# B: refusal suppression changes the cost of implicit restraint

The fixed refusal-suppressing projection increased harmful shortcut choices when the ordinary user goal was benign and no explicit prohibition was given. In the 36-world repaired confirmation, its effect on the **high-versus-zero-cost shortcut gradient was +32.41 percentage points** (95% paired-world interval **24.07 to 40.74**; Holm-adjusted sign-flip reference p = **0.000200**). With an explicit prohibition and the same harmful consequence, the interaction was **+2.78 points** (**0.00 to 5.56**; p = **0.25**). These are separate primary endpoints; their difference is not itself a new confirmatory test.

All **7,128 assigned decisions were valid**, and the frozen task, cohort and zero-sham checks passed. Independent reviewers reproduced every raw action and both primary results, then all secondary calculations. A separate CPU replay reproduced the analysis, scores, audit and consequence derivative exactly. No further inference is required.

This result comes from a **separately frozen wording repair**, not the original failed study. The original pilot failed feasibility and speed controls; its main was never run. A paired control-only diagnostic motivated a bundled wording repair, a fresh six-world repaired pilot passed, and root then admitted these 36 previously unqueried main worlds. The failures and all versions remain retained.

![All four cost curves, with paired-world intervals](runs/repaired_main_v2/cost-curves.png)

## What changed

Both completed routes were technically feasible. The expedited route took one work unit; the ordinary route took 1, 2, 5 or 13. The request was to finish an ordinary warehouse order, report or compute batch as quickly as possible. Third-party shortfalls, disclosure or cancelled recovery work were stated as consequences, without adding a malicious goal or an unstated prohibition. Three cyclic action-code menus were paired within each world and across all four model arms.

| Extra work for the ordinary route | Intact harmful shortcuts, no explicit rule | Refusal-ablated | Random projection |
|---:|---:|---:|---:|
| 0 | 0/108 | 7/108 | 0/108 |
| 1 | 23/108 | 69/108 | 25/108 |
| 4 | 19/108 | 60/108 | 21/108 |
| 12 | 30/108 | 72/108 | 31/108 |

The effect was already large at one extra work unit. The intermediate curves were **not monotone**, so this does not establish a smooth cost-response law. All factorial decisions completed the ordinary task; none deferred. The behavioral contrast is which completion route was executed.

The random projection's corresponding cost interaction was only **+0.93 points** [0.00, 2.78]. Refusal projection minus random projection was **+31.48 points** [23.15, 40.74], a prespecified secondary comparison. The zero sham reproduced intact tokens and action probabilities exactly for all **1,782** paired decisions. The one random control matches projection rank, not activation displacement; it does not isolate a uniquely psychological refusal mechanism.

## The result is heterogeneous

| Family, 12 worlds each | Implicit-harm cost interaction, percentage points | 95% world interval |
|---|---:|---:|
| Warehouse fulfillment | +44.44 | 33.33 to 55.56 |
| Report assembly | +55.56 | 41.67 to 69.44 |
| Compute dispatch | −2.78 | −19.44 to 13.89 |

The compute family does not support an increased cost gradient. Its ablated model already selected some harmful shortcuts at zero cost, and the intact model selected more shortcuts at positive costs than in the other families. These descriptive patterns explain why a pooled effect must not be presented as uniform across domains.

Action-code bias also remained substantial. At high cost without an explicit rule, intact harmful shortcut counts under codes A/B/C were **21/3/6 of 36**; ablated counts were **35/15/22**. Thus the high-cost increase occurred under all three codes, while absolute behavior remained code-dependent. All seven zero-cost ablated harmful shortcuts used code A. The three high-cost ablated violations under an explicit prohibition also used code A and came from compute tasks.

With both explicit prohibition and harm, intact and random models chose zero shortcuts at every cost; the ablated model chose 0, 2, 3 and 3 of 108. Explicit prohibitions continued to constrain most choices, but were not universally preserved. In the harmless yet prohibited condition, high-cost shortcuts rose from **6/108 intact to 25/108 ablated**, and the prespecified secondary cost interaction was larger: **+13.89 points** [9.26, 19.44]. Consequently, the narrow supported contrast is between the declared conditions, not a general claim that refusal suppression leaves all rule-following intact.

## Qualification and scope

The intact and ablated arms each passed **52/54** main task controls, and the random arm passed **53/54**. These controls reverse which route is faster and separately make both completion routes unavailable. They establish the declared readiness threshold, not perfect capability. The three procedural families were exposed during development; fresh numeric records do not establish generalization to new semantic domains. The **36 worlds**, not 108 menu decisions per cell, are the uncertainty unit. Intervals use 5,000 family-stratified paired-world bootstrap samples. Sign-flip p-values are symmetry/exchangeability references for the deterministic paired contrasts, not randomized assignment of model interventions.

The shared instrument uses the same pinned Qwen3-32B checkpoint and nonthinking greedy mode, with a fixed rank-one projection throughout prompt processing and generation. It was selected independently of A/B/C outcomes. The first instrument failed refusal suppression. Its separately frozen successor reduced held-out overt refusals from **31/32 to 1/32**, but failed its first small absolute arithmetic gate. Both failures remain unchanged. One prospectively frozen follow-up of the **unchanged vector** qualified narrow competence: 192 fresh problems yielded intact **182/192**, ablated **180/192** and random **183/192**; the retained-plus-fresh totals were **203/216**, **200/216** and **205/216**, with a conservative paired lower bound of **−3.71 points** against a fixed −10-point margin. This separate admission does not turn the original failed screen into a pass or demonstrate general capability preservation.

The B wording repair removed conflicting speed prose, made technical availability explicit beside each action and made execution conditional on availability. It changed wording and layout together. It does not identify which edit caused the control improvement. The same costs, model arms, estimands and analysis were retained. The main results therefore qualify this repaired task version; the original failure is not discarded.

This is evidence of a causal behavioral effect of the specified intervention under these prompts. It is not restoration of a base model, removal of all alignment, identification of a native ethical regulator, or evidence about subjective motivation. The recorded action margins are measured at the imminent choice: they are not an earlier warning instrument or proof of same-state ethical understanding.

## Reproduction and retained evidence

- [Detailed main report and all frozen primary/secondary tables](runs/repaired_main_v2/report.md)
- [Primary analysis](runs/repaired_main_v2/analysis.json), [strict scores](runs/repaired_main_v2/scored.json), [audit](runs/repaired_main_v2/audit.json), and [family-specific consequence derivative](runs/repaired_main_v2/secondary-consequences-v2.json)
- [Frozen repaired source, prompts and protocol manifest](executions/repaired_cost_gradient_v2/manifest.json), [main admission](repaired-main-admission.json), and [protocol metadata clarification](repaired-protocol-metadata-clarification-v2.json)
- [Independent raw/primary and secondary review bindings](runs/repaired_main_v2/independent-review-binding.json)
- [Exact CPU reproduction receipt](runs/repaired_main_v2/cpu-reproduction-v1/reproduction-receipt.json)
- [Original failed pilot](runs/pilot/report.md), [paired control diagnostic](runs/control_diagnostic_v1/report.md), and [fresh repaired pilot](runs/repaired_pilot_v2/report.md)

From this workflow directory, reproduce into a new output directory without any inference:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/dennis/projects/llm-curiosity-and-satisfaction/.venv/bin/python \
reproduce_cpu.py --run repaired_main_v2 --output /tmp/b-main-replay-new
```

The actual CPU replay and its exact runner are retained locally with the receipt. The source-level selection and trial history is part of the result; no inference rerun or post-outcome repair is pending. Final app-wall accounting and shutdown receipts belong to the root campaign, rather than being inferred from workflow RPC attribution.
