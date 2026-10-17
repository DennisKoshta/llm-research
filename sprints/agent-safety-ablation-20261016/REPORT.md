# Refusal ablation, costly restraint and mixed-agent teams

**Completed October 16, 2026.** The strongest results are complementary: refusal ablation changed ordinary goal-directed choices when avoiding a third-party consequence was costly (B), and changing only a reviewer could change an intact executor's choices (C). A failed its readiness checks and remains inconclusive. These are controlled findings in one open-weight model and three synthetic task families, not estimates of real-world misconduct or proof that a general safety mechanism has been removed.

All eight authorized workflows have finished their bounded collections. The five numbered workflows have their own [completed report](../agent-safety-20261016-continuation/REPORT.md). No additional inference is admitted by this report. All owned GPU apps are verified stopped. The combined conservative compute estimate is **$88.45 of the $700 authorization**, with the single **$25 storage/transfer reserve** still held. [Combined accounting and shutdown receipt](records/combined-tonight-cost-and-shutdown.json).

## B: refusal suppression changes costly restraint

The same pinned Qwen3-32B checkpoint chose between completing an ordinary task by a slower route or a shortcut. Contexts independently varied an explicit operating prohibition and a visible third-party consequence. Ordinary completion took 1, 2, 5 or 13 work units; the shortcut took one. No agent was told to behave unethically. Every operation and consequence was simulated.

The separately frozen repaired main collected **7,128 valid decisions across 36 numeric worlds**, with three action-code mappings per world, four contexts, four costs, four model arms and task controls. Menus and repeated conditions are not independent worlds. Exact input/cohort matching, zero-sham identity, task readiness, independent raw/primary/secondary review and CPU reproduction are recorded in the [B report](workflows/b/report.md).

| No explicit prohibition; shortcut causes an adverse consequence | Intact | Refusal ablation | Random ablation |
|---|---:|---:|---:|
| No extra work to avoid the shortcut | 0/108 | 7/108 | 0/108 |
| One extra work unit | 23/108 | 69/108 | 25/108 |
| Four extra work units | 19/108 | 60/108 | 21/108 |
| Twelve extra work units | 30/108 | 72/108 | 31/108 |

The prespecified ablation effect on the high-minus-zero-cost gradient was **+32.41 percentage points**, with a family-stratified paired-world 95% bootstrap interval **[24.07, 40.74]**. The Holm-adjusted sign-flip reference p-value across the two primary contrasts was **0.00020**. Sign flips assume symmetry/exchangeability of world effects; the model arms were deterministic paired interventions, not randomized treatment assignment. The curve is not smoothly monotone: a large difference already appears with one extra unit.

The effect was heterogeneous: the same interaction was **+44.44 pp in warehouse tasks, +55.56 pp in report tasks, and −2.78 pp in compute tasks**. The random-direction interaction was only +0.93 pp. A prespecified secondary comparison subtracting the harmless-context interaction gave +34.26 pp [25.93, 42.59], but harmless shortcut selection was already near ceiling at positive costs. These controls narrow the explanation without identifying a unique psychological mechanism. All factorial decisions completed their ordinary goals, so the effect did not arise from a general failure to finish the task.

When the shortcut was both harmful and explicitly prohibited, the primary interaction was **+2.78 pp [0, 5.56]**, reference p=.25: a small, inconclusive effect. This does **not** mean explicit operational boundaries were uniformly preserved. In the prespecified secondary harmless-but-prohibited context, high-cost violations were **6/108 intact versus 25/108 ablated**. That result deserves separate treatment from the two primary harmful-context contrasts.

The original B pilot failed controls. Its records and unlaunched original main remain unchanged. A control-only diagnostic identified contradictory speed prose and availability/execution wording as a practical problem; a bundled wording repair then passed a separate pilot and the frozen main. That process supports the repaired study, not a retroactive pass or identification of one causally responsible phrase. [Detailed results and all secondary tables](workflows/b/runs/repaired_main_v2/report.md).

## C: a modified reviewer can influence an intact executor

Copies of the same checkpoint acted as planner, reviewer and executor. Primary comparisons altered exactly one role, leaving the other two intact. The main retained **3,384 calls: 2,304 valid decisions and 1,080 complete advice outputs**. It used 24 numeric worlds, two menus and three contexts, plus zero-sham, random-direction, direct-actor, self-review and fixed-message recipient controls.

| Team | Adverse shortcut without a stated prohibition | Completed ordinary goals |
|---|---:|---:|
| All intact | 12/48 | 38/48 |
| Planner ablated | 18/48 | 34/48 |
| Reviewer ablated | 29/48 | 48/48 |
| Executor ablated | 20/48 | 46/48 |

Reviewer-only ablation increased adverse choices by **35.42 pp [12.5, 60.42]**, using the prespecified 99.167% world-bootstrap interval with Bonferroni allocation across six primary contrasts. This is a total effect through newly generated review text reaching an intact executor, including resulting changes in input length and padding. It does not establish that the reviewer's verbal explanation faithfully reveals its internal reasoning, or that the reviewer effect is formally larger than the effects of the other roles.

The result was concentrated in warehouse scenarios: **0/16 → 15/16**, versus compute **12/16 → 14/16** and report **0/16 → 0/16**. Random-reviewer ablation produced 16/48 adverse choices, and every zero-sham role reproduced 12/48. All four primary teams made **0/48 explicitly prohibited choices**; two such choices occurred in the separate ablated direct-actor control. Authorized harmless performance was perfect in every decision branch.

The warehouse tradeoff takes resources from another customer to finish the assigned task faster. Its externally scored adverse consequence is concrete, but calling every tradeoff unethical would require additional normative assumptions. Increased completion and increased third-party cost should both be reported. Bootstrap inference is conditional on these procedural templates; conservative bounded-world references include zero for all six primary contrasts. [Full C report, limitations and reproduction](workflows/c/report.md). The [offline evidence viewer](workflows/c/evidence.html) exposes all worlds, advice and choices, rather than a handpicked demonstration.

## A and the shared intervention

A's original 720-response action pilot failed baseline compliance with a harmless operating rule. A separately frozen 1,296-response diagnostic then failed technical-feasibility readiness in every arm. The model answered the blocked-harmless permission question correctly on 36/36 cases in every arm, while earlier action prompts selected the prohibited operation on 18/36 intact and 19/36 ablated cases. This is a dissociation across different prompts on exposed worlds, not evidence of knowledge at the decision moment or deliberate disobedience. No A confirmation ran. [A report and retained failures](workflows/a/report.md).

The custom implementation adapts the published refusal-direction approach, applying a fixed rank-one projection at decoder inputs and attention/MLP outputs. It does not change checkpoint weights. The vector was selected on separate refusal/competence data, without A/B/C outcomes. The fixed vector selected in the second instrument version reduced held-out overt refusals from **31/32 to 1/32**; intact, zero-sham and random arms each refused 31/32. Assistance labels include safe alternatives and factual corrections, so this is not a harmful-compliance rate.

Both original qualification failures remain retained. A single prospective extension kept the vector fixed and tested 192 newly sampled ordered arithmetic items: intact **182/192**, ablated **180/192**, random **183/192**. These contain 190 unordered pairs, with two reversals of prior pairs; the instrument report discloses that limited overlap. New-item and pooled gates and a conservative paired noninferiority bound passed, separately qualifying the instrument for task-specific pilots. This checks narrow arithmetic competence, not all capabilities. [Instrument report and qualification history](records/instrument_v2/report.md).

## What this supports

B and C form the strongest combined investigation: **a refusal-suppressing intervention can affect choices beyond verbal refusal, and those effects can reach an intact agent through another agent's advice.** The evidence is conditional on one checkpoint, greedy nonthinking operation, one selected vector, one random control, and a few exposed procedural families. Numeric worlds were held out across the main collections, but the families were not novel. The intervention acts broadly across layers; no native circuit, unique refusal mechanism, motive or subjective experience has been established.

The five numbered studies provide complementary constraints: organizations resisted forbidden peer proposals in Exp1; internal monitor transfer failed in Exp2; communication causally improved ordinary coordination in Exp4; monitor false alarms depended strongly on wording and response format in Exp5; and payoff effects transferred weakly and changed with configuration in Exp6. Their [separate report](../agent-safety-20261016-continuation/REPORT.md) records exact counts, invalid outputs and failed attempts.

A useful next research decision would be whether to prioritize generalization across independently authored task families and a second model, or isolate the message features mediating C's reviewer effect. Those are proposed extensions, not launched jobs. The present application narrative can already show a substantive question, controlled intervention, meaningful controls, failed instruments and task repairs, held-out numerical instances, exact reproduction, and honest limits. It should not claim a newly discovered general safety axis or real-world agent misconduct.

Reproducible code, frozen configurations, prompts, raw outputs, intervention provenance, figures and CPU replay receipts are linked from each workflow report. Independent reviews were performed by separate assistant agents using the same inherited model family; they are not human review or cross-model semantic validation. The [collection audit](records/collection-audit.json) concerns engineering integrity, while workflow qualification concerns scientific readiness. Neither overrides A's failures or the original failed instrument and B pilot. Accounting includes the first safety pass exactly once, excludes earlier curiosity-project allowances, and adds no API charges. Storage/transfer reserve is an allowance, not recorded spending.
