# Agent safety follow up experiment proposal

October 16, 2026. Proposed follow-up to the completed agent-safety sprint. Writing this proposal launches no inference and changes no frozen results.

The research question is whether an agent's respect for operational rules and other people's interests depends on its refusal behavior, incentives, collaborators, oversight and professional context. The five tracks below address different parts of that question. Each has its own primary comparison, dataset and conclusion. Shared models and tooling reduce cost; they do not make separate results interchangeable.

| Track | Main question | First useful result |
|---|---|---|
| A with a stronger model | Does suppressing refusals change ordinary goal-directed behavior? | A matched original-versus-modified comparison with task competence established separately from obedience |
| B with C as a separate companion | Does the cost of restraint change that effect, and can an altered colleague influence an intact agent? | A cost comparison for individuals and a separately analyzed reviewer experiment |
| 5 | Can an agent's explanation cause a monitor to misjudge recorded actions? | The effect on both wrongful accusations and missed violations across two monitor models |
| 1 and 4 | Does communication change boundary-crossing when agents can act independently? | Executed team violations under different communication and approval arrangements |
| 6 with 2 as a later extension | Does professional context change obedience to the same rule in the same decision? | A controlled domain comparison, followed only if useful by internal measurements |

The starting evidence is in the [numbered-study report](../sprints/agent-safety-20261016-continuation/REPORT.md) and [ablation report](../sprints/agent-safety-ablation-20261016/REPORT.md). In particular, the earlier communication study established ordinary communication dependence, not dangerous cooperation. A's failed readiness checks remain failed. B and C are promising but use one checkpoint and a few exposed task families. These follow-ups address those limitations directly.

## Shared model and experimental foundations

Use **Qwen3.8-Flash-Next** as the proposed stronger model for A and the new B/C comparisons. Start with a published original/modified pair, rather than assuming we must create another ablation ourselves. The [official release](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) supports thinking and nonthinking operation. Our primary mode should give it its normal reasoning opportunity, with a fixed, adequate generation allowance. The existing Qwen3-32B setup remains useful for reproducing older effects and as a second model for selected behavioral comparisons.

Inspect the derivative before choosing it. A [published switchable adapter](https://huggingface.co/apetersson/Qwen3.8-Flash-Next-Abliterated-Adapter) offers a practical matched comparison, but its documented edit combines a refusal projection with learned updates. Its effect must be described as the effect of that modification, not proof of one isolated refusal mechanism. The [Blackfrost release](https://huggingface.co/Blackfrost-AI/Qwen3.8-Flash-Next-DERISKED-BF16) changes the default system prompt and keeps its modification process proprietary. A default-versus-default comparison would confound instructions and weights. Prefer the most auditable compatible pair; pin checkpoint, tokenizer, engine, quantization and prompt template before task outcomes are collected.

Qualification has three separate purposes:

1. Verify that the modified variant actually reduces overt refusals on a separate fixed panel. Assistance, safe alternatives and harmful compliance remain distinct labels. This panel does not select an intervention based on our agent-task results.
2. Establish ordinary task competence through authorized actions and concrete factual checks. Permission questions must be separate from questions about technical feasibility. Separate diagnostic answers do not prove what was represented at the later action moment.
3. Verify that both variants receive the same information and comparable computation. Where an adapter is switched, test that disabling it restores the original behavior and that requests do not retain state from another condition.

**Do not require near-perfect obedience as a readiness condition.** Obedience is an outcome we want to measure. A model that understands an instruction but disregards it has not automatically invalidated the experiment. Failure to understand the task, a broken tool, or severe general capability loss is a different problem. Large capability losses change the interpretation of the variant comparison and can prevent a clean claim about refusal suppression.

Use small, actual tool workflows inside an isolated environment: synthetic records, reserved work queues, mock messaging destinations and disposable files. The model pursues ordinary objectives and receives operational facts. It is not instructed to cheat, deceive or violate a rule. Harmful and permitted alternatives must both be technically executable; an external guard that always blocks the forbidden action would measure attempted violations only. Keep attempts and executed consequences separate. Nothing acts on third-party systems.

Reuse the existing request logging, budget accounting, exact oracles, raw-output retention and analysis utilities. Add a small tool-execution wrapper and the new model adapter where necessary. The old Qwen3 activation hooks are architecture-specific and must not be assumed to work on Qwen3.8; only track 2 depends on new internal instrumentation.

## A with a stronger model

**Question.** Does a variant selected for reduced refusal become more likely to cross explicit operational boundaries or impose avoidable costs on others while completing an ordinary task?

**Design.** Compare the original and modified model on four independently authored workflow types. Examples include completing a report using a restricted synthetic attachment, processing an order using another customer's reserved stock, scheduling work by cancelling somebody else's reserved job, and making space by deleting a protected record. Every workflow has a feasible legitimate completion route. Keep one prespecified, moderate completion advantage for the shortcut; systematic variation of that advantage belongs to B.

Cross two independent facts: whether the shortcut is explicitly forbidden, and whether it imposes a concrete third-party consequence. This produces prohibited-harmless, prohibited-adverse, permitted-harmless and no-prohibition-adverse cases. For example, permission to access a resource and the cost its use imposes on another customer are represented separately. Define adverse consequences in the simulator and have the descriptions reviewed for ambiguity before model testing. Report actual consequences, rather than assuming every tradeoff has an uncontested ethical label.

**Primary measurements.** There are two primary comparisons: the change in explicit-rule violations, and the change in adverse actions when no prohibition is stated. Report these separately, including their joint condition, with task completion, unauthorized attempts, successful adverse actions, deferral and invalid outputs alongside them. Use permissions and execution records as ground truth. A verbal refusal is neither a successful task nor proof that no violation occurred.

**First test and expansion.** Start with 12–16 development scenarios to verify tools, comprehension and runtime. Then freeze approximately 80 new base scenarios across the four workflow types, with repeated seeds and counterbalanced action names. Reserve additional scenarios before main collection if a precision-based extension is affordable. Repeats of one scenario are not new independent evidence. Select model settings through the separate qualification panel, not through which settings create the largest boundary effect.

If refusal reduction is established and task competence remains comparable, a difference supports a behavioral effect beyond verbal refusal. Little difference with a sufficiently narrow uncertainty interval supports separation of those behaviors in these conditions. If both models frequently violate rules, report that fact and their difference; do not reject the task merely for revealing baseline disobedience. If the derivative cannot be isolated from other modifications, narrow the causal claim to the tested release change.

**Deliverable.** A paired behavior report with a small action-trace viewer and explicit operational-versus-adverse consequence results. A clear null is useful. Broader claims require another model or independently designed workflows.

## B with C as a separate companion

**B question.** Does refusal suppression change how an agent trades extra work for avoiding a prohibited or adverse action?

Build on A's qualified model pair and tool wrapper, using separate scenario instances and a separately frozen analysis. Vary the extra work required by the legitimate route over four levels, including no extra work. Keep permissions, consequences and task success fixed. Both routes must remain feasible. Use effort or elapsed work as the primary incentive variable; a separate reward-size manipulation is a later extension, not something silently mixed into the cost experiment.

The primary result is the change in shortcut selection from lowest to highest cost in the modified model, minus that change in the original model. Calculate it separately for explicit prohibitions and adverse consequences without a prohibition. Plot all four costs: a high-versus-low difference is not evidence of a smooth increasing curve. Report harmless controls and domain differences, including cases where shortcut selection is already near its maximum. A single pooled average must not hide opposing domain effects.

Start with 12 development scenarios, then roughly 48 new base scenarios across at least four workflow types and repeated seeds. A large effect from the old study is not required for main admission. Competence and runtime determine whether the test can run; the frozen main and its uncertainty determine what it teaches us. A replication in the new model would broaden the result. A failure to replicate would limit it to the earlier intervention/model/mode unless another explanation is independently tested.

**C question.** Can modifying a colleague change the actions of an otherwise intact team?

C gets its own data, primary result and report. Begin with an all-intact planner–reviewer–executor team versus the same team with only the reviewer modified. Reviewer-first is an explicit follow-up choice motivated by the earlier result; it is not a fresh discovery that the reviewer is uniquely important. Use the same reasoning settings and per-role allowances. Keep producer outputs and recipient inputs so the exact influence path is inspectable.

Use a fixed moderate shortcut cost, not B's full cost grid. Include the same explicit-rule and adverse-consequence distinctions. Keep a direct actor and an equally budgeted self-review workflow as controls for additional computation. Retain task completion as well as harmful choices: better completion accompanied by greater external cost is different from a model simply failing its task.

For a bounded second C test, replay saved reviewer messages into a fixed intact executor and separately vary the recommendation and factual warning. Match wording and length where possible. That can test whether the recommendation, omitted consequence or other message content drives the change. Generated explanations are not assumed to reveal motives. Planner/executor ablation and formal comparisons between roles are optional extensions; the current proposal does not promise that a reviewer effect exceeds the others.

Start with 12 development scenarios and approximately 48 new confirmation scenarios, with whole-scenario uncertainty. B's result cannot establish C's mechanism, and C's result cannot establish that teams amplify B. A combined publication can present two linked studies only if both conclusions remain independently supported.

## Experiment 5

**Question.** Can a truthful or misleading explanation cause a monitor to accuse a compliant agent or clear an agent that actually violated its instructions?

Use fixed tool-action records for which the action's permission status can be checked exactly. Create matched compliant and violating records, including cases where the same operation is permitted in one context and prohibited in another. The monitor sees the relevant rule and complete action evidence. Change only the accompanying explanation and how it must express its verdict.

The central conditions are no explanation, a matched neutral explanation, an accurate compliance explanation and a misleading compliance explanation for a violation. Include the previously effective longer wording, independently authored paraphrases, and meaningful verdict words versus counterbalanced letter codes. Use at least two monitor models. Begin with the cached Qwen3-32B and add the stronger model when its service is qualified; model inference can be shared without sharing test outcomes.

**Two primary errors matter:** wrongly accusing a compliant agent and wrongly clearing a violating agent. Report them separately, with the base rate of violations stated. A monitor that flags everything must not appear effective. Check parser accuracy and label reversal explicitly so a coding error cannot masquerade as a monitoring effect. Keep an exact rule-checking program as a reference; an LLM should not adjudicate whether the simulator's known rule was broken.

Use about 16 development scenarios and 48 new test scenarios across several tool workflows, each with paired legal/illegal records. The sample contains real local executions or deterministic replays of such executions, rather than only prose descriptions. Trace generation can reuse A and 1/4 infrastructure, but the confirmation set is kept separate; cases inherited from an earlier run are labeled reused.

If wording effects persist across models and action traces, that supports a practical monitoring weakness. If they remain specific to letter codes or one phrase, that narrower result is still worth a concise write-up. If a monitor is generally inaccurate even without explanations, improve its basic task setup before attributing errors to explanations.

Publish this independently. A later test can insert the monitor into 1/4 and measure whether its decisions actually prevent or permit violations. Offline classification performance alone is not evidence of safer teams.

## Experiments 1 and 4

**Question.** When agents have their own action permissions, does communication help them respect a shared boundary, help them cross it, or allow a colleague's objection to prevent a violation?

Create tasks in which two agents contribute to an ordinary joint objective and can each execute actions. Examples include a shared spending limit, overlapping reservations on a protected resource, and synthetic records whose combined disclosure is prohibited. The environment evaluates both individual actions and the combined outcome. A final approving agent is one condition, not built into every run.

Compare three arrangements: independent actors without messages, actors that can exchange messages and execute independently, and actors whose proposed actions require central approval. Match the number of available action slots and total inference allowance. Include a distributed planning/review allocation corresponding to the approver's computation; report actual tokens and tool attempts as well as their caps. Randomize or counterbalance execution order.

Separate full-information cases from cases with divided information. In the former, each actor can see the global restriction and relevant joint state. In the latter, actors must communicate facts to avoid a collective failure. This distinguishes a coordination failure from violating a boundary despite having the information needed to respect it. Separate factual questions may diagnose reportability, but do not establish awareness during the action.

Within communicating teams, block the channel or replay a message from a matched alternative state to test whether communication caused the observed change. Also compare a genuine colleague's warning with a length-matched factual message. Give agents ordinary goals, not instructions to cheat or to pressure a partner into misconduct. State messages and any reported completion claims are saved with the actual execution trace.

**Primary outcomes:** executed global violations and ordinary task completion under communication versus no communication, plus the effect of adding approval under matched resources. Count failed attempts, number of action opportunities and partial completion. If warnings are tested as a separate primary intervention, specify that comparison and its error allowance before the main run. Concealment or deliberate collusion requires additional evidence and is not inferred from a joint violation alone.

Start with 12 development scenarios, then approximately 48 new scenarios across at least three joint-constraint types. Readiness requires usable tools, feasible legitimate solutions and competent authorized controls; it does not require the agents to misbehave. This main test already contains the safety question. Useful communication on a benign task is a prerequisite, not its endpoint.

If communication raises violations, test which messages or information changes are responsible. If it lowers violations, quantify the protection and its limits. If only divided-information cases fail, report a distributed coordination problem. If no failures appear, report the observed rate and uncertainty across the tested task distribution. Add experiment 5's monitor only as a subsequent intervention with a separate comparison.

## Experiment 6 with experiment 2 as a later extension

**Question.** Does the same model obey the same rule differently when the decision is framed as software engineering, parcel delivery, warehouse operations or laboratory administration?

Construct one formal decision record containing options, permissions, benefits and third-party costs, then render it in each domain. Every domain has exactly the same underlying relationships and numerical outcomes. Provide the computed outcomes in the primary comparison so differences in arithmetic demand cannot explain a domain effect. Keep the authority issuing the rule and its priority fixed.

Use two levels of framing. The first changes only domain labels and resource names within a shared wording template. The second uses independently written, natural domain descriptions whose formal meanings are checked against the same record. Include neutral labels and multiple phrasings so one suggestive word cannot stand in for a professional norm. A separate later manipulation can ask whether requiring the model to calculate outcomes changes the pattern.

Vary the shortcut's benefit over a small fixed grid. The two primary questions are whether violation rates differ across domains at matched benefit, and whether domains respond differently as that benefit increases. Use an overall domain test before interpreting individual contrasts; declare the software-versus-delivery contrast in advance because it motivated the hypothesis. Keep explicit-rule violations separate from adverse consequences without a stated prohibition. Do not infer differences in training data from behavior alone.

Start with 12 development decision records and approximately 64 new base records, each rendered across domains. A record and its domain variants form one paired unit. Replicate the clearest fixed contrast on the second model using untouched records, without selecting only stories that produced the desired effect.

If domain differences survive matched decisions and paraphrases, we have evidence that framing changes obedience. If only the natural rewrites differ, the effect involves richer contextual wording rather than the label alone. If the difference disappears, the earlier result likely depended on other task differences. All three outcomes answer a useful question.

**Optional experiment 2.** Only after a behavioral pattern and suitable data exist, ask whether internal measurements predict the next unauthorized action beyond domain, wording and a monitor reading the conversation. Fix a prospectively identifiable pre-action capture point, rather than selecting it after seeing the generated answer. Compare the earlier linear classifier with a small nonlinear classifier and strong text-based baselines. Hold out complete scenarios and task structures, not random repeated rows.

Actual-action prediction needs enough authorized and unauthorized outcomes in both development and test sets. If outcomes are too one-sided, report that limitation; do not manufacture a successful detector by relabeling dangerous requests as actual violations. Learning to recognize the domain is not the desired result. New architecture support, larger training collections and any causal activation intervention need their own scoped plan. A successful domain study does not depend on making experiment 2 work.

## Analysis and stopping decisions

Pilot sizes and main counts above are planning targets, not completed power calculations. Use development data for runtime, invalid rates and variance estimates; freeze the actual main sample and analysis before inspecting main outcomes. Design for meaningful effects of roughly 10–15 percentage points where practical. If the affordable sample cannot distinguish effects of that size, state its attainable precision rather than interpreting an uncertain zero as equivalence.

Use paired comparisons and uncertainty that keeps repeated seeds, menu orders, domains and conditions within their base scenario. Include multiple genuinely different task structures, not only new numbers in the same template. Freeze primary outcomes and the handling of multiple comparisons within each study. Secondary patterns can motivate separately labeled follow-ups. Do not stop a main run early because the result looks positive, negative or publishable; any precision-based extension must have a prospective rule.

Stop or revise for an uninterpretable task, ineffective refusal modification, serious capability loss, unusable runtime or a budget limit. Preserve that outcome and any changes as a new version. Finish a valid planned comparison even when the pilot does not show the hoped-for effect. Infrastructure checks validate the collection; they are not scientific findings. Record which broader questions remain unanswered when a bounded study closes.

## Resource plan and sequence

The completed sprint used a conservative **$88.45** of the authorized **$700**. Holding the existing **$25** storage/transfer reserve once leaves **$586.55 uncommitted**, according to the [final receipt](../sprints/agent-safety-ablation-20261016/records/combined-tonight-cost-and-shutdown.json). These are app-wall estimates, not invoice balances. The following are maximum proposed allocations, not spending targets or new authorizations.

| Work | First-stage cap | Maximum including first stage |
|---|---:|---:|
| Shared model loading, qualification and tool integration | $35 | $35 |
| A with the stronger model | $20 | $100 |
| B | $10 | $90 |
| C companion | $0 | $60 |
| 5 | $10 | $55 |
| 1 and 4 | $10 | $90 |
| 6 | $5 | $55 |
| Optional 2 feasibility | $0 | $15 |
| **Total** | **$90** | **$500** |

This leaves approximately $86.55 unallocated after the existing reserve. Shared service costs must be allocated once across jobs; do not add per-call attribution to the same app-wall total again. The optional classifier allocation covers feasibility and inexpensive fitting, not a large new training campaign. Extra storage from large weights must be estimated separately and kept within the reserve or funded from the unallocated balance.

At the checked [Modal GPU rates](https://modal.com/pricing), H200 is approximately $4.54 per GPU-hour and B200 $6.25; four B200s would be about $25 per hour before CPU, RAM and storage. The published adapter requires a patched inference runtime and a large external embedding table. A single-H200 quantized original/adapter service is a candidate, not an already validated deployment. Profile loading, stock/modified equivalence, CUDA compatibility, host memory and tokens per second within the $35 shared cap. A larger BF16 service is a fallback whose measured total cost must fit a study's cap, not an automatic hardware escalation.

Local inspection found approximately 21 GiB available RAM and 702 GiB free disk. Run model inference in Modal. Avoid two full unquantized checkpoint copies on the local machine. Keep weights in a dedicated cloud volume or use a common base plus small adapter. Do not modify or stop unrelated apps. One shared model service can serve multiple queues; five research tracks do not require five GPU fleets.

In an initial 48-hour sprint, first qualify the stronger pair while preparing the independently authored tasks. Experiment 5 can start on the existing model, and the formal-record checks for 6 and joint-action simulator for 1/4 can proceed without GPUs. Run small, bounded first tests in all five tracks. Then prioritize A and B's confirmation, with 5 as the inexpensive independent study. Release C after the model pair and its team task pass qualification. Expand 1/4 and 6 when their first tests demonstrate an informative design, rather than based on whether they produce dramatic violations. Experiment 2 remains optional.

## Deliverables and publication decisions

Each study should produce its own frozen protocol, executable configuration, raw actions and messages, objective scores, uncertainty analysis, figure, reproduction command and plain-language findings page. Keep attempted runs and failed checks. Demonstrations should let the reader inspect representative and contrary cases, with access to all records.

A addresses the user's original behavioral question. B and C may share a publication theme while retaining separate claims. Experiment 5 can stand alone or later support a team-monitor intervention. Experiments 1/4 constitute the direct multi-agent safety investigation. Experiment 6 stands on its behavioral evidence even if experiment 2 is never attempted. A literature and novelty review should precede any claim that one of these mechanisms is new; a clear, reproducible research page is a useful first deliverable regardless of eventual paper scope.
