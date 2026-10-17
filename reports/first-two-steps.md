# First two steps: implementation and baseline results

October 6, 2026

The task simulator and inference infrastructure are implemented, and the first unsteered Qwen3-8B baseline has been completed on Modal. The engineering checks pass. **The current tasks do not yet pass the comprehension and competence checks needed to interpret information purchases psychologically.** Task validation should precede curiosity/satisfaction interventions.

## Delivered

Both task families have deterministic generators, independent finite-hypothesis oracles, verified decisive and redundant clues, simulator-owned point accounting, stopping and invalid-output rules, and isolated branches. Four inspectable examples are in [simulator.json](../examples/simulator.json).

The runner uses pinned model revisions and dependencies, full-vocabulary sampling, fresh contexts, separate diagnostic branches, activation capture, removable hooks, immutable assignment manifests, and checksummed records. The Modal wrapper supports bounded execution, durable checkpoints, restart reconciliation, and local artifact export. [README and commands](../README.md)

Eighteen automated tests pass. The local 1.7B integration model and the cloud 8B model pass the real-model hook checks. Both cloud audits verify exact baseline/sham output equality, changed activations under a nonzero test perturbation, single-position hook execution, cleanup after exceptions, and an unaffected subsequent branch. The test perturbation is an engineering fixture, not a curiosity or satisfaction direction.

## Baseline observations

The primary model was `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`, in BF16 with thinking disabled. There were 100 development mysteries per family, each crossed with two evidence conditions and four prices: 1,600 behavioral trials and 800 independent diagnostic branches. All 2,400 assigned records are present, with no duplicates or recorded execution failures. All 1,600 first choices were valid.

| Diagnostic | Rule machines | Hidden objects |
| --- | ---: | ---: |
| Fully revealed identification | 31/100 correct | 100/100 correct |
| Instrumental clue selection | 49/100 correct | 19/100 correct |
| First-action format validity | 800/800 | 800/800 |

The machine family fails the proposed 90% fully revealed accuracy gate. Instrumental service comprehension is inadequate in both families. The accounting control returned `9` instead of `6` in every repetition. That control repeats one identical greedy prompt; it is **one failed accounting problem**, not 200 independent demonstrations of poor numeracy.

For unresolved objects, target purchases occurred in 8%, 7%, and 7% of trials at prices 1, 2, and 4 respectively, below the provisional 10% lower sensitivity target. For unresolved machines the corresponding rates were 22%, 24%, and 14%, but the failed competence and comprehension checks prevent motivational interpretation. Resolved trials often bought additional target observations, and no first choices selected independent novelty. These patterns are descriptive diagnostics, not evidence for curiosity, satisfaction, or their absence.

The complete condition table and denominators are in the [baseline report](../runs/modal-qwen8b-baseline-v3/report.md). [Machine-readable audited summary](baseline-summary.json)

## Independent decoder check

After the failures appeared, the custom inference loop was compared with Transformers' standard `generate()` on three fixed greedy prompts for each model. All generated token sequences matched. Both models answered the symbolic expression `12 - 3 * 2` correctly but failed the original accounting word problem. This supports treating the word-problem failure as task/prompt-dependent behavior in the tested inference setting; it does not establish a general inability to do arithmetic. The extra prompts are post-hoc engineering probes, not replacements for the failed baseline controls.

[1.7B comparison](generation-parity.json) · [8B comparison](generation-parity-qwen8b.json)

## Replay, resources, and cost

The cloud restart preserved all 86 previously committed records byte-for-byte at the canonical-record level. The final audit also verified every trial's balance transitions and uniqueness of every assigned ID. Checksums were verified after downloading the artifacts to the local project. Original source and the infrastructure revision are archived beside the run.

The L40S profile measured about 15.43 GiB peak allocated GPU memory and 15.82 GiB peak reserved memory. Loading from the cached weights took 10.8 seconds. The resumed segment completed its remaining 2,314 records in 321.4 seconds, excluding loading and integrity checks. Across the complete baseline there were 3,215 inference calls totaling 206.8 seconds inside the inference method. This is a baseline profile; full activation extraction and cache experiments will require additional profiling.

Every task-related Modal app was verified **stopped with zero tasks**. Including failed integration launches and the independent decoder check, total app wall time was 763 seconds. Applying the published L40S rate plus the configured CPU and maximum host-memory rates throughout that entire interval gives a conservative **USD 0.52 compute estimate**, comfortably within the authorized USD 25 pilot budget. This is not a billing statement; stored weights and any transfer charges are separate, and account credits were not inspected. [Modal pricing](https://modal.com/pricing) · [Saved app states](modal-app-status.json)

Weights remain in `curiosity-study-weights` for reuse. Research records remain in `curiosity-study-artifacts` and in the local run directory.

## Next research step

Use the remaining bounded task-development revision to improve service explanations and machine-task comprehensibility, then repeat the comprehension and competence controls before collecting another behavioral baseline. Preserve the current failures. Make any change to reasoning mode prospective and explicit, since it changes the inference setting. Broaden the accounting controls to genuinely distinct problems and validate discretionary point valuation separately.

Candidate extraction and causal emotion-related interventions should wait until those checks pass. The object task's perfect identification score is encouraging for that narrow ability, but its service-comprehension failure means the full assay is not yet validated.
