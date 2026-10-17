# Bounded task revision 2: prospective validation plan

October 6, 2026. Written before revision-2 model inference.

This is the second and final task-development revision allowed by the current protocol. The user authorized improving task comprehension and repeating validation. Prior runs, including failures, remain intact. The pre-revision implementation is archived at `archives/before-revision2.tar.gz`.

## Frozen changes

1. Retain the same affine machine rules, candidates, hidden answers, and evidence. Add exact lookup tables for the two relevant inputs. This removes modular-arithmetic demands from the identification check. The revised task tests lookup-supported rule identification; its distinction from object-table identification is weaker and must not be presented as a replication of arithmetic reasoning.
2. Spell out each service's action on the hidden candidate. Use the same service descriptions in behavioral and instrumental/comprehension prompts. Stop is explicitly free and gives no observation. No instruction favors exploration or conserving points.
3. Add separate checks of target-service mapping and repeated-observation mapping. These distinguish reading the menu from understanding which evidence identifies the candidate.
4. Replace the repeated accounting question with 200 distinct transaction contexts, sampled by a deterministic rule before inference. List purchases on separate lines and provide four shuffled possible balances. Every context has exactly one correct answer. The original accounting failure remains reported; these are revised controls, not corrected scores for the old run.

The model, model revision, thinking-disabled setting, BF16 precision, sampling settings, response-format instruction, initial behavioral endowment, prices, hidden tasks, and model weights remain fixed. No candidate vectors are extracted or applied. Changes to reasoning mode, model selection, or further prompt revisions are outside this bounded revision.

## Sample and advancement rule

Use Qwen3-8B on the same 100 development mysteries per family as the earlier engineering pilot. These are reused development items, not held-out confirmation. Each item receives six fresh-context diagnostic branches: fully revealed identification, unresolved candidate probabilities, instrumental clue selection, target-service mapping, repeated-observation mapping, and accounting. Total: 1,200 diagnostic branches.

Before any new behavioral baseline, require all assigned diagnostics to finish. In **each family separately**, every scored diagnostic must have at least 100 items, at least 90% correct responses, and at least 95% valid-format responses. Unresolved candidate probabilities are recorded but have no correctness gate. Invalid responses and execution failures remain failures in the assigned denominator. Thresholds are fixed here before observing the revision's outputs.

If every check passes, collect the same crossed behavioral baseline: 100 items per family × two evidence conditions × four prices = 1,600 trials, with the first sampled action primary and at most three purchases per episode. The baseline manifest must reference the passing validation's hashes and match its source, model, task revision, and planned assignments. Report action validity and informative-choice sensitivity, including floors and ceilings. Passing the diagnostic checks does not itself establish discretionary point valuation or psychological interpretation.

If any check fails, stop task revision under the current allowance and report the assay as unvalidated. Do not collect another behavioral baseline, lower thresholds, switch inference modes, or search for a passing subset. Any later change requires a prospective protocol amendment rather than being silently folded into this revision.

## Execution and resource bounds

Run CPU-only invariants first, then the frozen research-model diagnostics on one Modal L40S. Model-integrity audits run before collection. Use immutable manifests, saved code snapshots, retained attempt records, checkpoints every 25 records and at completion, a two-hour aggregate deadline per scheduled phase, and at most three worker admissions. Copy and verify final artifacts locally and confirm workers are stopped.

Both phases, if needed, fit within the previously authorized USD 25 pilot ceiling: the earlier work was estimated at USD 0.52 compute, and the combined runtime limits for these two single-worker phases imply approximately USD 9.72 of compute at the configured maximum CPU/RAM rates, plus startup allowances where applicable. These are scheduling estimates, not billing statements; no new USD 25 allowance is assumed. Expected actual runtime is much shorter based on the completed pilot profile.
