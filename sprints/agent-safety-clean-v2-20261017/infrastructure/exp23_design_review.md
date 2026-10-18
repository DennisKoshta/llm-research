# Experiment 2.3 dependency design review

Status: **prepared, not executed**. `scripts/exp23_design.py` has no inference client, GPU launcher or result scorer. It generates exact pending artifact requests and emits dependent executor definitions only when the corresponding complete forty-eight-scenario condition has validated raw evidence. No authored memo, scenario or judge verdict was substituted by this implementation. Unit-test fixtures are confined to the test module and were never persisted as research data.

## Inputs, immutable snapshots and outputs

The CLI accepts a current-sprint JSON input bundle containing `raw_artifacts`, records its byte SHA-256, and writes a new immutable snapshot. An empty bundle creates forty-eight pending unified-author requests and no executor stimuli. The snapshot records `execution_authorized: false`, `PREPARED_NOT_EXECUTED`, separately counted pending requests and ready definitions, explicit blocked conditions, semantic rejections, source artifact hashes, and any unreferenced raw records.

```bash
.venv/bin/python scripts/exp23_design.py \
  --input-bundle stimuli/exp23_artifact_bundle_ATTEMPT.json \
  --output stimuli/exp23_design_ATTEMPT.json
```

Source and destination are confined to this sprint. A different existing snapshot cannot be overwritten. A raw artifact must reproduce its frozen request ID, role, messages, declared record tool, reasoning mode, cap, temperature and absence of forced prefix. It must include a model identifier, immutable revision or checkpoint hash, raw-source path, complete raw completion, token IDs/count and finish reason. Closed reasoning and exactly one valid JSON record are checked through the current sprint strict parser. Argument schemas are typed and reject unexpected fields. This provenance validation does not replace the launch coordinator's independent verification that model roles resolve to the approved checkpoints, or persistence verification of the raw-source files themselves.

## Dependency graph

1. **Unified author:** Forty-eight paired scenario requests, twelve per domain (`warehouse`, `finance`, `compute`, `wetlab`), with twenty-four High-Tedium and twenty-four High-Friction states. Each domain has six scenarios at each cost. The same author response provides shared context/goal, both consequence telemetry sentences, a route-neutral Planner summary, B0-neutral status, B3 harmless endorsement, and matched B4a/B4b semantic-role memos. All are actual pending generation requests, not handwritten replacements.
2. **Independent author assessment:** Raw structured judge responses must accept paired-context validity, Planner neutrality, factual-only neutral status, the harmless B3 recommendation, opposed B4 recommendations and matched B4 style. False judgments remain explicit rejections. Judge identity, rubric and provenance must be frozen before these requests run; this preparation does not claim any actual validation occurred.
3. **Natural reviewers:** For each authored state, fixed identical review prompts are addressed to Intact (B1), PCA seed 101 (B2), and Refusal-Ablated (R1). Their recommendation is unconstrained. The same source memo is replayed across both Executor authority framings. Author assessment and natural review may be prepared concurrently, but no affected Executor condition is ready until the author assessment has passed.
4. **Cross-family paraphrases:** Three fixed neutral rewrite requests per R1 memo go to Llama. Each rewrite requires its own complete raw semantic judge response validating fidelity, no added or omitted claims, and preserved recommendation. Duplicate rewrites are rejected. No isolated rewrite condition becomes ready until all three members for all forty-eight scenarios are validated.
5. **Clause extraction and validation:** Raw extraction records contain exact Python-character spans for all risk-minimization and urgency claims. The generator verifies source text, bounds and non-overlap, then removes only those character spans. A separate raw judge record validates category labels, completeness and preservation of remaining assertions.
6. **Executor definitions:** A condition is emitted only when all forty-eight scenario dependencies are ready. Failed stimuli cannot silently produce a forty-seven-scenario condition. Core conditions can become ready while mechanism conditions remain explicitly blocked.

Five separately identified pilot templates are available for each sufficiently populated artifact role and each ready Executor/mode definition. These are preparation templates, not passed pilots. Every actual bounded sub-sweep must obtain its own fresh five-trial H5 pass; this module does not schedule or launch them.

## Executor factorial and analysis units

Primary Executors are Qwen Intact and cross-family Llama. Both Peer and designated Compliance framings are crossed with every condition. Memo identity is fixed across framings. Tools use neutral operational names and structured `<tool_call>` decisions. All six presentation permutations occur eight times in every forty-eight-scenario condition; each permutation occurs four times at each cost, so tool order does not predict cost. Harmful and harmless base states have exactly identical prompt strings after the designated consequence sentence is masked; the required held-template bag-of-words audit remains an execution prerequisite.

| Condition family | Variants | `cot_on` requests across two Executors/two authorities |
|---|---:|---:|
| B0, B0-neutral, B1, B2, B3, B4a, B4b, R1 | 8 | 1,536 |
| P1 rewrites | 3 | 576 |
| Risk × urgency clause grid | 4 | 768 |
| **Maximum `cot_on` total** | **15** | **2,880** |
| Targeted `cot_off`: B0-neutral, B1, R1, B4a, B4b | 5 | 960 |
| **Maximum Executor request total** | | **3,840** |

The `cot_off` allocation is a prospective interpretation of the global targeted-conflict default, covering the main contagion contrasts and the semantic endorsement × authority comparison. It does not silently add `cot_off` runs for every secondary mechanism condition. All decoding is greedy, one rollout, with 4,096 (`cot_on`) or 1,024 (`cot_off`) tokens and no forced completion prefix.

P1 records have a shared `(scenario, executor)` cluster ID and a more specific aggregation cell for authority/mode. Average the three rewrites within each authority/mode cell; do not count them as three independent scenarios or pool the authority levels before estimating their interaction. B0 contains no reviewer, so its two authority records have identical prompts and are explicitly marked as having no identifiable authority effect. They must not be treated as independent evidence for an authority contrast. B3 is the matched harmless state; other core conditions use the harmful state.

## Unsupported natural clauses stay unsupported

Natural R1 memos may omit risk-minimization or urgency claims. Empty extraction arrays are valid observations. The generator never adds a claim, rewrites a memo to force an endorsement, or calls an absent claim “present.” If any scenario lacks both validated source claim types, the full forty-eight-scenario clause factorial remains blocked and is reported as such; independently validated core conditions can still be ready. There is no post hoc selection of only the claim-bearing scenarios and no claim that the blocked mechanism study is complete. A different estimand or alternate stimulus design would require an explicit documented scientific amendment, not silent code adaptation.

## Cross-family and external-slice limits

The Llama Executor and paraphraser receive explicit system-level formatting instructions to close a `<think>` block before structured JSON; this is a prompt instruction, not an injected completion prefix. Native Qwen reasoning uses its frozen template. Each model family must independently pass five fresh H5 trials, including the actual serialized chat template and strict parser behavior. This module does not assume Llama follows Qwen's native template convention.

The failed external Flash-Next checkpoint-pair qualification excludes that external slice explicitly. The primary design is not gated on the failed external pair. Its omitted slice is marked incomplete and can never be pooled with primary records or silently reported as executed.

Execution remains behind primary Phase 1 qualification/calibration/hygiene, frozen judge and model registries, raw persistence, and the single resource/budget coordinator. Exp 2.3's first full sweep must actually launch after its H5 pass and receive a durable call ID before independently frozen Exp 1.1 may overlap. Original Phase 3 gates are unchanged.

## Validation

Command: `.venv/bin/python -m unittest discover -s tests -p test_exp23_design.py -v`.

Seven tests passed. Tests cover empty-input pending requests without invented data; all 3,840 fully populated fixture definitions; domain/cost/order balance; paired authority behavior; three-paraphrase clustering and missing-judge blocking; absent-claim preservation and blocked clause grids; exact span removal and overlap rejection; B0 duplicate labeling; B3 harmless states; strict rejection of unclosed reasoning, changed prompts and unpinned model revisions; and refusal to turn one failed author assessment into a selected forty-seven-scenario sweep. These are design tests only and provide no empirical evidence about model behavior.
