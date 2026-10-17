# Target-relevant information — prospective conceptual amendment

October 14, 2026. This extension separates target relevance from the entropy of an accurate offered report. Following the successful relational comprehension smoke, its task oracle, 40-branch comprehension preparer, and independent audit are implemented and locally tested. No target-relevance outcomes have been collected at this revision. The first smoke tests comprehension; larger transfer and causal studies remain conditional. Existing completed collections remain unchanged. The motivation and primary-source precedents are recorded in [internal-mechanisms-literature-notes.md](internal-mechanisms-literature-notes.md).

## Construct and exact oracle

The present noiseless task equates full-world expected information gain with entropy of the offered report. This extension instead varies information about a designated target while holding report entropy, source accuracy, and full-world information gain constant. Its construct is **target-relevant information value**, not varying total-world information gain.

Four equally likely candidates contain two independent fair binary properties, X0 and X1, plus a derived target property T. Both possible reports accurately reveal their named property. No value of the actual hidden candidate is initially observed. For each base world, cross two factors:

1. u: which report property is independent of the target, either 0 or 1.
2. q: which report property is offered, either 0 or 1.

Define T = X(1-u) XOR b, where the fixed complement bit b is counterbalanced across world indices. Consequently, target gain is one bit exactly when q differs from u. In ordered cells (u,q) = (0,0), (0,1), (1,0), (1,1), the gain contrast is **[-,+,+,-]**, matching the existing axis orientation. Target and report properties always have different names; their relationship must be read from the candidate table. Property names, binary value labels, and row order are counterbalanced, with object and machine families and later table/prose renderings.

Before the first collection, independent review identified that complement `index % 2` would be confounded with action mapping `index % 6`. The implemented schedule instead uses complement `(index // 6) % 2`, crossed with all six action mappings in each 12-world block. The one-world-per-family smoke does not cover that full crossing; later confirmation must balance or explicitly stratify it.

| u | q | Target-linked report | Target gain, bits | Full-world gain, bits | Report entropy, bits |
|---|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 1 | 1 |
| 0 | 1 | 1 | 1 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 | 1 |
| 1 | 1 | 0 | 0 | 1 | 1 |

Prospective oracle fields must distinguish these quantities explicitly:

- `remaining_candidates_before`: 4; `expected_remaining_candidates_after`: 2.
- `world_entropy_before_bits`: 2; `target_entropy_before_bits`: 1.
- `offered_report_entropy_bits`: 1; `report_accuracy`: 1.
- `expected_world_information_gain_bits`: 1 in every cell.
- `expected_target_information_gain_bits`: 1 if q != u, otherwise 0.
- `expected_target_entropy_after_bits`: 1 minus target gain.
- `independent_report_attribute`: u; `queried_attribute`: q; `target_linked_report_attribute`: 1-u.

A generic legacy information-gain field must not silently change from full-world gain to target gain. Exact enumeration, separate from response scoring, must verify both quantities and their entropies before use.

## Prompt order and transfer unit

The candidate context and target designation come first, followed by the offered query, then **End of record**, then the question and response mapping. The query is before the record marker, as in the existing task. This permits query-dependent appraisal at the frozen record position; placing it afterward would force the record-position gain interaction to zero by construction and confound the position comparison. Opposite question polarities and their response mappings remain after the marker. For each (u,q) cell, the factual prefix through that marker is identical across readout kinds.

One base-world index supplies latent candidates, vocabulary, complement, and row order. Its four u-by-q cells include two counterfactual target-relation tables and form one indivisible analysis and split unit. Keep every kind, rendering, and intervention arm for that unit together. Renderings are repeated measurements, not additional independent worlds. Use fresh worlds with explicit semantic-overlap checks.

Confirmation must additionally canonicalize the paired counterfactual tables and designated-target role, ignoring display order, query, kind, rendering, and action labels. The unused hidden truth does not affect these prompts and must not make duplicate tables count as distinct semantic worlds. Deduplicate whole semantic blocks and exclude every inspected pilot/development block before outcomes. Existing transfer and causal adapters intentionally reject this task until its separate target-gain semantics and overlap rules are implemented.

Frozen-axis transfer retains the existing direction, sign, threshold, layer, and position; no recentering, refitting, or selection on this task's outcomes. Report both polarities, each family, all planned sites, and missingness. Subsequent causal work would require its own prospective schedule and interpretation gates.

## Conditional 40-branch comprehension smoke

The smallest proposed smoke has one base world per family, four cells per world, and five readout kinds: **2 families × 1 world × 4 cells × 5 kinds = 40 assigned branches**, using one fixed rendering per family. It is an engineering and comprehension check, not a powered validation or independent replication.

- **New:** would the accurate offered report reduce uncertainty about the designated target? Yes means target gain one.
- **Repeat:** would uncertainty about the target remain unchanged after receiving the report? Yes means target gain zero. This is the reverse question; the offered report itself is always new information about the full world.
- **Memory:** retrieve a named property value from a specified candidate-table row, using counterbalanced response labels. This checks access to table facts without revealing a fact about the actual hidden candidate.
- **Count:** count candidates still compatible with the record. The answer is always four here, so this is only a weak readout-stability check. A separate balanced 1/2/4-count competence block remains necessary before a specificity claim; it is outside these 40 branches.
- **Instrument:** under an explicit goal of obtaining information about the target, choose the offered report when it helps, otherwise stop; rereading the already supplied table is a zero-gain distractor. Bind report/reread/stop labels prospectively. Forty branches do not exhaust all six action-label mappings, which would be required in a larger controlled study.

Preserve every assigned branch, including invalid and capped completions, and report coverage and correctness by family, kind, and cell. One world per family cannot establish the existing minimum-world validity or competence gates, justify causal interpretation, or select a preferred axis. A larger study, costs, and any actual acquisition stage remain conditional decisions after these prerequisites.

## Prospective development expansion

Before inspecting the 40-branch smoke outcomes, the next conditional baseline is fixed at 24 world indices per family, all four cells and all five readout kinds: 960 branches. Use seed `2026101430`, indices 0 through 23, table rendering, and 4,096 generated tokens for every branch. Two full 12-world complement-by-action-label blocks occur within each family. The implementation is `scripts/prepare_target_relevant_development.py`; it cannot prepare a collection without the complete, bound, passing smoke and independent oracle audit.

All 40 family/kind/cell groups must meet the development thresholds: at least 24 assigned world indices, 90% correctness, and 95% valid format. No successful subset replaces the global gate. The 24 indices do not imply 24 distinct semantic worlds; report canonical overlap and duplicates. Every assignment becomes inspected development data and is excluded from later confirmation, including invalid or capped outputs.

Apply the existing frozen axes, signs, and extraction thresholds at all captured sites for descriptive transfer. Layer 47 at the record token remains the prespecified causal candidate; a different best site in the transfer report is not an independent replication. Indices 0–11 and 12–23 can support separately declared development calibration and checking, but neither is confirmation. Every subsequent natural intervention requires its own prospective schedule, engine-matched engineering pass, and fresh baseline competence checks.

## Limits of inference

The oracle matches initial target uncertainty, report predictability, and source accuracy. The model's subjective confidence is unmeasured and need not match those stipulated quantities. Successful transfer could establish sensitivity to target relevance, but would not alone identify a distinct utility representation or exclude answer bias, relational computation, or task-policy explanations. Full-world information gain is constant by design. The instrumental instruction tests compliance with a stated target-information goal; it does not establish intrinsic curiosity or a preference for acquiring information without that goal.
