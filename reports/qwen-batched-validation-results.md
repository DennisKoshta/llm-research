# Batched Qwen comprehension validation: all checks pass

October 7, 2026. **All 1,400 assigned responses were valid and correct.** Every one of the fourteen frozen family/check cells passed its minimum 90% correctness and 95% format-validity requirements at n=100. This establishes task comprehension in the tested setting; it does not establish curiosity, satisfaction, or inner experience.

## Setting and result

Qwen3-8B used its pinned revision, exact H100 hardware, BF16/eager attention, temperature 0.7 with full-vocabulary sampling, and a 2,048-token maximum. The new `torch_batched_v1` backend processed independent trial RNG streams in cohorts of up to 32. Fresh seed-20261011 pairs were frozen before collection. The earlier 768-token and OLMo failures, the serial 2,048-token pass, and the incomplete serial calibration remain separate records.

| Comprehension check | Machines correct / valid, n=100 | Objects correct / valid, n=100 |
| --- | ---: | ---: |
| Identify X | 100 / 100 | 100 / 100 |
| Identify Y | 100 / 100 | 100 / 100 |
| Useful clue for X | 100 / 100 | 100 / 100 |
| Useful clue for Y | 100 / 100 | 100 / 100 |
| Consequence of choosing X | 100 / 100 | 100 / 100 |
| Consequence of choosing Y | 100 / 100 | 100 / 100 |
| Consequence of declining | 100 / 100 | 100 / 100 |

No response reached the token cap. The 566,521 generated tokens averaged about 405 per response. These observed rates are development observations, not guarantees of future performance or evidence of equivalence to serial inference.

## Execution and integrity

The separate 28-response cloud smoke tested memory and timing only and was excluded from these denominators. A first single worker saved 96 full-sample responses before a timing-only scheduling change. The remaining 1,304 assignments ran on four exactly reserved workers within the original deadline. Whole original cohorts and their row order were preserved; an independent comparison verifies all 44 cohorts against the original manifest. Two initial app creations were rate-limited before allocation and recovered under the same claims. No completed response was regenerated or discarded.

All five real-model integrity audits passed, including baseline/sham checks and batch-specific row isolation, padding, and hook cleanup. The independent record audit verified exact 1,400-record coverage, assignments, prompts, parsing, sampling settings, actual cohort membership, and removed hooks. A separate local smoke matched the pinned Transformers greedy batch generator on variable-length prompts with early termination; it does not establish sampled cross-shape equivalence.

The sum of unique cohort inference times was 2,427.86 seconds (0.674 GPU-hours). Per-row latencies overlap and must not be added. Completed worker profiles show peak reserved memory no greater than 31.41 GB. Conservative cumulative compute at the all-stopped boundary was **$53.7648**, an increase of **$3.3757** since the separate batch smoke. Costs include orchestration and the scheduling interruption through app wall time; they are not invoice totals. Different samples, execution paths, and overheads prevent treating the difference from the earlier serial study as a controlled benchmark.

## Advancement

The complete audited pass admitted a matching 1,200-choice calibration with the same model, source, backend, hardware, pairs, and sampling configuration. That calibration must still satisfy its own fixed format, choice-frequency, and layout-sensitivity criteria. Its scientific readout includes absolute resolution contrasts, paired descriptive uncertainty, and baseline headroom. No desired resolution effect is required.

Evidence: [prospective amendment](batched-execution-amendment.md), [full audit](qwen-batched-validation-audit.json), [cohort audit](qwen-batched-validation-cohort-audit.json), [response diagnostics](qwen-batched-validation-response-diagnostics.json), [cost/shutdown](qwen-batched-validation-cost-and-shutdown.json), [local reference parity](batched-reference-parity.json), [readout plan](choice-readout-plan.md), and [run report](../runs/modal-qwen-batched-validation/report.md).

## Exact task diversity

A CPU-only audit canonicalized each pair to both-unresolved evidence in XY order, retaining candidate table order and offered queries while excluding hidden truth, seeds, item IDs, and action menus. Each family had 100 distinct visible pair signatures out of 100 assigned pairs, with no exact overlap with the serial 2,048-token validation. This does not establish semantic diversity or held-out-template transfer: both families still use narrow fixed generators and wording. [Audit and definition](qwen-batched-task-diversity.json).
