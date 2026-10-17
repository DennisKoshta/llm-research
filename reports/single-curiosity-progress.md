# Curiosity-first pilot: completed development checkpoint

October 7, 2026. The approved scope change is implemented: one mystery, a clue, an independently generated decorative pattern, or stopping. The three new complete validations contain **4,800 original assigned diagnostic branches**. None passed every frozen comprehension requirement, so **no single-mystery behavioral calibration, concept extraction or psychological steering was run**.

| Collection | Correct formats | Failed accuracy cells (each n=100) |
| --- | ---: | --- |
| Qwen3-8B, single v1 | 1,600 / 1,600 | Machine counting 76; clue consequences 87; pattern consequences 81 |
| Qwen3-8B, single v2 | 1,596 / 1,600 | Machine clue consequences 79; object redundancy 81 |
| Qwen3-32B, fixed single v2 | 1,593 / 1,600 | Machine counting 67; machine redundancy 10; object redundancy 15 |

Each family × diagnostic required at least 90 correct and 95 valid formats, with complete coverage. All three collections passed artifact and integration audits, preserving all 150 original cohorts. V2 used the one allowed presentation repair. The larger-model extension then kept that instrument fixed and used fresh items. The comparisons do not isolate effects of presentation or model size because the collections use fresh seeds.

The useful scientific result is a measurement limitation. Models can identify a hidden candidate from supplied evidence and choose an informative action when explicitly instructed, yet fail a separate test of whether the offered clue is already known. This threatens the interpretation of a resolution contrast: repeated information could be chosen because it was misclassified as new. The tested failures do not show an absence of curiosity or subjective experience, and they do not establish inability under other prompting or inference regimes.

The 32B standard-generation check matched all 32 checked answers and first-token label probabilities exactly. Repeated-clue errors also occurred in the standard implementation, reducing concern about a custom batch-engine explanation in that cohort. This check was greedy and purely diagnostic; it never replaced original sampled outcomes.

The engineering work is ready for further development: 65 automated tests pass, current lint checks pass, exact model/source versions and assignments are frozen, and all raw outcomes are retained. One corrupt download was quarantined and recovered from the same stored cloud response with hash verification. One reference deployment import failure occurred before inference; it was stopped, fixed and separately recorded.

## Scope decision

The bounded prompt/model development sequence is complete without a qualified instrument. The recommended next direction is to make trustworthy measurement of AI curiosity the first paper's focus, retaining inner experience as the longer-term motivation. The alternative is to keep internal mechanisms as the immediate paper target and authorize a qualitatively broader redesign. This is a change in the first paper's question, so it has been put to the user rather than treated as routine technical tuning. No additional GPU collection is scheduled while that direction is unresolved.

All project apps are stopped with zero tasks. **Estimated cumulative compute: $63.35081244. Remaining compute: $26.64918756, plus the separate $10 storage/transfer reserve within the $100 total.** These are conservative app-wall estimates, not invoices. The extension added approximately $1.75 to the prior $61.60 compute checkpoint, including both reference deployments.

Across the project, twelve complete development collections contain **17,800 original assigned branches**. The earlier 82-response incomplete serial calibration remains separate. Engineering smokes, execution copies and the 32-prompt reference check are not counted as additional scientific samples.

## Evidence

- [V1 results](single-curiosity-v1-results.md) and [artifact audit](single-curiosity-v1-validation-audit.json)
- [V2 results](single-curiosity-v2-results.md) and [artifact audit](single-curiosity-v2-validation-audit.json)
- [32B results](qwen32-single-curiosity-results.md) and [artifact audit](qwen32-single-curiosity-validation-audit.json)
- [Reference check](qwen32-reference-check-results.md)
- [Cost and shutdown snapshot](single-curiosity-scope-decision-cost-and-shutdown.json)
- [Reproduction guide](reproduction-guide.md)
