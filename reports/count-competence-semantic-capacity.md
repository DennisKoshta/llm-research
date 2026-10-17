# Count-competence development and semantic capacity

October 14, 2026. This is a prospective, assignment-only engineering and design assessment. No count collection was prepared, frozen, or run for this assessment, and no count response outcomes were inspected.

## Development preparation

The count preparer now has an explicit `development` mode. It fixes 16 sampled world indices per family, two families, and all six correct-count × query cells per world: **32 index worlds and 192 baseline branches**. Every branch receives the same 4,096-token maximum. The manifest declares `phase=development_validation` and `inference_scope=development`.

The development instrument must pass **12 separate family × correct-count × query gates**, each with 16 assigned branches. The thresholds remain accuracy ≥90% and validity ≥95%; at this sample size they require at least **15 correct and all 16 valid responses in every cell**. Missing responses remain in the assigned denominator and prevent completion. The evaluator also requires all six cells in each assigned world and unique assignment IDs. Pooling the two query cells cannot rescue a failing query-specific gate.

The legacy validation path retains its minimum of 100 world indices per family and its existing six family × count gates. That path's raw index count is not certification of semantic independence. A future confirmation preparation needs the prospective plan's semantic-uniqueness, nonoverlap, and query-specific qualification requirements; the existing legacy preparer must not be presented as implementing them.

Six branches per world and 32 branches per baseline cohort require source blocks of `LCM(6, 32) = 96` branches. The development source therefore has two complete blocks and defaults to two workers, with no world split between workers. Four later natural conditions—baseline, zero sham, and information ±D—would produce **768 continuations**. This preparer does not create or launch that later intervention schedule.

## Semantic identity and finite capacity

For this count assay, an underlying semantic world is identified by its family, named properties, unordered set of candidate rows, and realized true candidate. The true candidate matters because the one-candidate cells reveal both of its properties. Sorting named properties and candidate rows removes display permutations. A new index, seed, template, candidate-row order, or query does not itself create a new semantic world.

The two-candidate cells can observe either of the two properties. This produces two six-cell assay variants for a single underlying world. These variants share the underlying world and must not be counted as independent worlds for the planned noninferiority guarantee.

The current held-out object generator has four color pairs and four shape pairs. Each pair combination defines four candidates, with four possible realized candidates. Its exact underlying capacity is therefore:

`4 color pairs × 4 shape pairs × 4 true candidates = 64 worlds`.

There are at most 128 variants if the partially observed property is included. Even counting these dependent variants would not supply 120 confirmation variants plus 16 disjoint development variants. Resampling indices cannot solve this limitation.

The machine generator chooses two distinct named inputs from nine and an unordered pair of outputs from 20 for each named input, then one of four true candidates. Its underlying capacity is:

`C(9, 2) × C(20, 2)^2 × 4 = 5,198,400 worlds`.

This is a combinatorial upper capacity, not evidence that a particular sampled collection is unique or that it avoids earlier inspected worlds. The preparation metadata inventories canonical identities and duplicate raw indices for each selected family. It explicitly records that overlap against all inspected collections has not yet been audited and that confirmation is unready.

## Confirmation remains unready

The proposed **120 fresh semantic count worlds per family cannot be implemented with the current object vocabulary**, even before excluding development or other inspected worlds. The prospective plan's sample sizes, preservation margin, and alpha allocation remain unchanged. This finding invalidates a claim that the current generator already supports that confirmation sample; it does not justify treating repeated indices as independent or reducing the confirmation sample after results.

A later implementation can introduce a count-module-owned vocabulary parameter or a new count task revision with an independent finite-world constructor. It must leave `information_state.py` and existing frozen tasks unchanged. For example, eight disjoint color pairs and eight disjoint shape pairs would provide 256 underlying object worlds, before any exclusion ledger; larger vocabularies would give more room for previously inspected samples. Such a change needs independent oracle tests, canonical de-duplication, a ledger excluding every inspected count world, and a deterministic outcome-independent selection rule before any confirmation freeze. Named-vocabulary expansion alone does not establish structural generalization.

No vocabulary expansion is implemented here. Its detailed design can wait until the target mechanism candidate qualifies for the next stage. The development check can proceed as explicitly labeled exploratory competence qualification, with its raw-index and semantic-duplicate counts disclosed, if that stage is otherwise admitted.
