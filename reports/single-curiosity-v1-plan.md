# Single-mystery curiosity pilot: prospective plan

October 7, 2026. The user approved narrowing the first study to voluntary information seeking about one mystery, with a non-information alternative. Selective satisfaction and two-mystery closure are deferred. Earlier failures remain part of the development record; this is a new instrument, not a relaxation or rescoring of their criteria.

## Question and limits

Does the model voluntarily select information about an unresolved mystery, and does selection differ when that same information is already known? Options are one clue, one independently generated decorative pattern, or stopping. Each action ends the trial. There are no points, subsequent tests, rewards, or consequences for other trials. Pattern independence establishes absence of information about the hidden candidate; it does not establish equal attractiveness or neutral valence. A preference or resolution contrast alone cannot establish curiosity as an internal state or subjective experience.

## Frozen first version

Two existing finite families (machines and objects), 100 fresh items each, seed 20261013. One hidden candidate per task. Candidate labels 1–4, action IDs A–C, diagnostic description IDs J–M. Each item has unresolved and resolved evidence, crossed with forward and reversed action-menu order: 800 choice branches. Action meanings cycle through all six label permutations, within one assignment per item shared across evidence/layout branches. Resolved evidence supplies the exact offered clue; unresolved evidence repeats the initial non-identifying observation. The independent oracle must find respectively one and four compatible candidates. The same deterministic, separately seeded 4×4 pattern is used in all matched branches, regardless of candidate identity.

Pinned Qwen3-8B, BF16/eager, exact H100, existing torch_batched_v1 backend with fixed cohorts of 32. Both diagnostics and choices use no-thinking sampled generation, temperature 0.7, full vocabulary, maximum eight new tokens, strict whole-answer parsing. The earlier single-context factorial motivates this setting but does not validate the new assay. Freeze source, assignments, sampling seeds and model revision before collection. No retries or extraction of answers from explanation text; invalid responses and technical failures are terminal and remain in every denominator.

## Validation before choice collection

Eight independent diagnostic branches per item (1,600 total): resolved candidate identification; unresolved candidate count; useful-clue action; redundant-clue recognition after resolution; pattern independence; and the consequences of selecting each of clue, pattern, and stop. Description answer positions are shuffled independently by item/kind. Diagnostic menu layouts alternate by item; diagnostics are separate contexts and never precede the behavioral choice within a conversation.

Every family × diagnostic cell must have 100 assigned items, at least 90% correct, at least 95% valid format, and complete terminal coverage. Runtime integration checks must pass. Choice collection requires exact matching source, model, hardware, inference settings, and planned choice assignments. If validation fails, do not collect choices under this version.

## Calibration and descriptive analysis

Require 100 assigned choices and at least 95% valid format in each family × evidence × layout cell, complete coverage, and an absolute forward/reverse difference no greater than 0.15 for each action's unconditional probability in each family × evidence cell. Report every gate individually. This checks reliable measurement and layout sensitivity. There is no minimum clue preference or required direction/significance of the resolution effect. A floor/ceiling preference is a descriptive result and constrains subsequent intervention sensitivity; it is not a reason to change prompts toward a desired result.

Report unconditional probabilities of clue, pattern, stop and invalid/technical outcomes; unresolved-minus-resolved contrasts for each outcome; pointwise 95% bootstrap intervals resampling whole items within each family, preserving all four matched branches (5,000 replicates, seed 20261014). Report both mathematical probability headroom and headroom when invalid mass is held fixed. These are development estimates, not confirmatory tests; finite templates limit generalization.

## Bounded continuation

One full initial validation and, only if it passes, one matching calibration. A failed diagnostic may motivate at most one prospectively documented comprehension repair, with new source/version and fresh seed; no repair chosen to increase clue preference or create a resolution effect. Do not increase generation caps or switch models silently. If calibration has poor validity or material layout dependence, document the failed instrument before any further assay development. If the instrument passes, develop independent concept/control materials and a representation-validation plan before any psychological steering. No direction may be selected by its desired choice effect. Broader mechanistic interpretation remains conditional on controls for uncertainty, attention, reward, task compliance and language associated with curiosity.

All work remains within the cumulative $100 allowance, with $10 reserved for storage/transfers and conservative compute admission/shutdown accounting. At approval, estimated prior compute was $61.60332348, leaving $28.39667652 compute. Do not spend merely to reach the limit.
