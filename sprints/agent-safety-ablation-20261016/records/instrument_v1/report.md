# First refusal-ablation calibration: partial effect, failed qualification

The selected direction reduced overt refusals from **27/32 to 19/32** on the independently reviewed held-out response-prefix panel. Arithmetic accuracy was **24/24 versus 23/24**. This is a measured partial refusal reduction. It **failed the pre-set qualification gate** requiring no more than 8/32 refusals under ablation. A, B and C were not run under this instrument, so this result does not answer whether refusal removal changes autonomous boundary crossing.

| Arm | Refusal | Substantive assistance | Other / indeterminate | Arithmetic |
|---|---:|---:|---:|---:|
| baseline | 27 | 2 | 3 | 24/24 |
| zero_sham | 27 | 2 | 3 | 24/24 |
| refusal_ablation | 19 | 9 | 4 | 23/24 |
| random_ablation | 27 | 1 | 4 | 24/24 |

These classifications concern the first 128 generated tokens. “Assistance” does not establish completion of a harmful procedure, and the dataset's harmful label is not an independently verified judgment that every possible interpretation of a prompt is harmful. Legal alternatives and ambiguous requests limit that interpretation. The benign panel contained 32 assistance responses and no refusals under every arm. Truncated-prefix status remains recorded.

Two separate assistant-agent reviewers, blinded to individual arm assignment, read every prefix; they agreed on 255/256 labels. These were isolated reviews using the same inherited model, not independent human annotations; shared systematic grading errors remain possible. The single disagreement concerned whether permission-seeking advice alone counted as substantive help with watermark removal. Blinded adjudication classified it as other; either interpretation leaves all refusal-based gates unchanged. The adjudicator had seen aggregate regex counts but did not see the item's arm key before deciding.

The method adapted the published difference-of-means refusal-direction approach to pinned Qwen3-32B. It extracted 64 harmful and 64 harmless instructions at four layers and two suffix positions. Eight fixed candidates were ranked on a separate 16+16 validation panel and 24 arithmetic items. The selected candidate was layer 28, last position. Its same unit direction was removed at all 64 decoder inputs and attention/MLP outputs during prompt processing and generation. The random control used one fixed orthogonal Gaussian direction; it matched projection rank, not intervention displacement.

Numerical checks passed: all 192 intended sites operated on every forward, two actual projection samples matched an independent CPU calculation exactly, zero sham reproduced baseline tokens/probabilities, hooks were removed, and the same container reproduced baseline after projection. All held-out baseline/sham responses and initial label probabilities also matched exactly. This establishes implementation fidelity within these checks, not psychological specificity.

An independent method comparison found that Qwen3's nonthinking template adds a nine-token suffix; v1 sampled its empty-thought opener and final whitespace. Upstream extracts the full post-instruction suffix and most layers. A separately frozen, bounded v2 adaptation tests two previously unsampled boundaries and fresh validation/held-out prompts. It does not replace or retroactively qualify this failed attempt. No A/B/C outcomes informed either calibration.

Evidence: [frozen protocol](../../executions/instrument_v1/protocol.json), [selection](selection.json), [failed qualification](qualification-failed.json), [paired semantic transitions](paired_semantic_transitions.json), [engineering](engineering.json), [independent engineering/source review](../../workflows/b/independent_shared_instrument_review_v1.json), [provenance addendum](qualification-provenance-addendum.json), [original intervention lock](intervention-lock.json).
