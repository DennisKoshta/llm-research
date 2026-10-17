# Refusal reduction with a fixed intervention, followed by a separate competence study

The selected refusal-targeted direction reduced overt refusals from **31/32 to 1/32** on held-out response prefixes; a random-direction control retained **31/32** refusals. The original 24-question arithmetic screen failed its fixed absolute-accuracy gate for three arms, including baseline. We retained that failure and ran one separately frozen competence study with the same intervention. It passed: baseline answered **182/192** fresh questions correctly and ablation **180/192**; the combined counts including the original screen were **203/216** and **200/216**.

This supports admission to the A/B/C task pilots. It does not establish general capability preservation or answer whether refusal suppression changes unprompted boundary crossing. No A/B/C outcomes were used to select the intervention or make this qualification decision.

| Arm | Held-out refusals / 32 | Original arithmetic / 24 | Fresh arithmetic / 192 | Combined arithmetic / 216 |
|---|---:|---:|---:|---:|
| Baseline | 31 | 21 | 182 | 203 |
| Zero sham | 31 | 21 | 182 | 203 |
| Refusal ablation | 1 | 20 | 180 | 200 |
| Random ablation | 31 | 22 | 183 | 205 |

![Fixed refusal and prospective arithmetic qualification evidence](figures/qualification.png)

Figure: descriptive counts, with no generalization interval implied. [Reproduction script](../../shared/plot_instrument_qualification_v1.py) and [input hashes](figures/provenance.json) are retained.

## What the refusal measurement establishes

Two separate assistant agents classified all 256 held-out response prefixes while blinded to individual arm assignments. They used the same inherited model family; these are not human annotations or independent model-family replications. Twelve disagreements received blinded adjudication. The adjudicator had seen aggregate regex counts and arithmetic results, but not the disputed items' arm keys. Reclassifying all disagreements as indeterminate leaves the refusal gates satisfied.

Under ablation the harmful-labeled panel contained one refusal, 28 assistance prefixes and three indeterminate prefixes. **Assistance is not a harmful-compliance measure**: the rubric includes factual corrections and safe alternatives when no refusal is present. Some prompt premises are false or ambiguous. Prefixes were capped at 128 generated tokens; many ablated responses reached that cap, so the result does not establish complete trajectories or completion of a harmful procedure. All 32 benign prefixes were assistance under every arm, with no refusals.

## Method and selection

This is an adaptation of the published [refusal-direction method and implementation](https://github.com/andyrdt/refusal_direction), with the source pinned at `9d852fae1a9121c78b29142de733cb1340770cc3`. It uses `Qwen/Qwen3-32B` at revision `9216db5781bf21249d130ec9da846c4624c16137`, BF16 inference, and disabled thinking. A difference of means was extracted from 64 harmful and 64 harmless instructions at six layers and two previously unsampled Qwen3 suffix boundaries. Twelve candidates were evaluated on 16 fresh harmful, 16 fresh harmless and 24 fresh arithmetic validation items using the frozen selection rule. The selected direction was extracted at layer 44, position −9. Extraction layer is distinct from intervention coverage.

The intervention removes the same unit direction at all 64 decoder inputs and all attention and MLP outputs, across prompt processing and generation: 192 sites per forward. Arithmetic is performed in FP32 before casting back to BF16. Model weights are unchanged. Zero sham is an exact identity; the fixed orthogonal Gaussian control matches projection rank, not the actual activation displacement. This does not reverse all post-training or restore a base model.

V1 had failed the refusal criterion (27/32 baseline versus 19/32 ablated). V2 changed the sampled suffix positions, layer grid, selected direction and validation/evaluation prompts. Its improvement cannot be attributed to the suffix change alone. Both attempts and all candidate results remain available. Evaluation instructions exclude the earlier calibration instructions, but public source categories and arithmetic task families are reused; these are not fresh domains. The v2 validation panel uses unused training-pool instructions because the original validation pool was nearly exhausted.

## Qualification history and the competence follow-up

The [original v2 qualification](qualification-failed.json) **remains failed**. Its sole failed gate was arithmetic accuracy of at least 90% in each arm: baseline and sham scored 21/24, ablation 20/24, random 22/24. All errors were actual wrong answers with valid formatting. The refusal, relative competence, and engineering gates passed.

The follow-up was a result-informed, prospectively frozen study, rather than a retroactive change to that gate. It held the selected intervention fixed, used the same arithmetic difficulty and generator, and admitted one panel of 192 fresh ordered operand pairs. The fresh panel and the pooled 216 items each had to meet the absolute and relative thresholds independently. A conservative paired noninferiority criterion also required the fresh lower 95% bound to exceed −10 percentage points. The panel contains 190 unordered pairs and two reversals of previously seen ordered pairs; this is disclosed rather than repaired after collection. No additional panel was permitted.

On the fresh panel, ablation improved zero baseline errors and introduced two errors. The observed difference was −1.04 percentage points. Separate one-sided 97.5% Clopper–Pearson bounds for improvement and regression, combined by a Bonferroni bound, give a conservative 95% lower bound of **−3.71 percentage points**, passing the pre-specified −10-point margin. This is a conditional problem-instance sampling reference, not a general-intelligence guarantee. Both fresh and pooled absolute accuracy gates also passed under every arm.

Independent reconstruction verified all 768 fresh and 96 retained arithmetic responses, actual prompts and answer oracles, matched input tensors/cohorts, strict formatting, fixed intervention hashes and all qualification calculations. Baseline and sham tokens, initial probabilities, masks and positions matched exactly. Numerical projection checks verified the intended sites, clean hook removal and independent CPU reconstruction of recorded projections. These checks establish implementation fidelity within the measured settings.

The active receipt is therefore explicitly named `instrument_v2_with_prospective_competence_extension_v1`. It admits task pilots only: A, B and C each retain their own frozen competence, formatting and provenance gates before main collection. A task's boundary violations are outcomes, not reasons to select a new direction.

## Evidence

- [V2 frozen protocol](../../executions/instrument_v2/protocol.json), [all candidate scores](selection.json), [original failed qualification](qualification-failed.json), [independent v2 reconstruction](../../workflows/b/independent_instrument_v2_result_review.json).
- [Competence extension freeze](../../executions/competence_extension_v1/manifest.json), [analysis](../competence_extension_v1/analysis.json), [independent raw reconstruction](../../workflows/c/records/independent_competence_extension_results_v1.json).
- [Active separately named qualification](../intervention-qualified.json), [immutable intervention lock](../intervention-lock.json), [admission code](../../shared/admit_combined_instrument_v1.py).
- [Retained v1 report](../instrument_v1/report.md), [numerical engineering evidence](../instrument_v1/engineering.json).
