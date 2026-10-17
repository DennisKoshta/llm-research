# Epistemic readout qualification v1

October 15, 2026. Prospective materials and scoring specification for **48 unsteered generations**. This document is not a frozen execution or paid admission. The finite pain-method control closes first. No native direction has been fitted or selected. This study asks whether the proposed reports are usable measurements; it does not test curiosity, satisfaction, steering, or subjective experience.

The original target remains changing voluntary seeking while beliefs, confidence and competence are preserved. Passing these long-reasoning reports **does not admit the previously sketched 792-generation causal panel**. Before causal admission, the program would additionally need an independently validated passive epistemic measurement on the actual treated choice trajectory, or a validated shared-state short-readout fork. Independently reasoned diagnostic branches can reconstruct facts from an intact prompt and conceal an earlier appraisal change. This limitation cannot be removed by labeling a new direction “native.”

## Fixed materials

Use only object world slots **0, 1, 2, 3** from the immutable `qwen32-voluntary-resolution-v2` manifest, digest `b2ef2c264176003c9a0bcedf03a3e7bd80668cef38167d29f9ce3d9005861ab6`. Their known-property indices are respectively **0, 1, 0, 1**: both color and shape are covered. The source revision is `voluntary-resolution-v2`, with source digest `88a91e516e2580b23b9468bb1d38ba1c8a2b1e818e5f4e363ded74874a666623` and archive SHA-256 `042a017d07d97ee149a7648608d7c51aeac4ef3f2f6d241f83179249ffe9a174`.

Select the existing choice/menu-0 assignment for each world under decisive evidence and redundant evidence. Preserve its system message and entire evidence context through `End of record.` from the frozen v2 task source. The context already says every row was equally possible initially, observations are accurate and noiseless, and the hidden candidate does not change. Thus the normative posterior is uniform on the candidates compatible with **both** supplied observations: one candidate after decisive evidence, two after redundant evidence. This is not an assumed subjective prior.

Append one neutral question after the common marker. The [draft builder and scorer](../scripts/prepare_epistemic_readout_qualification.py) binds each source assignment, context, new request, oracle, sampling seed and answer ordering. Offline tokenizer preflight must establish exact equality of the original and new token prefixes through the marker; textual similarity is insufficient. No target value of the still-unknown property is disclosed in the suffix.

| Readout | Required final response | Oracle |
|---|---|---|
| Remaining candidates | JSON array containing every still-possible candidate label, without duplicates | Complete compatible subset of W, X, Y, Z |
| Candidate distribution | JSON object with exactly W, X, Y, Z as keys and numerical probabilities | Probability 1 on the resolved candidate, or 0.5 on each of two compatible candidates |
| This report's incremental information | One letter A–D | Whether a newly requested report adds a previously unestablished property value or repeats an already-established value |

The report question explicitly occurs **after all evidence already in the record**. Its four options describe new information, repeated information, a decorative pattern, or no supplied item. The oracle independently partitions currently compatible candidates by the offered report value; its conditional information gain is 1 or 0 bit. The categorical answer is not a continuous report of subjective expected information gain.

There are two fixed answer orderings. For sets and probabilities, the displayed candidate order is W/X/Y/Z or Z/Y/X/W; response array/key order is not scored. For incremental information, the semantic options rotate by world slot and reverse under the second ordering, balancing correct A–D labels. Both orderings share the same row-local seed for a given world/state/readout. They are paired format checks, not additional independent worlds.

The complete crossing is **4 worlds × 2 evidence states × 3 readouts × 2 orderings = 48 responses**, with 16 per readout and eight distinct evidence contexts. A deterministic assignment-only shuffle with seed `2026101551` creates one 32-row cohort and one 16-row cohort. No branch is selected from its answer. Machines and independent-novelty delivery are outside scope; no generalization to those conditions is claimed. These object worlds and their earlier behavior have already been inspected and remain development material.

## Frozen scoring proposal

Use the same closed-thinking final-section convention as v2: the final answer is the stripped text after the last `</think>`. A missing delimiter fails. Parsing does not remove code fences, explanations, boxed answers or trailing punctuation. JSON duplicate keys, duplicate set members, unknown labels, booleans masquerading as probabilities, NaN/Infinity, incomplete distributions and values outside [0,1] fail. A probability sum must be within **0.000001 of 1**; no renormalization is performed. Decimal comparisons of parsed numerical values avoid binary-float rejection at exact declared boundaries.

Generation evidence must be terminal: EOS, or the full 4,096-token cap. A shorter non-EOS output is unavailable. Reaching 4,096 without EOS is capped and blocks passage; EOS at exactly 4,096 is not a cap. Missing and execution-failed assignments remain in the 48-row denominator. Grammatical validity, factual correctness and distribution quality remain separate fields.

All of the following are required:

- All **48** responses are observed, thinking-closed, strictly valid and uncapped.
- All **16** remaining-set responses exactly equal the compatible set; all **16** report-information judgments are correct.
- Each of the **16** probability allocations has total-variation distance from the normative distribution at most **0.05**, and total mass on incompatible candidates at most **0.05**. Here TV is one half of the sum of absolute differences across all four candidates.
- The maximum assigned probability is at least **0.95** for every resolved allocation and lies in **[0.45, 0.55]** for every unresolved allocation.
- For each of the **eight** world/state pairs, the two answer-order allocations differ by TV at most **0.05**.

No average can hide a failing state, world or ordering. Export all individual allocations, errors and ordering differences. Do not convert a probability report into simple answer accuracy or silently discard invalid rows from its denominator. There is no rescored primary result or wording retry. Any explanatory content sensitivity, if later requested, must remain labeled descriptive and cannot rescue these gates.

## Smallest execution and stopping contract

Use pinned `Qwen/Qwen3-32B` revision `9216db5781bf21249d130ec9da846c4624c16137`, the verified v2 numerical/source environment, thinking enabled, sampled temperature 0.7/top-p 1.0, and a 4,096-token cap for every branch. The existing `BatchedHFModel.infer_batch` can consume these requests directly with `labels=''`, suppressing irrelevant initial-token label scores. There is no task-dispatch change, intervention, activation capture, receipt delivery, model training or new general runner. A thin worker must retain exact rendered tokens, seeds, cohort companions, raw outputs and runtime/cleanup evidence; an independent scorer checks the oracles and all assigned results.

The 48 scientific requests have no hooks, capture or interventions. Before them, the inherited `verify_model` and `verify_batch_model` routines run separate fixed synthetic sham, nonzero and cleanup checks; these startup probes are neither scientific observations nor direction selection.

The prospective ceiling is **one H200, two CPUs, 48 GiB RAM, 1,800 seconds total wall plus 120 seconds reserved teardown, and $3**, with no paid retry. Startup, inference and export count within the wall. Final source, materials, numerical identity, cost admission and shutdown requirements must be separately frozen and reviewed before any launch.

Failure closes this wording/format qualification without prompt repair or another allowance. Passing supports only accurate, ordering-stable **elicited normative reports on these eight contexts**. It does not establish general calibration, population equivalence, a validated internal confidence signal, contemporaneous belief preservation, or causal admission. Four worlds and two natural posterior states cannot support those stronger claims.
