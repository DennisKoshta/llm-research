# Choice support readout: offline feasibility v1

Status: **prospective specification awaiting root's immutable admission; no fit has been run.** October 15, 2026. This is one no-spend falsification screen for a possible measurement instrument, using only the closed voluntary-resolution v1/v2 collections. The document alone authorizes no fit, model inference, new collection, native curiosity/satisfaction fit, intervention, or causal admission. The active coverage-bridge collection and its outcomes are outside scope. Root remains the sole freeze/admission operator; a separate immutable manifest must admit any execution.

The question is narrow: can one fixed linear readout recover **which displayed candidates remain compatible with the supplied evidence** from a late state on the actual optional-choice trajectory? Success would justify considering a prospective measurement qualification. It would not establish internal beliefs, confidence, information gain, motivational regulation, or experience. Failure closes this decoder specification; it does not establish that the model lacks the information.

## Closed sources and fixed rows

Use all object-family optional-choice assignments from [v1](../runs/qwen32-voluntary-resolution-v1/manifest.json) for fitting and all corresponding assignments from [v2](../runs/qwen32-voluntary-resolution-v2/manifest.json) for one evaluation. Each split has 12 exact semantic worlds × three deliveries (`decisive_evidence`, `redundant_evidence`, `independent_novelty`) × two menu layouts = **72 assigned rows**. World slots 0–11 are fixed; select by assignment fields, never by answers, activations or decoder performance. Do not add diagnostics, machines, selected-receipt states, other collections, or new seeds. Retain each menu/delivery row rather than averaging its different post-reasoning state. Every world has equal total weight; every delivery and menu has equal weight within it.

The independent units are at most **12 worlds per split**, not 72 examples. All copies of a world stay together; there is no row-wise random split or cross-validation search. Both collections, their behavior and earlier candidate projections have been inspected. Their complete two-family readiness gates failed; the already selected object-only development scope does not repair those failures. See the [v1 results](voluntary-resolution-v1-results.md), [v2 results](voluntary-resolution-v2-results.md), and [earlier feasibility note](native-resolution-contribution-v1-feasibility.md).

Input-only verification for this draft found 72/72 available, finite layer-57 decision captures in each split, each with one closure and a valid uncapped source response. Each split has 24 singleton-support rows and 48 two-candidate-support rows, and six worlds for each initially known property. These are inventory facts, not decoder results. No projection, feature selection, regression, prediction or performance calculation has been run.

Exact semantic-world IDs do not overlap, but named tables do: v1 has nine unique named tables, v2 ten, and three are shared. One **visible support problem** also overlaps: v1 world 9 and v2 world 3 have the same named table and the observed fact `shape = diamond`. It contributes four v2 rows: redundant/independent-novelty × both menus. The same display support is W,Y even though W/Y's named candidate contents exchange order. This is not fresh or independent table generalization.

## Oracle and candidate coordinates

For each row, independently parse the actual rendered prompt's four-row `Candidate | ...` table and explicit property observations before `End of record.`. The initially supplied fact and a decisive/redundant delivery constrain support; the independent decorative pattern does not. Enumerate each table row against the conjunction of observed attribute/value facts. Cross-check this enumeration against the matching archived task module and saved task oracle. An inconsistency is an integrity failure, never a label to repair from the generated answer.

The target is `y = [1(W compatible), 1(X compatible), 1(Y compatible), 1(Z compatible)]`, in **displayed W,X,Y,Z order**. Do not sort the target bits by property value or hidden truth. Displayed row ordering must remain bound to its prompt; semantic canonicalization is only for grouping and exposure assessment. A,B,C,D are action labels and are not these coordinates. The saved `decision_label_logits` and label probabilities describe actions; they are not candidate posteriors and are not decoder inputs or targets. The oracle uses supplied evidence, not an unobserved realized truth or any later selected report.

For exposure sensitivity, define a visible-problem object with exactly `attributes` (sorted names), `candidates` (rows reordered to those names, then lexically sorted), and `observed` (sorted unique explicit `[attribute_name, value]` pairs). Hash its canonical JSON. Ignore hidden truth, row labels/order, menus, action mapping, delivery wording and irrelevant pattern content. Build the seen set from all 72 v1 inputs. The sole shared key is `5e85fe2ddf66df3c53d08d031bc5c720402e6ead207a6cbe69ac2b26f68c5d3c`.

## Fixed capture and timing limitation

Use only the zero-indexed **layer 57**, `decision` vector: 5,120 features stored as float32 captured activations. The archived arrays have shape `[32,4,7,5120]`, layers `[15,31,47,57]`, and positions `record`, `prefill`, `token_32`, `token_128`, `token_512`, `think_end`, `decision`. Bind each selected trial to its original NPZ row, token position and availability mask. No layer average, alternative position, normalization by answer length, or C/S component projection is allowed.

The frozen `ReasoningTrace.after_sample` saves the preceding forward buffer after it observes the first nonwhitespace sampled token **or EOS** after a closing `</think>`. Every subsequent closing delimiter clears the saved decision, so the retained capture is relative to the **last** closure. Its hidden state precedes the sampled token, but selecting that state uses sampled-token and later-closure history. This is a retrospective event annotation, not an intervention boundary knowable beforehand. In these closed inputs each selected row has one closure; the methodological limitation still applies. A future first-closure capture would be a different prospectively defined instrument requiring separate engineering and qualification.

The state may encode intended action, answer preparation, generated text, formatting or lexical identity. Identical initial prefixes do not acquire native record-state variation merely from different sampling seeds; post-reasoning variation comes with differing generated histories. This screen measures neither native C variation nor its relation to optional behavior.

## One fixed estimator

Use CPU float64 arithmetic for fitting and prediction. Let `X` be the 72 × 5120 v1 matrix and `Y` its 72 × 4 binary support matrix. Compute **only on v1** the feature mean `mu`, target mean `ybar`, and scalar `s = sqrt(mean_i(sum_j((X_ij - mu_j)^2)))`. Require finite positive `s`; otherwise return unavailable. Set `Z = (X - mu) / s` and `Yc = Y - ybar`. There is no feature-wise variance scaling, PCA, whitening, sparse selection or fitted nuisance removal.

Fix `lambda = 1/12` before fitting. Solve once:

```text
B = solve(Z @ Z.T + 72 * lambda * I_72, Yc)
W = Z.T @ B
score(x) = ybar + ((x - mu) / s) @ W
predicted_bit_j = 1 if score_j >= 0.5 else 0
```

This minimizes mean squared four-output error plus `lambda * ||W||_F^2`, with the intercept unpenalized. Scores are unrestricted regression outputs, **not probabilities**; do not clip, renormalize, calibrate, choose the top-k entries, or use the oracle cardinality to repair predictions. Empty, three-candidate and four-candidate predictions remain predictions and ordinarily fail exact-set scoring. A numerical failure produces an unavailable result, not a search over regularization or solver perturbations.

There are 20,480 nominal slope coefficients, but the centered training matrix has rank at most 71. Scalar normalization gives `trace(Z @ Z.T) = 72`; therefore the ridge effective degrees of freedom `trace(K @ inverse(K + 6I))` are at most 12 **per output**, plus its intercept. Report the realized rank and effective degrees of freedom without using them to choose another estimator. This shrinkage does not manufacture independent observations: four responses share only 12 world clusters, and high feature count, correlated rows and the small development split remain substantial overfitting risks. Poor performance can reflect either an unsuitable estimator or inaccessible support at this site.

## Shortcut comparator and once-only evaluation

Fit one label-only shortcut on v1: for each oracle cardinality 1 or 2, choose its most frequent **complete support mask** among the 72 training labels; resolve ties lexicographically on the four-bit string. At evaluation this comparator receives the true cardinality, an intentionally favorable allowance, but no candidate identity evidence. Its masks and frequencies are fixed before v2 prediction. Do not use v2 frequencies to select a comparator or alter the decoder. Also report the decoder's cardinality accuracy separately; getting the number of possibilities right is insufficient.

An input-only feasibility check gives singleton shortcut `0100` (X) and pair shortcut `0110` (X,Y; the lexicographic winner of a three-way training tie). Their v2 label matches are 6/24 and 12/48 respectively, both 25%. The required improvement below is therefore arithmetically attainable. This calculation uses oracle labels only and is not a decoder result.

Publish all 72 v1 and all 72 v2 row predictions, flags and targets. Training metrics are descriptive only. Evaluate v2 once using the following fixed gates, all required for `instrument_feasibility_passed`:

1. Provenance, grouping, labels and all selected capture bindings pass. Missing/invalid handling follows the rule below.
2. Exact-set accuracy is at least **90% in each delivery × menu cell**: at least 11/12 in all six primary cells. Integer counts, not rounded percentages, decide the gate.
3. The input-defined **no-visible-overlap sensitivity also passes the same 90% floor**: decisive cells require 11/12 each; redundant and independent-novelty cells require 10/11 each. Its denominator is 68, with all four overlapping rows still present in the primary report. This mandatory sensitivity cannot rescue a failed primary result or justify replacing a world.
4. Decoder exact-set accuracy exceeds the trained oracle-cardinality shortcut by at least **25 percentage points separately in both cardinality strata**: at least six additional correct sets among the 24 singleton rows and twelve among the 48 pair-support rows. These are paired all-assigned count differences; unavailable predictions receive no credit.
5. Evidence sensitivity is correct for at least 11/12 worlds in each menu: both decisive and redundant predictions equal their respective oracles, with the decisive set a proper singleton subset of the redundant set. Irrelevant-pattern invariance is correct for at least 11/12 worlds per menu: both redundant and novelty predictions equal their common oracle. Menu invariance is correct for at least 11/12 worlds per delivery: both menu predictions equal the same oracle. Report these paired counts and their input-defined nonoverlap counterparts; only the explicitly listed full-split paired thresholds are additional gates.

Report per-bit confusion counts, all six cells, singleton/pair strata, known-property roles, shortcut scores, paired sensitivity/invariance, and the number of worlds whose six sets are all correct. Preserve world membership in every summary; do not treat repeated rows as independent observations. The fixed nonoverlap sensitivity retains world 3's two decisive rows while omitting its four overlapping unresolved rows from that sensitivity only. No p-value or population-accuracy claim follows from these small fixed thresholds. In particular, 11/12 observed worlds is not a demonstrated 90% population lower bound.

## Missingness, integrity and immutable reporting

Keep all 144 assigned rows in the inventory, including any invalid/capped/unclosed source response, unavailable decision state or execution error. Source answer validity is not a criterion for choosing fitting rows. If an invalid response has the designated finite state, retain it for decoding and report its original flags; correct support prediction does not repair that answer. Missing v1 features make the fixed 72-row fit unavailable rather than authorizing imputation or reduced fitting. Missing v2 features are unsuccessful predictions in every applicable denominator and fail the complete-capture integrity requirement. Never fill a missing menu/delivery state from a companion, reuse diagnostic states, or discard a difficult world. This rule is fixed despite the present inventory finding no such cases.

Before any future fit, root must freeze this plan plus a small implementation/tests and a machine-readable ledger of exact assignments, oracle labels, row/artifact bindings and input file hashes. Reverify hashes before and after use. Load version-specific source from its hash-verified archive, not current task code. Produce one immutable output directory containing the estimator, normalization, all predictions, metrics, each individual gate, software/code hashes and input ledger. Do not inspect intermediate v2 performance and then refit. A genuine implementation defect would require a separately documented revision preserving the original attempt; failure offers no automatic alternate layer, position, decoder, threshold or collection.

The separate root manifest must bind `admitted: true`, the complete input ledger and its digest, and exactly one project-relative output destination. The ledger binds this plan, producer, independent auditor and both test files by stable logical paths and byte hashes, so copying unchanged code into a frozen execution directory does not change its identity. Preparation reads capture headers/metadata only; feature values become accessible only after exact admitted-ledger reproduction. An unavailable fit retains all 144 prediction slots with explicit unavailable scores and produces **no model NPZ**, rather than surrogate weights. An attempt marker and exclusive directory creation prevent reusing the admitted destination, including after failure.

## Verified source bindings for the future freeze

Both sources use `Qwen/Qwen3-32B`, revision `9216db5781bf21249d130ec9da846c4624c16137`, with original 4096-token generation. Both original audits are complete and integrity-passed; their readiness failures remain unchanged. Canonical JSON digests below are distinct from file-byte hashes.

| Binding | v1 fit | v2 evaluation |
|---|---|---|
| Manifest canonical SHA256 | `c33d76eba3931aeacadf68f6282c7c1471be8edccec83e7ca2f37ab076e68feb` | `b2ef2c264176003c9a0bcedf03a3e7bd80668cef38167d29f9ce3d9005861ab6` |
| Frozen source SHA256 | `7e005358661c0de588faba66c9e9e5384984810abbc27385351ca5a2bdf8a911` | `88a91e516e2580b23b9468bb1d38ba1c8a2b1e818e5f4e363ded74874a666623` |
| Source archive byte SHA256 | `23254d727f0788da9a353f36c9adaf51e92aefaed19f8ecec4e96b131a2b79cf` | `042a017d07d97ee149a7648608d7c51aeac4ef3f2f6d241f83179249ffe9a174` |
| Original audit byte SHA256 | `d1a4bfdbdc68ee1ce63d18709c6640d805d6fa065623f3c8ea0b452c3d9bf4bf` | `aeec95021b15206e992ecf2a29655d128860b1c2230754d99d8fcac767a3412b` |
| Sorted 72-assignment digest | `1052a14bf9e73cd00697e51e4c1602b1f026ac4a4ef0f00c7f120340997b289e` | `adde817547c7f6dee5a125fa616a3cf1cb98a56708ec89e01280ce05f0f6330d` |
| Oracle-label ledger digest | `b7fd0e2336cb8091c8e74d769d1e03bc132f2b88dea8826923a308d63eaa6165` | `4508c0497c0b3a271c7c40ec2a21f0f99a60bd365661fa81e5a4cb1ca1f742e6` |
| Selected capture-row ledger digest | `2277f187965240341451ebabbfe5ad871e3d14ff957e5e24054dcf66323768ca` | `a0c2f530df644c2994a92869dd41a5836ef6c977496a7350f1bd65a30e6ab0db` |
| 22-file inventory digest | `c6c5635101a71b4c324ee5a380686cb9f7474c552a89060176d120acb408c70f` | `aaf41f020927602e2326ce00a2c30dedff991b4e002d0f98afc0da7fe35b5bc6` |

Sort selected rows by world slot, the delivery order stated above, then menu 0/1. A label-ledger row has `trial_id`, `assignment_sha256`, `world_slot`, `semantic_world_id`, `delivery`, `menu_layout`, `candidate_labels=[W,X,Y,Z]` and `compatible_bits`. A capture-ledger row adds `record_sha256`, `activation_path`, `activation_byte_sha256`, `activation_row`, `layer=57`, `position=decision`, `available`, `token_position` and `closure_count`.

The 22-file ledger is the canonical digest of a project-relative-path → byte-SHA256 dictionary: manifest, freeze, source archive, original audit, all nine cohort JSONs and all nine activation NPZs. Cohort/activation attempts are v1 `81fc07b16979410eba445ebb7a873ef8` and v2 `eeb0e3a93d644bf188d4c0dcfb1eee88`, offsets 0–256 in steps of 32, with seven-digit filenames. The original audit paths are [v1](qwen32-voluntary-resolution-v1-audit.json) and [v2](qwen32-voluntary-resolution-v2-audit.json). This binds the complete source capture inventory, not just favorable selected rows.

Archive bytes match the corresponding execution copies for the task, world generator, thinking dispatcher, batched generator and trace module. The two archived trace modules have identical byte SHA256 `1ede8ecfbcf1aa350919d933a0b47611d9ae8121d00d2d0b4abe3fed0c6c1e59`. Archived task hashes are v1 `eda9deec8a446c7af317ae3887c95c3d5af74ddd30b304d72646d4a4f0f8dc0e` and v2 `482af2d849d7bafa7ebbabbee4a39cb16745a326bf5e7cd057da0c2081b8b874`; using v2 to reconstruct v1 is prohibited.

Even a full pass establishes only a retrospective accessible-support instrument on previously exposed, unsteered development data. Treated-distribution reliability, contemporaneous belief use, native C/S variation and a prospectively triggerable measurement boundary remain unvalidated. It cannot rescue the [failed elicited-readout qualification](epistemic-readout-qualification-v1-results.md), replace the original research question with an appraisal project, or admit a causal study.
