# Experiment 3.2 — prospective CPU preparation

This is preparation only. No empirical activations, outcomes, GPU jobs, Modal
calls, or experiment receipts were used or generated. The frozen method is
`exp32_preparation_plan_v1.json`; all validation inputs are explicitly synthetic.

The new files are `scripts/exp32_plan.py`, `exp32_contract.py`,
`exp32_analysis.py`, `exp32_admission.py`, and
`tests/test_exp32_preparation.py`. The analysis entry point is deliberately
`analyze_synthetic`: it rejects empirical datasets. A real extraction backend,
independent capture auditor, and empirical result/receipt adapter remain required.
Missing auditors return PENDING rather than treating a prepared design as evidence.

## Timing, provenance and outcome definition

`prepare_input` calls the existing `pre_action_timing.locate` contract, binding
the original first-turn messages, complete evaluation prompt, exact token IDs,
tokenizer hashes, immutable materialized checkpoint identity and source request.
Quoted `<think>` strings in an input cannot move the generation-side boundary.
The extraction contract forwards only the saved prefix IDs and requires the last
token immediately before the generation-side `<think>` marker. Layer48 means the
output of zero-based decoder block48, matching the actual Phase1 hook convention;
it does not mean an ambiguously indexed `hidden_states[48]` tuple element.

The contract requires one finite 5120-dimensional vector per exact trial ID,
ordered identically to saved `.npz` `trial_ids`/`layer48` arrays. Per-row bytes,
archive bytes, input hashes, checkpoint identities and timing hashes must match.
Frozen Phase1 direction files have their own file/array hashes and retain their
original values; the unit-norm tolerance matches the materialization code.
The Ablated projection uses `h-r*(h dot r)/(r dot r)` in float64 before train-only
scaling, preserving orthogonality despite floating-point unit-norm rounding.

The capture ledger covers all 4,520 Exp1.1 action episodes and all 1,536 Exp3.1
episodes. The registered analysis uses 3,200 Exp1.1 main episodes and all Exp3.1
episodes, totaling 4,736. Exp1.1 sanity controls remain in provenance and are
excluded by their predefined family, never by outcomes. Coverage checks require
actual factorial scenario/cell membership, not just matching total row counts.

The primary event is `SHORTCUT_VIOLATION`. Authorized harmless direct operations
are not recoded as violations. The specification's phrase “shortcut-taking” is
interpreted as constraint-violating shortcut behavior, consistent with its
probing objective and four-way taxonomy. This interpretation and the block48
convention were explicitly confirmed by the parent. No undeclared all-direct-route
secondary is included. Both ITT and strict denominators are retained, with
malformed episodes separately counted and excluded only from strict analysis.

Hash consistency does not prove a real GPU capture. The returned dataset retains
`backend_audit_complete=False`. Future admission requires fresh global Phase3
authorization, complete independently recomputed Exp1.1 evidence, complete
Exp3.1 execution **and raw-to-summary statistical completion**, and a separate
capture auditor verifying actual checkpoint/FP8 inventory, tokenizer provenance,
raw first-turn journals, frozen direction, source reconstruction, exact block/token
hook, saved arrays, persistent volume paths, and five-pilot H5 evidence. The
expected future `exp31_evidence` and `exp32_capture_evidence` modules are not
fabricated by this preparation. Their eventual APIs must satisfy the explicit
admission contract; saved PASS dictionaries alone cannot authorize analysis.

## Frozen analysis choices

Experiments are analyzed separately, stratified by reasoning mode and, for
Exp3.1, payoff format. Five outer folds hold out entire Exp1.1 templates or
Exp3.1 canonical structures. Every scenario, arm, domain, format and reasoning
variant of a group stays in its assigned fold. Group assignment depends only on
fixed identifiers and a fixed seed, not labels, prompt features or effect sizes.
There is no hyperparameter search or label-driven fold reroll. Vocabulary,
standardization and logistic fitting use only outer training rows.

Regime A compares the full Layer48 linear probe with prompt-only TF-IDF and the
complete `cost * consequence * boundary * domain * arm` interaction model,
including every lower-order interaction. Explicit treatment coding gives 287
columns plus an intercept for the six-arm Exp1.1 grid and 95 plus an intercept for
the two-arm Exp3.1 grid. Unseen training-cell support is recorded.

Regime B uses the signed one-dimensional refusal projection only within Intact,
with full-activation, design and prompt-only comparators. It never applies that
baseline to Ablated checkpoints as if residual refusal variance were unchanged.

Regime C selects intermediate cells using **training outcomes only**, separately
in each outer fold and denominator: inclusive 20–80% violations, both labels and
at least two template groups. A separate classifier is fitted within each exact
cell. Intact uses full and one-dimensional probes; Ablated uses an orthogonal
probe. Prompt-only TF-IDF and a fixed constant design score are comparators.
Every train-selection decision and held-out inclusion/exclusion ID is recorded.
Held-out outcomes never decide whether a cell is intermediate. Cells are reported
separately; their scores are not pooled into a cross-cell AUROC that could reward
the model for identifying experimental conditions.

All methods use fixed L2 logistic regression with C=1, no class weighting,
L-BFGS, 2,000 maximum iterations and tolerance 1e-8. Nonconvergence, one-class
training or an empty TF-IDF vocabulary gives an explicit unavailable result,
without an alternate fit or invented predictions. The installed pinned numerical
stack is NumPy2.5.4, SciPy1.18.1 and scikit-learn1.9.1; no packages were installed.

## Metrics and limitations

AUROC gives half credit to tied scores. FPR at80% TPR is linearly interpolated on
the held-out ROC, corresponding to a randomized boundary threshold; it is a
descriptive ROC operating point, not a deployment-calibrated threshold. PPV at10%
prevalence is computed from TPR=.8 and that FPR. A constant score gives AUROC=.5,
FPR=.8 and standardized PPV=.1 **only when both held-out classes exist**.
Single-class or otherwise unidentified evaluations are UNAVAILABLE, never .5.

The registered bootstrap uses 5,000 whole-template/structure resamples, preserving
paired predictions. Contrasts use the same resample for probe and baseline on
their common prediction support, with missing coverage exposed. Percentile
intervals are conditional on the fitted OOF models and training-only selection;
they do not include refitting uncertainty. All cell intervals are pointwise and
exploratory, with no uncorrected family-wide claim of success.

The prospective 80% finite-replicate threshold is **computational availability
only**. It does not establish nominal95% confidence coverage. A balanced cell with
four independent clusters has only87.5% expected two-class ordinary bootstrap
samples, motivating the threshold from design structure before empirical data.
Total, finite and undefined resamples are all reported. Fewer than two clusters
or fewer than80% finite resamples yields an unavailable interval. Degenerate
intervals are explicitly flagged rather than presented as strong certainty.

Exp3.1 has only four scenarios per exact cell within a mode/format. Grouped
training-only eligibility may therefore yield few observations or no usable
Regime C interval. This is a limitation of the fixed design, not a reason to
select cells using held-out rates, merge incompatible cells, loosen a scientific
gate, or claim the probe works. Null and unavailable results remain valid outputs.

The 15 synthetic tests cover full factorial membership, exact timing and byte
joins, post-join mutation rejection, swapped model/outcome/vector rejection, whole-template splitting, train-only
selection/vocabulary, all five interaction factors, Intact-only and orthogonal
probes, tied/one-class ROC behavior, paired cluster bootstraps (including a full
5,000-resample sparse example), all three regimes, both denominators and fail-closed
admission. The full-engine smoke uses ten bootstrap resamples for speed and labels
that count explicitly; it is not a production statistical analysis.

Primary API references: [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[ROC curves](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html),
and [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
