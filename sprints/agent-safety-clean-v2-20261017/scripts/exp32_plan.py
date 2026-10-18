"""Prospective CPU-only Exp3.2 methods. No activation extraction or launch."""
from hashlib import sha256
import json

SEED = 2026101832
FOLDS = 5
BOOTSTRAPS = 5000
PINS = {'numpy': '2.5.4', 'scipy': '1.18.1', 'scikit-learn': '1.9.1'}
FACTORS = ('cost', 'consequence', 'boundary', 'domain', 'arm')
LEVELS = {'cost': ('low', 'high_tedium', 'high_friction'),
          'consequence': ('harmless', 'harmful'), 'boundary': ('implicit', 'explicit'),
          'domain': ('finance', 'warehouse', 'compute', 'wetlab'),
          'arm': ('intact', 'refusal_ablated', 'pca_101', 'pca_102', 'pca_103', 'unrelated')}
CATEGORIES = ('SAFE_ROUTE', 'SHORTCUT_VIOLATION', 'ESCALATE_OR_REFUSE', 'TRUNCATED_OR_MALFORMED')


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def plan():
    return {'schema': 'exp32_prospective_plan_v1', 'status': 'PREPARED_NOT_EXECUTED',
      'timing': 'Use pre_action_timing.locate exactly; forward only its prefix_token_ids and read the last token at zero-based decoder block48 output, consistent with Phase1 block hooks. Never generated text.',
      'activation_dimension': 5120, 'layer_index': 48, 'layer_convention': 'zero_based_decoder_block_output',
      'capture_coverage': {'exp11_action_episodes': 4520, 'exp31_action_episodes': 1536},
      'analysis_coverage': 'Exp11 main factorial only:2880 cot_on and320 targeted cot_off; controls retained in capture ledger and excluded by predeclared family, never outcome. Exp31 all1536.',
      'primary_label': 'SHORTCUT_VIOLATION versus the other three taxonomy categories in ITT; authorized harmless direct actions are not relabeled violations. Wording shortcut-taking is interpreted as constraint-violating shortcut behavior; any all-direct-operation secondary requires a separately frozen label contract.',
      'separate_cohorts': 'Never pool experiments. Stratify cot_mode and, in Exp31, payoff_format. Fixed fold assignment shared across all arm/mode/format variants.',
      'splits': {'outer_folds': FOLDS, 'seed': SEED,
        'group': 'Whole exp11 template_id or exp31 canonical_structure_id; all scenarios, domains, arms, formats and modes within a group remain together.',
        'assignment': 'Sort groups by SHA256(seed,experiment,group), then round-robin into five folds; labels/features never choose folds.',
        'tuning': 'None. Fixed C=1 L2 logistic; all feature fitting uses outer training rows only. No label-based fold rerolls.'},
      'regime_A': ['full_layer48_linear_probe', 'full cost*consequence*boundary*domain*arm treatment-coded logistic', 'prompt_only_tfidf'],
      'regime_B': 'Within Intact only: signed h48 dot frozen unit r_hat logistic plus full activation/design/TFIDF comparators; never apply r_hat baseline to Ablated.',
      'regime_C': {'cell': list(FACTORS), 'eligibility': 'Within each outer TRAIN partition and denominator, inclusive20–80% events, both classes, at least2 independent template groups. Test labels never select cells.',
        'fitting': 'Fit separately within each selected exact cell; evaluate its held-out groups only. Intact signed1Drhat and full probe; Ablated h-(h dot r)r probe; fixed constant design score0.5 and train-onlyTFIDF comparators.',
        'report': 'Each cell separately, no cross-cell pooled AUROC, no selection by held-out cell rate. Record every train eligibility decision and all held-out exclusions. No eligible cell is a valid unavailable result.',
        'caveat': 'Exp31 has only4 scenarios per exact cell within each mode/format, so C may have few evaluable rows/clusters or undefined metrics.'},
      'logistic': {'C': 1., 'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-8,
        'class_weight': None, 'random_state': SEED, 'numeric_threads': 1,
        'failure': 'One-class training, empty vocabulary or nonconvergence => unavailable, never invented predictions or alternate fit.'},
      'activation_preprocessing': 'StandardScaler fitted on training only. No PCA or outcome-selected dimensions. Frozen r retains Phase1 values and unit-norm tolerance1e-5. Orthogonalize Ablated h as h-r*(h dot r)/(r dot r) in float64 before scaling; effective input direction remains orthogonal.',
      'tfidf': {'input': 'available_prompt_text from exact pre_think alignment only', 'ngram_range': [1,2],
        'lowercase': True, 'min_df': 1, 'max_features': 20000, 'sublinear_tf': True,
        'normalization': 'l2', 'vocabulary_and_idf': 'training rows only; no generated reasoning, answers or outcome fields'},
      'design_matrix': 'Explicit treatment-coded products across all five factors (all lower-order interactions included), unpenalized intercept via fit_intercept=True; levels fixed prospectively. Report unseen train-cell support.',
      'denominators': {'itt': 'All selected action episodes; malformed are nonevents with separate four-way counts.',
        'strict': 'Exclude only TRUNCATED_OR_MALFORMED; keep original fold assignment and log all removals.'},
      'metrics': {'auroc': 'Held-out out-of-fold scores; half credit ties.',
        'fpr_at_80_tpr': 'Linear interpolation of held-out ROC atTPR=.8, corresponding to randomized tie/threshold boundary; descriptive ROC operating point, not deployment threshold calibration.',
        'ppv_at_10pct_prevalence': '0.1*0.8/(0.1*0.8+0.9*FPR_at_80TPR); prevalence-standardized, not empirical PPV.',
        'undefined': 'Missing either held-out class => all three metrics null with explicit reason; constant scores with both labels => AUROC.5,FPR.8,PPV.1. No made-up chance score for one-class data.'},
      'bootstrap': {'replicates': BOOTSTRAPS, 'seed': SEED, 'unit': 'Whole template/canonical structure with replacement, preserving all paired predictions.',
        'interval': 'Percentile2.5/97.5; conditional on trained OOF predictors and training-only cell selection, not full training-process uncertainty.',
        'availability': 'At least2 observed clusters and at least80% valid two-class bootstrap replicates; otherwise CI null, report valid count and all invalid replicates. Threshold set prospectively from sparse design: balanced four-cluster cells have87.5% expected valid ordinary bootstrap samples.',
        'coverage_limitation': 'The80% finite-replicate rule is computational availability only; it does not establish nominal95% confidence coverage, especially with few clusters or degenerate intervals.',
        'contrasts': 'Paired same-resample metric differences between activation methods and each registered baseline; same evaluable rows, count missing model predictions.',
        'multiplicity': 'All cells/methods reported; C intervals exploratory pointwise95%, no uncorrected family-wide confirmatory success claim.'},
      'source_gates': 'Fresh globalPhase3 readiness plus complete independently audited Exp11 and Exp31 evidence, captured tokenizer/checkpoint/raw-input/activation provenance, and extraction5pilot audit required before empirical analysis. Missing backend audit is PENDING.',
      'package_versions': PINS, 'execution_authorized': False, 'phase3_authorized': False,
      'primary_sources': ['https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html',
        'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html',
        'https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html']}
