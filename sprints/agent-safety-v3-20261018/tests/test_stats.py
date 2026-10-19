import itertools
import json
import math
import pathlib
import sys

import numpy as np
import pytest
from scipy import optimize, stats as sps
from scipy.special import expit, logit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from v3 import stats as S  # noqa: E402


def table_vectors(a, b, c, d):
    """Paired 0/1 vectors with a=both, b=A only, c=B only, d=neither."""
    A = np.array([1] * a + [1] * b + [0] * c + [0] * d)
    B = np.array([1] * a + [0] * b + [1] * c + [0] * d)
    return A, B


# ------------------------------------------------------------------ proportions / McNemar / Newcombe / Holm


def test_wilson_matches_published_and_statsmodels():
    # Newcombe (1998, Stat Med 17:857) Table I: 81/263 -> 0.2553 to 0.3662
    lo, hi = S.wilson_ci(81, 263)
    assert (round(lo, 4), round(hi, 4)) == (0.2553, 0.3662)
    from statsmodels.stats.proportion import proportion_confint
    for k, n in [(0, 10), (5, 10), (10, 10), (56, 270), (3, 40)]:
        ref = proportion_confint(k, n, alpha=0.05, method="wilson")
        assert np.allclose(S.wilson_ci(k, n), ref, atol=1e-12)
    assert S.wilson_ci(0, 0) == (0.0, 1.0)
    assert S.wilson_ci(5, 10, alpha=0.1)[0] > S.wilson_ci(5, 10, alpha=0.05)[0]


def test_exact_mcnemar_hand_values():
    r = S.exact_mcnemar(10, 2)  # 2 * P(Bin(12, .5) <= 2) = 2 * (1 + 12 + 66) / 4096
    assert r["p"] == pytest.approx(158 / 4096, rel=1e-12)
    assert (r["b"], r["c"], r["n_discordant"]) == (10, 2, 12)
    assert S.exact_mcnemar(2, 10)["p"] == pytest.approx(158 / 4096, rel=1e-12)
    r0 = S.exact_mcnemar(0, 5)  # 2 / 32; mid-p removes half of the point mass 1/32 twice counted
    assert r0["p"] == pytest.approx(2 / 32) and r0["p_mid"] == pytest.approx(2 / 32 - 1 / 32)
    assert S.exact_mcnemar(7, 7)["p"] == 1.0
    assert S.exact_mcnemar(0, 0) == {"p": 1.0, "p_mid": 1.0, "b": 0, "c": 0, "n_discordant": 0}
    with pytest.raises(ValueError):
        S.exact_mcnemar(-1, 3)


def test_newcombe_hand_computed_values():
    # Values computed independently with 40-digit Decimal arithmetic from Newcombe's formulas.
    # a=8, b=5, c=1, d=6: p1=.65, p2=.45, Wilson(13/20)=(.432854,.818808), Wilson(9/20)=(.258198,.657915),
    # ad-bc=43 > N/2=10 -> C=33, phi=.347677 -> (-0.042873, 0.406810)
    lo, hi = S.paired_diff_newcombe_ci(*table_vectors(8, 5, 1, 6))
    assert lo == pytest.approx(-0.042873, abs=5e-7) and hi == pytest.approx(0.406810, abs=5e-7)
    # negative correlation branch (ad-bc = -12, no correction): (-0.136272, 0.739074)
    lo, hi = S.paired_diff_newcombe_ci(*table_vectors(2, 6, 2, 0))
    assert lo == pytest.approx(-0.136272, abs=5e-7) and hi == pytest.approx(0.739074, abs=5e-7)
    # Agresti (CDA, presidential approval 794/150/86/570): Wald is 0.04 +- 0.019 = (0.021, 0.059)
    lo, hi = S.paired_diff_newcombe_ci(*table_vectors(794, 150, 86, 570))
    assert (round(lo, 3), round(hi, 3)) == (0.021, 0.059)
    assert lo == pytest.approx(0.021230, abs=5e-7) and hi == pytest.approx(0.058704, abs=5e-7)


def test_newcombe_reduces_to_unpaired_hybrid_score_when_uncorrelated():
    from statsmodels.stats.proportion import confint_proportions_2indep
    # ad = bc and (ad - bc) <= N/2  ->  phi = 0, which is Newcombe's independent-samples interval
    a, b, c, d = 6, 3, 6, 3  # ad-bc = 18-18 = 0
    lo, hi = S.paired_diff_newcombe_ci(*table_vectors(a, b, c, d))
    n = a + b + c + d
    ref = confint_proportions_2indep(a + b, n, a + c, n, method="newcomb", compare="diff")
    assert (lo, hi) == pytest.approx(tuple(ref), abs=1e-12)


def test_newcombe_properties_and_tango_agreement():
    A, B = table_vectors(8, 5, 1, 6)
    lo, hi = S.paired_diff_newcombe_ci(A, B)
    lo2, hi2 = S.paired_diff_newcombe_ci(B, A)  # swapping arms negates the interval
    assert (lo2, hi2) == pytest.approx((-hi, -lo), abs=1e-12)
    assert S.paired_diff_newcombe_ci([], []) == (-1.0, 1.0)
    assert S.paired_diff_newcombe_ci(*table_vectors(0, 0, 0, 10)) == pytest.approx((-0.277533, 0.277533), abs=1e-6)
    lo, hi = S.paired_diff_newcombe_ci(*table_vectors(0, 4, 0, 0))
    assert 0 < lo < hi == 1.0
    # Tango's score interval is an independent method: agree closely on moderate tables
    for t in [(8, 5, 1, 6), (794, 150, 86, 570), (20, 10, 4, 30), (5, 12, 3, 20)]:
        n_lo, n_hi = S.paired_diff_newcombe_ci(*table_vectors(*t))
        t_lo, t_hi = S.paired_diff_tango_ci(*table_vectors(*t))
        assert abs(n_lo - t_lo) < 0.04 and abs(n_hi - t_hi) < 0.04


def test_tango_is_score_inversion():
    t_lo, t_hi = S.paired_diff_tango_ci(*table_vectors(20, 10, 4, 30))
    a, b, c, d = 20, 10, 4, 30
    n = a + b + c + d
    z2 = sps.norm.ppf(0.975) ** 2
    for delta in (t_lo, t_hi):
        # brute-force restricted MLE of P(A=0,B=1) under the null difference `delta`
        grid = np.linspace(max(0, -delta) + 1e-9, (1 - delta) / 2 - 1e-9, 400001)
        ll = b * np.log(grid + delta) + c * np.log(grid) + (a + d) * np.log(1 - 2 * grid - delta)
        x = grid[np.argmax(ll)]
        stat = (b - c - n * delta) ** 2 / (n * (2 * x + delta - delta**2))
        assert stat == pytest.approx(z2, rel=1e-4)
    assert t_lo < (b - c) / n < t_hi
    assert S.paired_diff_tango_ci([], []) == (-1.0, 1.0)
    assert S.paired_diff_tango_ci(*table_vectors(0, 4, 0, 0))[1] == 1.0
    assert S.paired_diff_tango_ci(*table_vectors(0, 0, 0, 10)) == pytest.approx((-0.277533, 0.277533), abs=1e-6)


def test_newcombe_coverage_is_near_nominal():
    rng = np.random.default_rng(7)
    probs = np.array([0.3, 0.2, 0.1, 0.4])  # P(11), P(10), P(01), P(00)
    truth = probs[1] - probs[2]
    hits = 0
    m = 600
    for _ in range(m):
        a, b, c, d = rng.multinomial(50, probs)
        lo, hi = S._newcombe10(a, b, c, d, 0.05)
        hits += lo <= truth <= hi
    assert 0.92 <= hits / m <= 0.99


def test_paired_contrast_basic():
    A, B = table_vectors(8, 5, 1, 6)
    r = S.paired_contrast(A, B)
    assert r["n_pairs"] == 20 and r["rate_a"] == 0.65 and r["rate_b"] == 0.45
    assert r["diff"] == pytest.approx(0.2)
    assert (r["n_a_only"], r["n_b_only"], r["n_both"], r["n_neither"], r["n_discordant"]) == (5, 1, 8, 6, 6)
    assert r["p_mcnemar"] == pytest.approx(S.exact_mcnemar(5, 1)["p"])
    assert (r["ci_low"], r["ci_high"]) == pytest.approx(S.paired_diff_newcombe_ci(A, B))
    assert r["wilson_a"] == pytest.approx(S.wilson_ci(13, 20)) and r["ci_method"] == "newcombe10"
    assert "cluster_boot" not in r
    json.dumps(r)  # JSON-friendly


def test_paired_contrast_ids_and_clusters():
    a = np.array([1, 0, 1, 1, 0, 0])
    b = np.array([0, 0, 1, 0, 1, 0])
    keys = ["t1", "t2", "t3", "t4", "t5", "t6"]
    r = S.paired_contrast(a, b, ids=keys)
    assert r["n_pairs"] == 6 and r["n_a_only"] == 2 and r["n_b_only"] == 1
    # b given in a different order with one unmatched key on each side
    perm = [4, 0, 2, 5, 1, 3]
    r2 = S.paired_contrast(a[:5], b[perm][:5], ids=keys[:5], ids_b=[keys[i] for i in perm][:5])
    assert r2["n_pairs"] == 4 and r2["n_unmatched"] == 2
    with pytest.raises(ValueError):
        S.paired_contrast(a, b, ids=["x"] * 6)
    with pytest.raises(ValueError):
        S.paired_contrast(a, b[:5])
    with pytest.raises(ValueError):
        S.paired_contrast([0, 2], [0, 1])
    rc = S.paired_contrast(a, b, cluster_ids=["s1", "s1", "s2", "s2", "s3", "s3"], n_boot=200, seed=1)
    assert rc["cluster_boot"]["n_clusters"] == 3 and rc["cluster_boot"]["diff"] == pytest.approx(rc["diff"])


def test_holm_hand_values():
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> running max .06
    assert S.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    d = S.holm({"P1.1": 0.001, "P1.2": 0.02, "P1.3": 0.5})
    assert d == pytest.approx({"P1.1": 0.003, "P1.2": 0.04, "P1.3": 0.5})
    assert list(d) == ["P1.1", "P1.2", "P1.3"]
    assert S.holm([0.4, 0.9]) == pytest.approx([0.8, 0.9])
    out = S.holm([0.01, float("nan"), 0.04])  # NaN is not counted in m
    assert out[0] == pytest.approx(0.02) and math.isnan(out[1]) and out[2] == pytest.approx(0.04)
    assert S.holm([]) == []


# ------------------------------------------------------------------ cluster bootstrap


def test_cluster_bootstrap_diff_deterministic_and_degenerate():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(12), 10)
    a = (rng.random(120) < 0.6).astype(int)
    b = (rng.random(120) < 0.3).astype(int)
    r1 = S.cluster_bootstrap_diff(a, b, cl, n_boot=2000, seed=5)
    r2 = S.cluster_bootstrap_diff(a, b, cl, n_boot=2000, seed=5)
    r3 = S.cluster_bootstrap_diff(a, b, cl, n_boot=2000, seed=6)
    assert r1 == r2 and r1["ci_low"] != r3["ci_low"]
    assert r1["diff"] == pytest.approx(a.mean() - b.mean())
    assert r1["ci_low"] < r1["diff"] < r1["ci_high"] and r1["n_clusters"] == 12 and not r1["few_clusters"]
    same = S.cluster_bootstrap_diff(a, a, cl, n_boot=500, seed=1)
    assert (same["diff"], same["ci_low"], same["ci_high"], same["p_boot"]) == (0.0, 0.0, 0.0, 1.0)
    full = S.cluster_bootstrap_diff(np.ones(40, int), np.zeros(40, int), np.repeat(np.arange(8), 5), n_boot=999)
    assert full["ci_low"] == full["ci_high"] == 1.0 and full["p_boot"] == pytest.approx(1 / 1000)
    assert full["few_clusters"]
    with pytest.raises(ValueError):
        S.cluster_bootstrap_diff(a, b, [0] * 120)


def test_cluster_bootstrap_is_wider_than_naive_when_clustered():
    # 10 clusters of 20 pairs; a=1,b=0 for every pair in 5 clusters and a=b=0 in the rest
    cl = np.repeat(np.arange(10), 20)
    a = (cl < 5).astype(int)
    b = np.zeros_like(a)
    naive_lo, naive_hi = S.paired_diff_newcombe_ci(a, b)
    boot = S.cluster_bootstrap_diff(a, b, cl, n_boot=4000, seed=3)
    assert boot["diff"] == 0.5
    assert (boot["ci_high"] - boot["ci_low"]) > (naive_hi - naive_lo) + 0.1
    assert S.exact_mcnemar(100, 0)["p"] < 1e-20 < boot["p_boot"] < 0.05  # naive p is absurdly small


# ------------------------------------------------------------------ AUROC


def test_auroc_known_values():
    assert S.auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75  # classic sklearn doc example
    assert S.auroc([0, 1], [0.1, 0.9]) == 1.0 and S.auroc([0, 1], [0.9, 0.1]) == 0.0
    assert S.auroc([0, 1, 0, 1, 1], [3.0] * 5) == 0.5  # constant score (design-matrix baseline in Regime C)
    assert math.isnan(S.auroc([1, 1, 1], [0.2, 0.3, 0.4]))
    with pytest.raises(ValueError):
        S.auroc([0, 1, 2], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        S.auroc([0, 1], [0.1, float("nan")])


def test_auroc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.4).astype(int)
    score = np.round(rng.normal(y * 0.8, 1.0), 1)  # many ties
    assert S.auroc(y, score) == pytest.approx(roc_auc_score(y, score), abs=1e-12)


def test_auroc_bootstrap_ci():
    rng = np.random.default_rng(2)
    y = (rng.random(200) < 0.5).astype(int)
    score = rng.normal(y * 1.0, 1.0)
    r = S.auroc_bootstrap_ci(y, score, n_boot=500, seed=9)
    assert r == S.auroc_bootstrap_ci(y, score, n_boot=500, seed=9)
    assert r["ci_low"] < r["auroc"] < r["ci_high"] and r["n_skipped"] == 0 and r["n_clusters"] is None
    assert r["auroc"] == pytest.approx(S.auroc(y, score))
    # constant baseline: AUROC is exactly 0.5 in every resample
    base = S.auroc_bootstrap_ci(y, np.zeros(200), n_boot=100, seed=1)
    assert (base["auroc"], base["ci_low"], base["ci_high"]) == (0.5, 0.5, 0.5)
    # cluster resampling: 20 clusters of 10 rows, labels constant within half of the clusters
    cl = np.repeat(np.arange(20), 10)
    yc = (cl % 2).astype(int)
    sc = rng.normal(yc * 1.0, 1.0)
    rc = S.auroc_bootstrap_ci(yc, sc, n_boot=300, seed=4, cluster_ids=cl)
    assert rc["n_clusters"] == 20 and rc["ci_low"] <= rc["auroc"] <= rc["ci_high"]
    with pytest.raises(ValueError):
        S.auroc_bootstrap_ci(yc, sc, n_boot=10, cluster_ids=[0] * 200)
    # tiny, single-positive data: resamples without a positive are skipped and counted
    tiny = S.auroc_bootstrap_ci([1, 0, 0, 0, 0], [0.9, 0.1, 0.2, 0.3, 0.4], n_boot=200, seed=1)
    assert tiny["n_skipped"] > 0


def test_auroc_diff_bootstrap_ci():
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.5).astype(int)
    good = rng.normal(y * 1.5, 1.0)
    weak = rng.normal(y * 0.2, 1.0)
    r = S.auroc_diff_bootstrap_ci(y, good, weak, n_boot=500, seed=2)
    assert r["diff"] == pytest.approx(S.auroc(y, good) - S.auroc(y, weak))
    assert r["ci_low"] > 0 and r["p_boot"] < 0.05
    assert r == S.auroc_diff_bootstrap_ci(y, good, weak, n_boot=500, seed=2)
    same = S.auroc_diff_bootstrap_ci(y, good, good, n_boot=100, seed=2)
    assert (same["diff"], same["ci_low"], same["ci_high"]) == (0.0, 0.0, 0.0) and same["p_boot"] == 1.0


# ------------------------------------------------------------------ Firth logistic regression


def _two_group_design(n0=10, n1=10):
    return np.array([[1.0, 0.0]] * n0 + [[1.0, 1.0]] * n1)


@pytest.mark.parametrize("successes_group1", [0, 5, 10])
def test_firth_saturated_two_group_equals_half_added_logits(successes_group1):
    """For a saturated binary-covariate model the Firth estimate is logit((y + .5) / (n + 1)) per group
    (Jeffreys prior), including the perfectly separated cases (0/10 vs 10/10)."""
    X = _two_group_design()
    y = np.array([0] * 10 + [1] * successes_group1 + [0] * (10 - successes_group1))
    f = S.firth_logit(X, y, names=["Intercept", "grp"], profile=False)
    g0, g1 = logit(0.5 / 11), logit((successes_group1 + 0.5) / 11)
    assert f["converged"] and f["names"] == ["Intercept", "grp"]
    assert f["coef"] == pytest.approx([g0, g1 - g0], abs=1e-7)
    assert np.all(np.isfinite(f["se"])) and f["max_abs_score"] < 1e-6
    assert "ci_pl" not in f


def test_firth_penalized_score_and_independent_optimizer():
    rng = np.random.default_rng(11)
    n = 60
    X = np.column_stack([np.ones(n), rng.normal(size=n), (rng.random(n) < 0.5).astype(float)])
    y = (rng.random(n) < expit(0.5 + 1.2 * X[:, 1] - 0.8 * X[:, 2])).astype(int)
    f = S.firth_logit(X, y, profile=False)

    def neg_pen_ll(b):
        return -S._firth_pen_loglik(X, y.astype(float), b)

    ref = optimize.minimize(neg_pen_ll, np.zeros(3), method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 5000})
    assert f["coef"] == pytest.approx(ref.x, abs=1e-4)
    eps = 1e-6  # central-difference gradient of l* at the solution is ~ 0 (modified score = 0)
    grad = [(neg_pen_ll(f["coef"] + eps * e) - neg_pen_ll(f["coef"] - eps * e)) / (2 * eps) for e in np.eye(3)]
    assert np.max(np.abs(grad)) < 1e-5


def test_firth_close_to_mle_in_large_samples_and_shrinks_in_small():
    import statsmodels.api as sm
    rng = np.random.default_rng(5)
    n = 4000
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = (rng.random(n) < expit(-0.3 + 0.9 * X[:, 1])).astype(int)
    f = S.firth_logit(X, y, profile=False)
    mle = sm.Logit(y, X).fit(disp=0)
    assert f["coef"] == pytest.approx(mle.params, abs=5e-3)
    assert f["se"] == pytest.approx(mle.bse, rel=2e-2)
    # small sample, rare outcome: Firth estimate is closer to zero than the (biased-away) MLE
    Xs = np.column_stack([np.ones(30), np.r_[np.zeros(15), np.ones(15)]])
    ys = np.r_[np.zeros(14), 1, np.zeros(11), np.ones(4)]
    fs = S.firth_logit(Xs, ys, profile=False)
    mle_s = sm.Logit(ys, Xs).fit(disp=0)
    assert 0 < fs["coef"][1] < mle_s.params[1]


def test_firth_profile_ci_is_exact_penalized_lr_interval():
    X = _two_group_design()
    y = np.array([0] * 10 + [1] * 5 + [0] * 5)
    f = S.firth_logit(X, y, names=["Intercept", "grp"])
    crit = sps.chi2.ppf(0.95, 1)
    yf = y.astype(float)
    for j in (0, 1):
        lo, hi = f["ci_pl"][j]
        assert lo < f["coef"][j] < hi
        for limit in (lo, hi):
            # independent constrained maximisation with scipy (not the module's Newton)
            other = 1 - j

            def neg(b_other, limit=limit, j=j):
                beta = np.zeros(2)
                beta[j], beta[1 - j] = limit, b_other[0]
                return -S._firth_pen_loglik(X, yf, beta)

            prof = -optimize.minimize(neg, [f["coef"][other]], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14}).fun
            assert 2 * (f["loglik_pen"] - prof) == pytest.approx(crit, abs=1e-4)
    # asymmetric under separation-like sparsity and a penalized LR p-value consistent with the interval
    assert (f["ci_pl"][0, 1] - f["coef"][0]) != pytest.approx(f["coef"][0] - f["ci_pl"][0, 0], abs=0.1)
    assert 0 < f["p_pl"][1] < 0.05  # the slope is significant
    assert np.all((f["p_pl"] >= 0) & (f["p_pl"] <= 1))


def test_firth_perfect_separation_continuous_covariate_is_finite():
    x = np.arange(10, dtype=float)
    y = (x >= 5).astype(int)  # perfectly separated: the MLE does not exist
    X = np.column_stack([np.ones(10), x - 4.5])
    f = S.firth_logit(X, y, profile_cols=[1])
    assert f["converged"] and np.all(np.isfinite(f["coef"])) and f["coef"][1] > 0
    assert np.isfinite(f["ci_pl"][1]).all() and f["ci_pl"][1, 0] < f["coef"][1] < f["ci_pl"][1, 1]
    assert np.isnan(f["ci_pl"][0]).all()  # not requested -> NaN placeholder


def test_firth_intercept_only_is_jeffreys_logit():
    f = S.firth_logit(np.ones((12, 1)), np.zeros(12))  # no events at all: finite, equals logit(0.5 / (n + 1))
    assert f["coef"][0] == pytest.approx(logit(0.5 / 13), abs=1e-7) and np.isfinite(f["ci_pl"]).all()


def test_to_jsonable_is_strict_json():
    obj = {"a": np.array([1.0, np.nan, np.inf]), "b": (np.float64(0.5), np.int64(3), np.bool_(True)), "c": {"d": [np.nan]}, 5: "x"}
    out = S.to_jsonable(obj)
    assert out == {"a": [1.0, None, None], "b": [0.5, 3, True], "c": {"d": [None]}, "5": "x"}
    json.dumps(out, allow_nan=False)
    fit = S.fit_exp11_model(_exp11_records(seed=2, n_t=4, n_s=2), factors=["arm", "consequence"])
    json.dumps(S.to_jsonable(fit), allow_nan=False)


def test_firth_input_validation():
    with pytest.raises(ValueError):
        S.firth_logit(np.column_stack([np.ones(4), np.ones(4)]), [0, 1, 0, 1])  # collinear columns
    with pytest.raises(ValueError):
        S.firth_logit(np.ones((3, 1)), [0, 1, 2])
    with pytest.raises(ValueError):
        S.firth_logit(np.ones((3, 1)), [0, 1])


# ------------------------------------------------------------------ design matrices


def _grid_records():
    recs = []
    for arm, cost, cons in itertools.product(["intact", "ablated"], ["low", "high_friction", "high_tedium"], ["harmless", "harmful"]):
        recs.append({"arm": arm, "cost": cost, "cons": cons})
    return recs


def test_design_matrix_treatment_coding_and_interactions():
    recs = _grid_records()
    X, names = S.design_matrix_from_cells(recs, ["arm", "cost", "cons"], {"arm": "intact", "cost": "low", "cons": "harmless"})
    # 1 + (1 + 2 + 1) main + (2 + 1 + 2) two-way + 2 three-way = 12 columns, 12 cells -> saturated
    assert X.shape == (12, 12) and np.linalg.matrix_rank(X) == 12
    assert names[:5] == ["Intercept", "arm[T.ablated]", "cost[T.high_friction]", "cost[T.high_tedium]", "cons[T.harmful]"]
    assert "arm[T.ablated]:cost[T.high_tedium]:cons[T.harmful]" in names
    assert "cost[T.high_friction]:cost[T.high_tedium]" not in names
    row = {"arm": "ablated", "cost": "high_tedium", "cons": "harmful"}
    Xr, nr = S.design_matrix_from_cells([row] + recs, ["arm", "cost", "cons"], {"arm": "intact", "cost": "low", "cons": "harmless"})
    ones = {n for n, v in zip(nr, Xr[0]) if v == 1.0}
    assert ones == {"Intercept", "arm[T.ablated]", "cost[T.high_tedium]", "cons[T.harmful]",
                    "arm[T.ablated]:cost[T.high_tedium]", "arm[T.ablated]:cons[T.harmful]",
                    "cost[T.high_tedium]:cons[T.harmful]", "arm[T.ablated]:cost[T.high_tedium]:cons[T.harmful]"}
    Xm, nm = S.design_matrix_from_cells(recs, ["arm", "cost", "cons"], interactions=1)
    assert Xm.shape == (12, 5) and nm[0] == "Intercept"
    Xn, nn = S.design_matrix_from_cells(recs, ["arm"], {"arm": "ablated"}, intercept=False)
    assert nn == ["arm[T.intact]"] and Xn.shape == (12, 1)
    with pytest.raises(ValueError):
        S.design_matrix_from_cells(recs, ["arm"], {"arm": "missing"})


# ------------------------------------------------------------------ Firth-type random-intercept model


def _simulate_crossed(seed, n_t=10, n_s=4, reps=3, beta=(-1.0, 1.2, 0.8), sig_t=0.7, sig_s=0.5):
    """Scenarios nested in templates; every scenario is observed in the 2x2 design `reps` times."""
    rng = np.random.default_rng(seed)
    t_eff = rng.normal(0, sig_t, n_t)
    s_eff = rng.normal(0, sig_s, n_t * n_s)
    rows = [(t, t * n_s + s, a, c) for t in range(n_t) for s in range(n_s) for a in (0, 1) for c in (0, 1) for _ in range(reps)]
    rows = np.array(rows)
    X = np.column_stack([np.ones(len(rows)), rows[:, 2], rows[:, 3]]).astype(float)
    eta = X @ np.array(beta) + t_eff[rows[:, 0]] + s_eff[rows[:, 1]]
    y = (rng.random(len(rows)) < expit(eta)).astype(int)
    groups = {"template_id": [f"T{t:02d}" for t in rows[:, 0]], "scenario_id": [f"S{s:03d}" for s in rows[:, 1]]}
    return X, y, groups


def test_mixed_objective_gradient_matches_direct_formula():
    """Q = l - u'D^-1 u/2 + 0.5 log|X'(W^-1 + Z D Z')^-1 X|; the code's penalized score must equal dQ/d(beta,u)."""
    X, y, g = _simulate_crossed(1, n_t=3, n_s=2, reps=1)
    model = S._Glmm(X, y.astype(float), g)
    dinv = model.dinv(np.array([0.7, 0.5]))
    rng = np.random.default_rng(0)
    beta, u = rng.normal(0, 0.5, 3), rng.normal(0, 0.3, model.q)
    Z = model.Z.toarray()

    def q_direct(b, uu):
        eta = X @ b + Z @ uu
        w = expit(eta) * (1 - expit(eta))
        v = np.diag(1 / w) + Z @ np.diag(1 / dinv) @ Z.T
        info = X.T @ np.linalg.inv(v) @ X
        return np.sum(y * eta - np.logaddexp(0, eta)) - 0.5 * uu @ (dinv * uu) + 0.5 * np.linalg.slogdet(info)[1]

    st = model.evaluate(beta, u, dinv)
    assert st["Q"] == pytest.approx(q_direct(beta, u), abs=1e-9)
    g_beta, g_u, _ = model.score(st, u, dinv)
    th = np.concatenate([beta, u])
    eps = 1e-6
    num = np.array([(q_direct((th + eps * e)[:3], (th + eps * e)[3:]) - q_direct((th - eps * e)[:3], (th - eps * e)[3:])) / (2 * eps)
                    for e in np.eye(len(th))])
    assert np.max(np.abs(num - np.concatenate([g_beta, g_u]))) < 1e-6
    # the Newton direction solves H d = g for the Henderson matrix H
    d_beta, d_u, _ = model.newton_step(st, u, dinv)
    h_mat = np.block([[X.T @ (X * st["w"][:, None]), st["ztwx"].T], [st["ztwx"], Z.T @ (Z * st["w"][:, None]) + np.diag(dinv)]])
    assert h_mat @ np.concatenate([d_beta, d_u]) == pytest.approx(np.concatenate([g_beta, g_u]), abs=1e-8)


def test_mixed_with_negligible_variance_equals_firth_logit():
    X, y, g = _simulate_crossed(2, n_t=5, n_s=3, reps=2)
    ref = S.firth_logit(X, y, profile=False)
    f = S.firth_random_intercept_logit(X, y, g, sigma={"template_id": 1e-4, "scenario_id": 1e-4}, sigma_floor=1e-5)
    assert f["coef"] == pytest.approx(ref["coef"], abs=2e-4)
    assert f["se"] == pytest.approx(ref["se"], rel=1e-3)
    assert f["sigma_fixed"] == ["scenario_id", "template_id"] and f["at_floor"] == {}


def test_mixed_recovers_known_fixed_effects_and_variances():
    true = np.array([-1.0, 1.2, 0.8])
    X, y, g = _simulate_crossed(4, n_t=12, n_s=4, reps=3, sig_t=0.7, sig_s=0.5)
    f = S.firth_random_intercept_logit(X, y, g, names=["Intercept", "a", "c"])
    assert f["converged"] and f["names"] == ["Intercept", "a", "c"]
    err = np.abs(f["coef"] - true)
    assert np.all(err < 3 * f["se"]) and np.all(err < 0.6)
    assert 0.2 < f["sigma"]["template_id"] < 1.6 and 0.15 < f["sigma"]["scenario_id"] < 1.2
    assert f["n_groups"] == {"template_id": 12, "scenario_id": 48} and f["dropped_groups"] == []
    assert isinstance(f["approximation"], str) and "Firth" in f["approximation"] and "NOT an exact" in f["approximation"]
    assert set(f["random_effects"]) == {"template_id", "scenario_id"} and len(f["random_effects"]["scenario_id"]) == 48
    assert f["ci_wald"].shape == (3, 2) and np.all(f["ci_wald"][:, 0] < f["coef"]) and f["cov"].shape == (3, 3)
    assert f["variance_method"] == "laplace_ml"
    # deterministic given the data
    f2 = S.firth_random_intercept_logit(X, y, g, names=["Intercept", "a", "c"])
    assert np.array_equal(f["coef"], f2["coef"]) and f["sigma"] == f2["sigma"]


def test_mixed_average_recovery_over_replicates():
    true = np.array([-1.0, 1.2, 0.8])
    reps = 16
    errs, sig_t, sig_s, cover = [], [], [], []
    for seed in range(reps):
        X, y, g = _simulate_crossed(100 + seed, n_t=10, n_s=4, reps=3)
        f = S.firth_random_intercept_logit(X, y, g)
        errs.append(f["coef"] - true)
        cover.append(np.abs(f["coef"] - true) < 1.96 * f["se"])
        sig_t.append(f["sigma"]["template_id"])
        sig_s.append(f["sigma"]["scenario_id"])
    errs = np.array(errs)
    assert np.all(np.abs(errs.mean(axis=0)) < 3 * errs.std(axis=0, ddof=1) / math.sqrt(reps))  # no detectable bias
    assert np.mean(cover) > 0.85
    assert 0.35 < np.mean(sig_t) < 1.1 and 0.2 < np.mean(sig_s) < 0.8  # truth 0.7 / 0.5; ML is biased low


def test_mixed_reml_variant_runs_and_sigma_not_smaller():
    X, y, g = _simulate_crossed(6, n_t=10, n_s=4, reps=2)
    ml = S.firth_random_intercept_logit(X, y, g)
    rl = S.firth_random_intercept_logit(X, y, g, reml=True)
    assert rl["variance_method"] == "laplace_reml" and rl["converged"]
    assert rl["sigma"]["template_id"] >= 0.8 * ml["sigma"]["template_id"] - 1e-9
    assert rl["coef"] == pytest.approx(ml["coef"], abs=0.4)


def test_mixed_separation_all_zero_cell_is_finite():
    """An arm x consequence cell with no events at all (the common case here): estimates stay finite."""
    rng = np.random.default_rng(3)
    recs = []
    t_eff, s_eff = rng.normal(0, 0.6, 8), rng.normal(0, 0.5, 32)
    for t in range(8):
        for s in range(4):
            sid = t * 4 + s
            for arm, cons in itertools.product(["intact", "ablated"], ["harmless", "harmful"]):
                for _ in range(2):
                    eta = 0.8 + t_eff[t] + s_eff[sid] - (1.0 if arm == "intact" else 0.0)
                    if cons == "harmful":
                        eta = -8.0 if arm == "intact" else eta - 1.0
                    recs.append({"arm": arm, "cons": cons, "t": f"T{t}", "s": f"S{sid}", "y": int(rng.random() < expit(eta))})
    zero_cell = [r["y"] for r in recs if r["arm"] == "intact" and r["cons"] == "harmful"]
    assert sum(zero_cell) == 0
    X, names = S.design_matrix_from_cells(recs, ["arm", "cons"], {"arm": "intact", "cons": "harmless"})
    y = np.array([r["y"] for r in recs])
    groups = {"template_id": [r["t"] for r in recs], "scenario_id": [r["s"] for r in recs]}
    f = S.firth_random_intercept_logit(X, y, groups, names=names)
    assert f["converged"] and np.all(np.isfinite(f["coef"])) and np.all(np.isfinite(f["se"]))
    cell = S.contrast_from_fit(f, {"Intercept": 1, "cons[T.harmful]": 1})  # logit P(shortcut) in the all-zero cell
    assert -12 < cell["estimate"] < -2 and 0 < expit(cell["estimate"]) < 0.1
    # the exact Firth GLM (no random effects) agrees on the sign / rough size of the separated coefficient
    glm = S.firth_logit(X, y, names=names, profile=False)
    assert glm["coef"][names.index("cons[T.harmful]")] < -2 and f["coef"][names.index("cons[T.harmful]")] < -2


def test_mixed_input_handling():
    X, y, g = _simulate_crossed(8, n_t=4, n_s=2, reps=1)
    one_level = {**g, "constant_group": ["same"] * len(y)}
    f = S.firth_random_intercept_logit(X, y, one_level, sigma={"template_id": 0.5, "scenario_id": 0.5})
    assert f["dropped_groups"] == ["constant_group"] and "constant_group" not in f["sigma"]
    assert f["sigma"]["template_id"] == 0.5
    with pytest.raises(ValueError):
        S.firth_random_intercept_logit(X, y, {"only": ["a"] * len(y)})
    with pytest.raises(ValueError):
        S.firth_random_intercept_logit(X, y, {"g": ["a", "b"]})
    # single grouping factor path (1-D variance search)
    f1 = S.firth_random_intercept_logit(X, y, {"template_id": g["template_id"]})
    assert list(f1["sigma"]) == ["template_id"] and f1["converged"]


# ------------------------------------------------------------------ Exp 1.1 helper


def _exp11_records(seed=0, n_t=6, n_s=3):
    """Synthetic Exp 1.1 grid with placeholder labels only (no scenario text)."""
    arms = ["intact", "refusal_ablated", "pca_101"]
    costs = ["low", "high_tedium", "high_friction"]
    rng = np.random.default_rng(seed)
    t_eff, s_eff = rng.normal(0, 0.5, n_t), rng.normal(0, 0.4, n_t * n_s)
    recs = []
    for t in range(n_t):
        for s in range(n_s):
            sid = t * n_s + s
            for a, c, k, b in itertools.product(arms, costs, ["harmless", "harmful"], ["implicit", "explicit"]):
                eta = -2.5 + (2.5 if c != "low" else 0) - (3.0 if k == "harmful" else 0) - (1.0 if b == "explicit" else 0)
                eta += (2.0 if (a == "refusal_ablated" and k == "harmful") else 0) + t_eff[t] + s_eff[sid]
                recs.append({"trial_id": f"{a}-{t}-{s}-{c}-{k}-{b}", "arm": a, "cost_regime": c, "consequence": k, "boundary": b,
                             "template_id": f"T{t:02d}", "scenario_id": f"S{sid:03d}",
                             "direct_taken": bool(rng.random() < expit(eta))})
    return recs


def test_fit_exp11_model_mixed():
    recs = _exp11_records()
    fit = S.fit_exp11_model(recs)
    assert fit["n"] == len(recs) and fit["random_intercepts"] and fit["ci_method"] == "wald"
    assert fit["converged"] and len(fit["terms"]) == len(fit["names"]) == 3 * 3 * 2 * 2 and fit["dropped_columns"] == []
    assert fit["terms"][0]["term"] == "Intercept"
    assert fit["ref_levels"] == {"arm": "intact", "cost_regime": "low", "consequence": "harmless", "boundary": "implicit"}
    assert set(fit["sigma"]) == {"template_id", "scenario_id"} and "approximation" in fit and fit["approximation"]
    t = fit["terms"][1]
    assert t["or_ci_low"] < t["odds_ratio"] < t["or_ci_high"] and t["ci_low"] < t["estimate"] < t["ci_high"]
    json.dumps(fit["terms"])
    json.dumps({k: v for k, v in fit.items() if k not in ("coef", "cov")})
    # the ablated-vs-intact log-odds in the harmful / implicit / high_tedium cell via the contrast helper
    w = {"arm[T.refusal_ablated]": 1, "arm[T.refusal_ablated]:consequence[T.harmful]": 1,
         "arm[T.refusal_ablated]:cost_regime[T.high_tedium]": 1,
         "arm[T.refusal_ablated]:cost_regime[T.high_tedium]:consequence[T.harmful]": 1}
    c = S.contrast_from_fit(fit, w)
    wv = np.array([w.get(n, 0.0) for n in fit["names"]])
    assert c["estimate"] == pytest.approx(wv @ fit["coef"]) and c["se"] == pytest.approx(math.sqrt(wv @ fit["cov"] @ wv))
    assert c["ci_low"] < c["estimate"] < c["ci_high"]
    assert S.contrast_from_fit(fit, wv)["estimate"] == pytest.approx(c["estimate"])


def test_fit_exp11_model_options_missing_and_dropped_columns():
    recs = _exp11_records(seed=1, n_t=4, n_s=2)
    recs = [r for r in recs if not (r["arm"] == "pca_101" and r["cost_regime"] == "high_friction" and r["consequence"] == "harmful")]
    for r in recs[:5]:
        r["direct_taken"] = None  # missing outcome rows are dropped and counted
    fit = S.fit_exp11_model(recs, factors=["arm", "cost_regime", "consequence"], interactions=2,
                            outcome=lambda r: r["direct_taken"], random_intercepts=False, profile_cols=["arm[T.refusal_ablated]"])
    assert fit["n_dropped_missing"] == 5 and fit["n"] == len(recs) - 5 and fit["ci_method"] == "profile"
    assert fit["approximation"] is None and "sigma" not in fit
    ablated = next(t for t in fit["terms"] if t["term"] == "arm[T.refusal_ablated]")
    assert ablated["ci_low"] < ablated["estimate"] < ablated["ci_high"]
    other = next(t for t in fit["terms"] if t["term"] == "cost_regime[T.high_tedium]")
    assert math.isnan(other["ci_low"])  # profile interval not requested for this term
    with pytest.raises(ValueError):
        S.fit_exp11_model([{"arm": "intact", "direct_taken": None}], factors=["arm"])
    # an unobserved level combination is detected and reported rather than crashing
    recs2 = [r for r in recs if not (r["arm"] == "refusal_ablated" and r["consequence"] == "harmful")]
    fit2 = S.fit_exp11_model(recs2, outcome=lambda r: bool(r["direct_taken"]), random_intercepts=False,
                             factors=["arm", "consequence"], profile_cols=[])
    reasons = {d["term"]: d["reason"] for d in fit2["dropped_columns"]}
    assert "arm[T.refusal_ablated]:consequence[T.harmful]" in reasons
    assert all(isinstance(v, str) for v in reasons.values())


# ------------------------------------------------------------------ reviewer additions (adversarial / independent checks)


def test_exact_mcnemar_matches_statsmodels_and_mid_p_definition():
    """p against statsmodels over a grid; mid-p = 2 P(X<k) + P(X=k) from its definition, including b == c."""
    from statsmodels.stats.contingency_tables import mcnemar
    for b in range(0, 16):
        for c in range(0, 16):
            r = S.exact_mcnemar(b, c)
            n, k = b + c, min(b, c)
            if n == 0:
                assert r["p"] == 1.0 and r["p_mid"] == 1.0
                continue
            assert r["p"] == pytest.approx(mcnemar([[3, b], [c, 4]], exact=True).pvalue, abs=1e-12), (b, c)
            mid = min(1.0, 2 * (sps.binom.cdf(k - 1, n, 0.5) + 0.5 * sps.binom.pmf(k, n, 0.5)))
            assert r["p_mid"] == pytest.approx(mid, abs=1e-12), (b, c)
    assert S.exact_mcnemar(1, 1)["p_mid"] == 1.0  # used to be 0.5


def test_wilson_rejects_impossible_counts():
    for k, n in [(11, 10), (-1, 10)]:
        with pytest.raises(ValueError):
            S.wilson_ci(k, n)


def test_to_jsonable_zero_dim_and_nested_arrays():
    out = S.to_jsonable({"s": np.array(3.5), "m": np.array([[1, np.nan], [np.inf, 4]]), "t": (np.int32(2),)})
    assert out == {"s": 3.5, "m": [[1.0, None], [None, 4.0]], "t": [2]}
    json.dumps(out, allow_nan=False)


def test_newcombe_exact_coverage_by_enumeration_is_at_least_nominal():
    """No Monte Carlo: sum the multinomial probability of every N=20 table whose interval contains the truth.
    The last vector is strongly concordant, where a wrong sign of phi would show up as over- or under-coverage."""
    from scipy.stats import multinomial
    n = 20
    for probs in [(0.45, 0.15, 0.05, 0.35), (0.25, 0.25, 0.25, 0.25), (0.2, 0.3, 0.1, 0.4), (0.4, 0.05, 0.05, 0.5)]:
        truth, cover = probs[1] - probs[2], 0.0
        for a in range(n + 1):
            for b in range(n + 1 - a):
                for c in range(n + 1 - a - b):
                    lo, hi = S.paired_diff_newcombe_ci(*table_vectors(a, b, c, n - a - b - c))
                    if lo - 1e-12 <= truth <= hi + 1e-12:
                        cover += multinomial.pmf([a, b, c, n - a - b - c], n, probs)
        assert 0.94 <= cover <= 0.995, (probs, cover)


def test_tango_endpoints_satisfy_score_equation_on_edge_tables():
    z2 = sps.norm.ppf(0.975) ** 2
    for t in [(3, 7, 2, 1), (20, 2, 9, 3), (0, 3, 5, 12), (15, 1, 1, 15), (30, 12, 0, 5)]:
        a, b, c, d = t
        n = a + b + c + d
        lo, hi = S.paired_diff_tango_ci(*table_vectors(*t))
        assert -1 <= lo < (b - c) / n < hi <= 1
        for delta in (lo, hi):
            if abs(delta) > 0.999:
                continue
            grid = np.linspace(max(0, -delta) + 1e-12, (1 - delta) / 2 - 1e-12, 200001)
            ll = b * np.log(grid + delta) + c * np.log(grid) + (a + d) * np.log(1 - 2 * grid - delta)
            x = grid[np.argmax(ll)]
            assert (b - c - n * delta) ** 2 / (n * (2 * x + delta - delta**2)) == pytest.approx(z2, rel=1e-3)


def test_holm_matches_statsmodels_on_random_vectors():
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(0)
    for _ in range(100):
        p = rng.uniform(0, 1, int(rng.integers(1, 9))) ** 2
        assert S.holm(list(p)) == pytest.approx(multipletests(p, method="holm")[1], abs=1e-12)
        assert list(S.holm(dict(enumerate(p))).values()) == pytest.approx(multipletests(p, method="holm")[1], abs=1e-12)


def test_paired_contrast_cluster_ids_follow_the_rows_of_a_when_pairs_are_dropped():
    keys_a = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
    a = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    keys_b = ["t8", "t7", "t6", "t5", "t4", "t3", "t2", "zzz"]  # t1 has no partner; 'zzz' has no partner
    b = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    cl_a = ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"]
    r = S.paired_contrast(a, b, ids=keys_a, ids_b=keys_b, cluster_ids=cl_a, n_boot=300, seed=3)
    assert r["n_pairs"] == 7 and r["n_unmatched"] == 2  # t2..t8 match; t1 (in a) and 'zzz' (in b) do not
    common = [k for k in keys_a if k in keys_b]
    a_al = np.array([a[keys_a.index(k)] for k in common])
    b_al = np.array([b[keys_b.index(k)] for k in common])
    cl_al = [cl_a[keys_a.index(k)] for k in common]
    expected = S.cluster_bootstrap_diff(a_al, b_al, cl_al, n_boot=300, seed=3)
    assert r["cluster_boot"] == expected and r["cluster_boot"]["n_clusters"] == 4  # t1 is gone, so s1 only has t2
    assert len(cl_al) == 7 and cl_al.count("s1") == 1
    r2 = S.paired_contrast(a, b, ids=keys_a, ids_b=keys_b, cluster_ids=cl_al, n_boot=300, seed=3)  # already aligned
    assert r2["cluster_boot"] == expected


def _brute_signflip_p(s, m):
    obs = abs(sum(s)) / sum(m)
    flips = list(itertools.product([-1, 1], repeat=len(s)))
    return sum(abs(sum(e * x for e, x in zip(f, s))) / sum(m) >= obs - 1e-12 for f in flips) / len(flips)


def test_cluster_signflip_p_matches_brute_force_and_is_valid_with_few_clusters():
    rng = np.random.default_rng(4)
    for k in (2, 4, 7):
        cl = np.repeat(np.arange(k), 6)
        a = (rng.random(6 * k) < 0.7).astype(int)
        b = (rng.random(6 * k) < 0.3).astype(int)
        r = S.cluster_bootstrap_diff(a, b, cl, n_boot=100, seed=1)
        s = [float((a - b)[cl == c].sum()) for c in range(k)]
        assert r["p_signflip"] == pytest.approx(_brute_signflip_p(s, [6] * k)) and r["signflip_exact"]
    # 4 clusters can never reach p < 2 / 2^4, even for a perfectly consistent effect
    r = S.cluster_bootstrap_diff(np.ones(24, int), np.zeros(24, int), np.repeat(np.arange(4), 6), n_boot=200)
    assert r["p_signflip"] == 0.125 and r["few_clusters"]
    # null data, 6 clusters: the sign-flip test's size stays below the nominal 5% (the shift bootstrap's does not)
    null_rng, rejections, m = np.random.default_rng(11), 0, 300
    cl = np.repeat(np.arange(6), 8)
    for _ in range(m):
        base = null_rng.random(48) < 0.5
        swap = np.repeat(null_rng.random(6) < 0.5, 8)  # arm labels exchangeable within cluster under H0
        shared = null_rng.random(48) < np.repeat(null_rng.uniform(0.2, 0.8, 6), 8)
        a = np.where(swap, shared, base).astype(int)
        b = np.where(swap, base, shared).astype(int)
        rejections += S.cluster_bootstrap_diff(a, b, cl, n_boot=20, seed=2)["p_signflip"] < 0.05
    assert rejections / m < 0.08


def test_cluster_signflip_sampled_path_agrees_with_exact(monkeypatch):
    rng = np.random.default_rng(5)
    k = 18
    cl = np.repeat(np.arange(k), 5)
    a = (rng.random(5 * k) < 0.6).astype(int)
    b = (rng.random(5 * k) < 0.45).astype(int)
    sampled = S.cluster_bootstrap_diff(a, b, cl, n_boot=20000, seed=9)
    assert not sampled["signflip_exact"]
    assert sampled == S.cluster_bootstrap_diff(a, b, cl, n_boot=20000, seed=9)
    monkeypatch.setattr(S, "_SIGNFLIP_EXACT_MAX", 18)
    exact = S.cluster_bootstrap_diff(a, b, cl, n_boot=20000, seed=9)
    assert exact["signflip_exact"] and abs(sampled["p_signflip"] - exact["p_signflip"]) < 0.015
    # a perfectly consistent effect over 20 clusters: no sampled flip reaches it, so p is exactly 1 / (n_flip + 1)
    sure = S.cluster_bootstrap_diff(np.ones(60, int), np.zeros(60, int), np.repeat(np.arange(20), 3), n_boot=999)
    assert not sure["signflip_exact"] and sure["p_signflip"] == pytest.approx(1 / 1000)


def test_cluster_bootstrap_se_matches_analytic_cluster_robust_se():
    rng = np.random.default_rng(8)
    k, m = 60, 6
    cl = np.repeat(np.arange(k), m)
    shift = np.repeat(rng.uniform(0, 1, k), m)
    a = (rng.random(k * m) < 0.2 + 0.5 * shift).astype(int)
    b = (rng.random(k * m) < 0.1 + 0.3 * shift).astype(int)
    r = S.cluster_bootstrap_diff(a, b, cl, n_boot=8000, seed=3)
    cluster_means = (a - b).reshape(k, m).mean(axis=1)
    se = cluster_means.std(ddof=0) / math.sqrt(k)  # bootstrap SE of a mean of k equal-sized cluster means
    assert (r["ci_high"] - r["ci_low"]) / (2 * 1.96) == pytest.approx(se, rel=0.08)


def test_input_validation_for_names_sigma_and_n_boot():
    X, y, g = _simulate_crossed(9, n_t=4, n_s=2, reps=1)
    with pytest.raises(ValueError):
        S.firth_logit(X, y, names=["only_one"])
    with pytest.raises(ValueError):
        S.firth_random_intercept_logit(X, y, g, names=["only_one"])
    with pytest.raises(ValueError):
        S.firth_random_intercept_logit(X, y, g, sigma={"scenario": 0.5})  # typo for scenario_id: must not be ignored
    with pytest.raises(ValueError):
        S.firth_random_intercept_logit(X, y, g, sigma={"template_id": 0.0})
    with pytest.raises(ValueError):
        S.cluster_bootstrap_diff([1, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1], n_boot=0)


def test_fit_exp11_model_never_truncates_non_binary_outcomes():
    """Regression: int(0.7) == 0 used to turn a fractional outcome into a silent non-event."""
    recs = _exp11_records(seed=3, n_t=4, n_s=2)
    for r in recs:
        r["direct_taken"] = 0.7 if r["consequence"] == "harmless" else 0.0
    for bad in (0.7, "SAFE_ROUTE", 2):
        poisoned = [dict(r) for r in recs]
        poisoned[0]["direct_taken"] = bad
        with pytest.raises(ValueError):
            S.fit_exp11_model(poisoned, factors=["consequence"], random_intercepts=False, profile_cols=[])
    clean = [dict(r, direct_taken=bool(r["direct_taken"] > 0.5)) for r in recs]
    clean[0]["direct_taken"] = float("nan")  # NaN is a missing outcome: dropped and counted
    fit = S.fit_exp11_model(clean, factors=["consequence"], random_intercepts=False, profile_cols=[])
    assert fit["n_dropped_missing"] == 1 and fit["n"] == len(recs) - 1


def test_firth_profile_converged_flag_is_reported_and_catches_failed_refits(monkeypatch):
    X = _two_group_design()
    y = np.array([0] * 10 + [1] * 5 + [0] * 5)
    assert S.firth_logit(X, y)["profile_converged"] is True
    assert "profile_converged" not in S.firth_logit(X, y, profile=False)
    real = S._firth_newton

    def failing_when_constrained(X_, y_, beta, free, *args, **kwargs):
        out = real(X_, y_, beta, free, *args, **kwargs)
        return (out[0], out[1], False, out[3]) if len(free) < X_.shape[1] else out

    monkeypatch.setattr(S, "_firth_newton", failing_when_constrained)
    assert S.firth_logit(X, y, profile_cols=[1])["profile_converged"] is False


def test_mixed_model_agrees_with_exact_gauss_hermite_ml_for_one_random_intercept():
    """Independent check of the Laplace variance criterion and the SEs: with ample observations per cluster the
    penalised joint-mode fit must sit close to exact maximum likelihood (adaptive quadrature-free GH, 30 nodes)."""
    from numpy.polynomial.hermite_e import hermegauss
    rng = np.random.default_rng(3)
    n_groups, per = 50, 24
    g = np.repeat(np.arange(n_groups), per)
    x = rng.integers(0, 2, n_groups * per).astype(float)
    y = (rng.random(n_groups * per) < expit(-0.7 + 1.0 * x + rng.normal(0, 0.8, n_groups)[g])).astype(int)
    nodes, weights = hermegauss(30)
    weights = weights / np.sqrt(2 * np.pi)

    def nll(theta):
        eta = theta[0] + theta[1] * x[:, None] + np.exp(theta[2]) * nodes[None, :]
        per_group = np.zeros((n_groups, len(nodes)))
        np.add.at(per_group, g, y[:, None] * eta - np.logaddexp(0, eta))
        top = per_group.max(axis=1, keepdims=True)
        return -np.sum(np.log((np.exp(per_group - top) * weights).sum(axis=1)) + top[:, 0])

    ml = optimize.minimize(nll, [0, 0, 0], method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 3000})
    f = S.firth_random_intercept_logit(np.column_stack([np.ones(len(y)), x]), y, {"g": [str(i) for i in g]})
    assert f["coef"] == pytest.approx(ml.x[:2], abs=0.06)
    assert f["sigma"]["g"] == pytest.approx(math.exp(ml.x[2]), abs=0.08)
    assert 0.05 < f["se"][1] < 0.25  # same order as the ML standard error of the slope (about 0.13)


def test_inputs_are_not_mutated():
    a, b = [1, 0, 1, 1], [0, 0, 1, 1]
    ids = ["a", "b", "c", "d"]
    S.paired_contrast(a, b, ids=ids, ids_b=list(reversed(ids)), cluster_ids=["x", "x", "y", "y"], n_boot=50)
    assert (a, b, ids) == ([1, 0, 1, 1], [0, 0, 1, 1], ["a", "b", "c", "d"])
    recs = _exp11_records(seed=4, n_t=3, n_s=2)
    snapshot = json.dumps(recs, sort_keys=True)
    S.fit_exp11_model(recs, factors=["arm", "consequence"])
    assert json.dumps(recs, sort_keys=True) == snapshot


def test_cluster_bootstrap_pools_unequal_clusters_like_a_manual_resample():
    """Unequal cluster sizes: the statistic is the size-weighted pooled difference (what paired_contrast reports),
    checked against an explicit resample built from the same seeded draws (n_boot <= 1000 is a single draw block)."""
    rng = np.random.default_rng(6)
    sizes = [1, 2, 3, 5, 8, 13, 4, 2]
    cl = np.repeat([f"c{i}" for i in range(len(sizes))], sizes)
    a = (rng.random(len(cl)) < 0.7).astype(int)
    b = (rng.random(len(cl)) < 0.4).astype(int)
    r = S.cluster_bootstrap_diff(a, b, cl, n_boot=500, seed=12)
    assert r["diff"] == pytest.approx(a.mean() - b.mean())
    draws = np.random.default_rng(12).integers(0, len(sizes), size=(500, len(sizes)))
    members = [np.flatnonzero(cl == f"c{i}") for i in range(len(sizes))]
    manual = np.array([np.mean(np.concatenate([a[members[i]] - b[members[i]] for i in row])) for row in draws])
    assert r["ci_low"] == pytest.approx(np.percentile(manual, 2.5), abs=1e-12)
    assert r["ci_high"] == pytest.approx(np.percentile(manual, 97.5), abs=1e-12)
    obs = a.mean() - b.mean()
    assert r["p_boot"] == pytest.approx((1 + np.sum(np.abs(manual - obs) >= abs(obs) - 1e-15)) / 501)
    pc = S.paired_contrast(a, b, cluster_ids=cl, n_boot=500, seed=12)
    assert pc["cluster_boot"]["diff"] == pytest.approx(pc["diff"])


def test_fit_exp11_model_drops_perfectly_confounded_columns():
    """A boundary level that only ever occurs together with one consequence level is collinear, not 'unobserved':
    it must be dropped (and named), never passed to the solver as a rank-deficient design."""
    recs = []
    for t in range(4):
        for s in range(2):
            for cons in ("harmless", "harmful"):
                for rep in range(3):
                    recs.append({"arm": "intact", "cost_regime": "low", "consequence": cons,
                                 "boundary": "implicit" if cons == "harmless" else "explicit",
                                 "template_id": f"T{t}", "scenario_id": f"T{t}_{s}",
                                 "direct_taken": int((t + s + rep + (cons == "harmful")) % 3 == 0)})
    fit = S.fit_exp11_model(recs, factors=["consequence", "boundary"], random_intercepts=False, profile_cols=[])
    assert fit["names"] == ["Intercept", "consequence[T.harmful]"]
    reasons = {d["term"]: d["reason"] for d in fit["dropped_columns"]}
    assert set(reasons) == {"boundary[T.explicit]", "consequence[T.harmful]:boundary[T.explicit]"}
    assert reasons["boundary[T.explicit]"].startswith("collinear")
    assert fit["converged"]
