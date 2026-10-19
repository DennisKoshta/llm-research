"""Statistics for the v3 agent-safety sprint (numpy / scipy; deterministic given a seed).

README: which function answers which sentence of CLAUDE_RESEARCH_SPECIFICATION.md
------------------------------------------------------------------------------
Sec 2.1 "exact paired McNemar p-values + Newcombe 95% CIs on pre-specified contrasts"
    paired_contrast        one call per contrast (P1.1, P1.2, P1.3): rates, difference, Newcombe CI,
                           exact McNemar p, discordant counts (optionally + cluster bootstrap)
    exact_mcnemar          the exact two-sided binomial test on the discordant pairs
    paired_diff_newcombe_ci  Newcombe (1998) method 10 for a paired difference of proportions
    paired_diff_tango_ci   Tango (1998) score interval, an independent cross-check of the above
    holm                   Holm step-down adjustment across the pre-specified contrasts
    wilson_ci              per-arm rate intervals
Sec 2.1 "Firth penalized random-intercept logistic model (1 | template_id) + (1 | scenario_id)"
    firth_logit                    exact Firth (Jeffreys-prior) logistic regression, Wald + profile CIs
    firth_random_intercept_logit   APPROXIMATE Firth-type crossed random-intercept model (read its docstring)
    design_matrix_from_cells       treatment-coded main effects + interactions from trial records
    fit_exp11_model                outcome ~ arm * cost * consequence * boundary with the random intercepts
    contrast_from_fit              linear combinations of fitted coefficients (e.g. one cell's ablated-vs-intact)
Pseudo-replication (survey digests: trials sharing a scenario / template are not independent)
    cluster_bootstrap_diff         cluster (scenario or template) bootstrap CI, bootstrap p and sign-flip p
Spec Sec 3 Track 3 "Regimes A/B/C ... AUROC with 95% bootstrap CIs"
    auroc, auroc_bootstrap_ci      AUROC with row or cluster bootstrap
    auroc_diff_bootstrap_ci        paired bootstrap CI for AUROC(probe) - AUROC(baseline) on the same items

Output: `to_jsonable` turns any result dict (numpy arrays, tuples, NaN) into strict JSON for records/*.json.
Binary inputs are 0/1 (or bool) vectors; the caller decides how TRUNCATED_OR_MALFORMED etc. are encoded.
Default seed everywhere: DEFAULT_SEED. BLAS is pinned to one thread inside the Firth fits (speed, not results).

What is approximate (everything else is exact or a standard published method)
    * firth_random_intercept_logit / fit_exp11_model(random_intercepts=True): a Firth-TYPE penalized quasi-likelihood
      model with Laplace-estimated variances; see its docstring and the returned 'approximation' string.
    * cluster_bootstrap_diff percentile intervals / p_boot are anti-conservative with few clusters (flagged by
      few_clusters); its p_signflip (cluster sign-flip randomisation test) stays valid for any number of clusters.
    * Newcombe method 10 is asymptotic: with very few discordant pairs it can exclude 0 while exact McNemar does not.
"""
from __future__ import annotations

import contextlib
import functools
import itertools
import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy import optimize, sparse, stats
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit

DEFAULT_SEED = 20261018


def _single_thread():
    """Context manager pinning BLAS to one thread: these matrices are small and multi-threaded OpenBLAS
    is ~50x slower on them. Uses threadpoolctl if installed (it ships with scikit-learn), else a no-op."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except ImportError:
        return contextlib.nullcontext()


def _pinned(fn):
    """Decorator: run `fn` with BLAS limited to one thread."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _single_thread():
            return fn(*args, **kwargs)
    return wrapper


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------


def _as_binary(x, name: str = "x") -> np.ndarray:
    """1-D int array of 0/1 values; raises on anything else (including NaN)."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if arr.dtype == bool:
        return arr.astype(int)
    arr = arr.astype(float)
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return arr.astype(int)


def _cluster_groups(cluster_ids: Sequence, n: int) -> list[np.ndarray]:
    """Index arrays, one per distinct cluster id (sorted by id string, so deterministic)."""
    ids = np.asarray([str(c) for c in cluster_ids])
    if len(ids) != n:
        raise ValueError("cluster_ids must have the same length as the data")
    return [np.flatnonzero(ids == c) for c in np.unique(ids)]


def to_jsonable(obj):
    """Recursively convert numpy scalars / arrays / tuples to plain JSON types for records/*.json.
    NaN and +-inf become None so the output is valid strict JSON (`json.dumps(..., allow_nan=False)`)."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())  # tolist() also handles 0-d arrays
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def _holm_array(p: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(p)
    m = int(ok.sum())
    out = np.full_like(p, np.nan, dtype=float)
    order = np.argsort(p[ok], kind="stable")
    adj = (m - np.arange(m)) * p[ok][order]
    adj = np.minimum(1.0, np.maximum.accumulate(adj))
    vals = np.empty(m)
    vals[order] = adj
    out[ok] = vals
    return out


# ----------------------------------------------------------------------------------------------
# proportions, McNemar, Newcombe / Tango, Holm
# ----------------------------------------------------------------------------------------------


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. n == 0 gives the vacuous (0, 1)."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def exact_mcnemar(b: int, c: int) -> dict:
    """Exact two-sided McNemar test: binomial(b + c, 1/2) on the discordant pairs.

    b = pairs with A=1, B=0; c = pairs with A=0, B=1. Returns p (conservative exact), p_mid (mid-p, for
    sensitivity; 1 when b == c), and the counts. No discordant pairs gives p = 1."""
    b, c = int(b), int(c)
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return {"p": 1.0, "p_mid": 1.0, "b": b, "c": c, "n_discordant": 0}
    k = min(b, c)
    p = min(1.0, float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue))
    # mid-p = 2 P(X < k) + P(X = k) = p - P(X = k), except at b == c where the capped p = 1 is already exact
    p_mid = 1.0 if b == c else max(0.0, p - float(stats.binom.pmf(k, n, 0.5)))
    return {"p": p, "p_mid": p_mid, "b": b, "c": c, "n_discordant": n}


def _table_counts(a_vec, b_vec) -> tuple[int, int, int, int]:
    """(both, a_only, b_only, neither) for two aligned 0/1 vectors."""
    a, b = _as_binary(a_vec, "a_vec"), _as_binary(b_vec, "b_vec")
    if a.shape != b.shape:
        raise ValueError("a_vec and b_vec must have the same length")
    return (int(np.sum((a == 1) & (b == 1))), int(np.sum((a == 1) & (b == 0))),
            int(np.sum((a == 0) & (b == 1))), int(np.sum((a == 0) & (b == 0))))


def paired_diff_newcombe_ci(a_vec, b_vec, alpha: float = 0.05) -> tuple[float, float]:
    """Newcombe (1998, Stat Med 17:2635) method 10: CI for p_A - p_B from paired binary data.

    Table (A rows, B columns): a=both 1, b=A only, c=B only, d=neither, N=a+b+c+d. Wilson intervals
    (l1,u1), (l2,u2) for p1=(a+b)/N and p2=(a+c)/N are combined as
        L = D - sqrt((p1-l1)^2 + (u2-p2)^2 - 2 phi (p1-l1)(u2-p2))
        U = D + sqrt((u1-p1)^2 + (p2-l2)^2 - 2 phi (u1-p1)(p2-l2)),   D = p1 - p2,
    with phi the correlation between the two indicators, estimated with a Yates-type continuity
    correction: A = ad-bc; C = A-N/2 if A > N/2, 0 if 0 <= A <= N/2, A if A < 0; phi = C / sqrt((a+b)(c+d)(a+c)(b+d))
    (phi = 0 when a margin is empty). The interval is not a pure test inversion, so it can disagree
    slightly with McNemar's p near 0.05. N == 0 gives the vacuous (-1, 1)."""
    a, b, c, d = _table_counts(a_vec, b_vec)
    return _newcombe10(a, b, c, d, alpha)


def _newcombe10(a: int, b: int, c: int, d: int, alpha: float) -> tuple[float, float]:
    n = a + b + c + d
    if n == 0:
        return (-1.0, 1.0)
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson_ci(a + b, n, alpha)
    l2, u2 = wilson_ci(a + c, n, alpha)
    denom = math.sqrt(float((a + b) * (c + d) * (a + c) * (b + d)))
    if denom == 0:
        phi = 0.0
    else:
        cross = a * d - b * c
        if cross > n / 2:
            cross -= n / 2
        elif cross >= 0:
            cross = 0.0
        phi = cross / denom
    diff = p1 - p2
    lo = diff - math.sqrt(max(0.0, (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2)))
    hi = diff + math.sqrt(max(0.0, (u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2)))
    return (max(-1.0, lo), min(1.0, hi))


def paired_diff_tango_ci(a_vec, b_vec, alpha: float = 0.05) -> tuple[float, float]:
    """Tango (1998) score interval for p_A - p_B: invert the score statistic
    (b - c - N d)^2 / (N (2 x + d - d^2)) = z^2 over d, where x is the restricted MLE of P(A=0,B=1) under
    P(A=1,B=0) - P(A=0,B=1) = d. A test inversion, so it agrees with the asymptotic McNemar test.
    Used here as an independent cross-check of `paired_diff_newcombe_ci`. N == 0 gives (-1, 1)."""
    a, b, c, d = _table_counts(a_vec, b_vec)
    n = a + b + c + d
    if n == 0:
        return (-1.0, 1.0)
    crit = stats.norm.ppf(1 - alpha / 2) ** 2
    est = (b - c) / n

    def score_minus_crit(delta: float) -> float:
        lin = (b + c) - delta * (2 * n - b + c)
        x = (lin + math.sqrt(max(0.0, lin * lin + 8 * n * c * delta * (1 - delta)))) / (4 * n)
        var = n * (2 * x + delta - delta * delta)
        num = (b - c - n * delta) ** 2
        if var <= 0:
            return -crit if num == 0 else math.inf
        return num / var - crit

    eps = 1e-9
    lower, upper = -1.0, 1.0
    if est > -1 + eps and score_minus_crit(-1 + eps) > 0:
        lower = float(optimize.brentq(score_minus_crit, -1 + eps, est, xtol=1e-12))
    if est < 1 - eps and score_minus_crit(1 - eps) > 0:
        upper = float(optimize.brentq(score_minus_crit, est, 1 - eps, xtol=1e-12))
    return (lower, upper)


def _align_pairs(a_vec, b_vec, ids, ids_b) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """(a, b, keys, ia): the aligned vectors, their pair keys, and the positions of those pairs in `a_vec`."""
    a, b = _as_binary(a_vec, "a_vec"), _as_binary(b_vec, "b_vec")
    if ids is None:
        if ids_b is not None:
            raise ValueError("ids_b given without ids")
        if a.shape != b.shape:
            raise ValueError("a_vec and b_vec must have the same length when ids is None")
        return a, b, list(range(len(a))), np.arange(len(a))
    keys_a = [str(k) for k in ids]
    keys_b = keys_a if ids_b is None else [str(k) for k in ids_b]
    if len(keys_a) != len(a) or len(keys_b) != len(b):
        raise ValueError("ids / ids_b must match the length of a_vec / b_vec")
    if len(set(keys_a)) != len(keys_a) or len(set(keys_b)) != len(keys_b):
        raise ValueError("pair ids must be unique within each vector")
    if ids_b is None:
        return a, b, keys_a, np.arange(len(a))
    pos_b = {k: i for i, k in enumerate(keys_b)}
    common = [k for k in keys_a if k in pos_b]
    pos_a = {k: i for i, k in enumerate(keys_a)}
    ia = np.array([pos_a[k] for k in common], dtype=int)
    ib = np.array([pos_b[k] for k in common], dtype=int)
    return a[ia], b[ib], common, ia


def paired_contrast(a_vec, b_vec, ids=None, ids_b=None, cluster_ids=None, alpha: float = 0.05,
                    n_boot: int = 5000, seed: int = DEFAULT_SEED) -> dict:
    """One pre-specified paired contrast A - B on binary outcomes.

    Pairing: positional when `ids` is None; with `ids` (keys of a_vec) the pairs are checked for unique keys;
    with `ids` and `ids_b` the two vectors are inner-joined on their keys (unmatched keys are dropped and
    counted in `n_unmatched`). Returns rates (+ Wilson CIs), diff = rate_a - rate_b, the Newcombe method-10
    CI, the exact McNemar p, n_pairs and the discordant counts (n_a_only: A=1,B=0; n_b_only: A=0,B=1).
    If `cluster_ids` is given, `cluster_boot` holds `cluster_bootstrap_diff`. `cluster_ids` is aligned with the
    rows of `a_vec` (and `ids`); pairs dropped as unmatched are dropped from it too. (An already-aligned list,
    one per surviving pair, is also accepted.)
    Pairs that share a scenario/template are not independent, so McNemar p-values are optimistic;
    report the cluster bootstrap next to them. The Newcombe CI is asymptotic, so with very few discordant
    pairs it can exclude 0 while the exact McNemar p is above alpha (e.g. b=4, c=0)."""
    a, b, keys, ia = _align_pairs(a_vec, b_vec, ids, ids_b)
    n = len(a)
    both, a_only, b_only, neither = _table_counts(a, b)
    mc = exact_mcnemar(a_only, b_only)
    lo, hi = _newcombe10(both, a_only, b_only, neither, alpha)
    out = {
        "n_pairs": n, "rate_a": float(a.mean()) if n else float("nan"),
        "rate_b": float(b.mean()) if n else float("nan"),
        "wilson_a": wilson_ci(int(a.sum()), n, alpha), "wilson_b": wilson_ci(int(b.sum()), n, alpha),
        "diff": float(a.mean() - b.mean()) if n else float("nan"),
        "ci_low": lo, "ci_high": hi, "ci_method": "newcombe10",
        "p_mcnemar": mc["p"], "p_mcnemar_mid": mc["p_mid"],
        "n_both": both, "n_neither": neither, "n_a_only": a_only, "n_b_only": b_only,
        "n_discordant": mc["n_discordant"],
        "n_unmatched": 0 if ids_b is None else len(a_vec) + len(b_vec) - 2 * n,
    }
    if cluster_ids is not None:
        clusters = list(cluster_ids)
        if len(clusters) == len(a_vec) and len(clusters) != n:
            clusters = [clusters[i] for i in ia]
        out["cluster_boot"] = cluster_bootstrap_diff(a, b, clusters, n_boot, seed, alpha)
    return out


def holm(pvals):
    """Holm step-down adjusted p-values. Accepts a mapping (returns a dict with the same keys) or a
    sequence (returns a list in the same order). NaN p-values are left as NaN and not counted in m."""
    if isinstance(pvals, Mapping):
        keys = list(pvals)
        adj = _holm_array(np.array([pvals[k] for k in keys], dtype=float))
        return {k: float(v) for k, v in zip(keys, adj)}
    return [float(v) for v in _holm_array(np.asarray(pvals, dtype=float))]


# ----------------------------------------------------------------------------------------------
# bootstrap: cluster bootstrap for paired differences, AUROC
# ----------------------------------------------------------------------------------------------


def cluster_bootstrap_diff(a_vec, b_vec, cluster_ids, n_boot: int = 5000, seed: int = DEFAULT_SEED,
                           alpha: float = 0.05) -> dict:
    """Cluster bootstrap for the paired risk difference mean(a) - mean(b).

    Whole clusters (e.g. scenarios or templates) are resampled with replacement; the statistic in each draw
    is the pooled (size-weighted) difference, matching `paired_contrast`'s `diff`. CI = percentile interval.
    p_boot is a two-sided shift-bootstrap p: (1 + #{|d* - d| >= |d|}) / (n_boot + 1), i.e. the bootstrap
    distribution of the estimation error is used as the null distribution. With few clusters (< 10)
    the percentile interval is too narrow and p_boot is anti-conservative (simulated null rejection at a
    nominal 5%: ~12% with 6 clusters, ~8% with 10); `few_clusters` flags that. For the p-value use
    `p_signflip` instead: the cluster-level sign-flip randomisation test (arm labels exchangeable within a
    cluster under H0), exact by enumerating all 2^k flips when k <= 16 clusters and `n_boot` seeded random
    flips otherwise. Its smallest possible p is 2 / 2^k, so with 4 clusters it cannot go below 0.125.
    Needs >= 2 clusters."""
    a, b = _as_binary(a_vec, "a_vec"), _as_binary(b_vec, "b_vec")
    if a.shape != b.shape:
        raise ValueError("a_vec and b_vec must have the same length")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    groups = _cluster_groups(cluster_ids, len(a))
    k = len(groups)
    if k < 2:
        raise ValueError("cluster bootstrap needs at least 2 clusters")
    diff = (a - b).astype(float)
    s = np.array([diff[g].sum() for g in groups])
    m = np.array([len(g) for g in groups], dtype=float)
    obs = float(s.sum() / m.sum())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for start in range(0, n_boot, 1000):
        stop = min(n_boot, start + 1000)
        idx = rng.integers(0, k, size=(stop - start, k))
        boot[start:stop] = s[idx].sum(axis=1) / m[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = (1 + int(np.sum(np.abs(boot - obs) >= abs(obs) - 1e-15))) / (n_boot + 1)
    return {"diff": obs, "ci_low": float(lo), "ci_high": float(hi), "p_boot": float(p),
            "p_signflip": _signflip_p(s, m, n_boot, seed), "signflip_exact": k <= _SIGNFLIP_EXACT_MAX,
            "n_clusters": k, "n_boot": int(n_boot), "seed": int(seed), "few_clusters": k < 10}


_SIGNFLIP_EXACT_MAX = 16


def _signflip_p(s: np.ndarray, m: np.ndarray, n_flip: int, seed: int) -> float:
    """Two-sided cluster sign-flip randomisation p for H0: no arm difference. `s` are the per-cluster sums of
    (a - b) and `m` the cluster sizes; the statistic is |sum_c e_c s_c| / sum_c m_c with e_c = +-1."""
    k, total = len(s), float(m.sum())
    obs = abs(float(s.sum())) / total
    if k <= _SIGNFLIP_EXACT_MAX:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=k)))
        return float(np.mean(np.abs(signs @ s) / total >= obs - 1e-12))
    signs = np.random.default_rng([abs(int(seed)), 1]).choice([-1.0, 1.0], size=(n_flip, k))
    return float((1 + np.sum(np.abs(signs @ s) / total >= obs - 1e-12)) / (n_flip + 1))


def auroc(y, score) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic with mid-ranks for ties
    (a constant score gives exactly 0.5). NaN if y has only one class."""
    y = _as_binary(y, "y")
    s = np.asarray(score, dtype=float)
    if s.shape != y.shape:
        raise ValueError("y and score must have the same length")
    if np.isnan(s).any():
        raise ValueError("score contains NaN")
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = stats.rankdata(s)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _boot_index_sampler(n: int, cluster_ids, rng: np.random.Generator) -> Callable[[], np.ndarray]:
    """Returns a function drawing one bootstrap sample of row indices (rows, or whole clusters)."""
    if cluster_ids is None:
        return lambda: rng.integers(0, n, size=n)
    groups = _cluster_groups(cluster_ids, n)
    k = len(groups)
    if k < 2:
        raise ValueError("cluster bootstrap needs at least 2 clusters")
    return lambda: np.concatenate([groups[i] for i in rng.integers(0, k, size=k)])


def auroc_bootstrap_ci(y, score, n_boot: int = 2000, seed: int = DEFAULT_SEED, cluster_ids=None,
                       alpha: float = 0.05) -> dict:
    """AUROC with a percentile bootstrap CI. Rows are resampled, or whole clusters when `cluster_ids` is
    given (use scenario / template ids for repeated draws of the same item). Draws containing a single
    class are skipped and counted (`n_skipped`)."""
    y = _as_binary(y, "y")
    s = np.asarray(score, dtype=float)
    point = auroc(y, s)
    draw = _boot_index_sampler(len(y), cluster_ids, np.random.default_rng(seed))
    vals = []
    for _ in range(n_boot):
        idx = draw()
        v = auroc(y[idx], s[idx])
        if not math.isnan(v):
            vals.append(v)
    out = {"auroc": point, "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()), "n_boot": int(n_boot),
           "n_skipped": int(n_boot - len(vals)), "seed": int(seed),
           "n_clusters": None if cluster_ids is None else len(set(map(str, cluster_ids)))}
    if vals:
        lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out.update(ci_low=float(lo), ci_high=float(hi))
    else:
        out.update(ci_low=float("nan"), ci_high=float("nan"))
    return out


def auroc_diff_bootstrap_ci(y, score_a, score_b, n_boot: int = 2000, seed: int = DEFAULT_SEED,
                            cluster_ids=None, alpha: float = 0.05) -> dict:
    """Paired bootstrap for AUROC(score_a) - AUROC(score_b) on the same items (same resamples for both
    scores). p_boot is the two-sided shift-bootstrap p used in `cluster_bootstrap_diff`."""
    y = _as_binary(y, "y")
    sa, sb = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    obs = auroc(y, sa) - auroc(y, sb)
    draw = _boot_index_sampler(len(y), cluster_ids, np.random.default_rng(seed))
    vals = []
    for _ in range(n_boot):
        idx = draw()
        va, vb = auroc(y[idx], sa[idx]), auroc(y[idx], sb[idx])
        if not (math.isnan(va) or math.isnan(vb)):
            vals.append(va - vb)
    out = {"auroc_a": auroc(y, sa), "auroc_b": auroc(y, sb), "diff": obs, "n_boot": int(n_boot),
           "n_skipped": int(n_boot - len(vals)), "seed": int(seed)}
    if vals:
        vals_arr = np.asarray(vals)
        lo, hi = np.percentile(vals_arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out.update(ci_low=float(lo), ci_high=float(hi),
                   p_boot=float((1 + np.sum(np.abs(vals_arr - obs) >= abs(obs) - 1e-15)) / (len(vals) + 1)))
    else:
        out.update(ci_low=float("nan"), ci_high=float("nan"), p_boot=float("nan"))
    return out


# ----------------------------------------------------------------------------------------------
# Firth logistic regression (exact: Jeffreys-prior penalized likelihood)
# ----------------------------------------------------------------------------------------------


def _check_design(X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError("X must be 2-D with one row per observation")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN/inf")
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        raise ValueError(f"design matrix is rank deficient (rank {rank} < {X.shape[1]} columns); "
                         "drop collinear or all-zero columns first (see fit_exp11_model)")


def _firth_pen_loglik(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """l(beta) + 0.5 log |X'WX| (the Firth-penalized log-likelihood)."""
    eta = X @ beta
    w = expit(eta) * expit(-eta)
    sign, logdet = np.linalg.slogdet((X.T * w) @ X)
    if sign <= 0:
        return -math.inf
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)) + 0.5 * logdet)


def _firth_newton(X: np.ndarray, y: np.ndarray, beta: np.ndarray, free: np.ndarray, maxit: int = 100,
                  tol: float = 1e-8, maxstep: float = 5.0) -> tuple[np.ndarray, float, bool, int]:
    """Maximise the Firth-penalized log-likelihood over the coordinates `free` (others stay at their
    value in `beta`). Newton with the modified score U*_j = sum_i (y_i - pi_i + h_i (1/2 - pi_i)) x_ij,
    h_i the hat-matrix diagonal of the FULL model, step cap and step halving on the penalized likelihood."""
    beta = beta.copy()
    ll = _firth_pen_loglik(X, y, beta)
    if len(free) == 0:  # nothing to optimise (profile of a one-parameter model)
        return beta, ll, True, 0
    for it in range(1, maxit + 1):
        eta = X @ beta
        pi = expit(eta)
        w = pi * (1 - pi)
        info = (X.T * w) @ X
        cov = np.linalg.inv(info)
        h = w * np.sum((X @ cov) * X, axis=1)
        score = X.T @ (y - pi + h * (0.5 - pi))
        delta = np.linalg.solve(info[np.ix_(free, free)], score[free])
        biggest = np.max(np.abs(delta)) if len(delta) else 0.0
        if biggest > maxstep:
            delta *= maxstep / biggest
        step = 1.0
        for _ in range(40):
            cand = beta.copy()
            cand[free] += step * delta
            ll_new = _firth_pen_loglik(X, y, cand)
            if ll_new >= ll - 1e-12:
                break
            step /= 2
        else:
            return beta, ll, False, it
        beta, ll = cand, ll_new
        if np.max(np.abs(step * delta)) < tol:
            return beta, ll, True, it
    return beta, ll, False, maxit


def _firth_profile_limit(X, y, beta_hat, ll_hat, j: int, direction: float, crit: float, se_j: float,
                         inner_ok: list) -> float:
    """Profile penalized-likelihood limit for coefficient j (direction +1 upper, -1 lower). Every inner
    constrained refit appends its convergence flag to `inner_ok`, so a failed refit is never silent."""
    free = np.array([k for k in range(len(beta_hat)) if k != j], dtype=int)

    def excess(b: float) -> float:
        start = beta_hat.copy()
        start[j] = b
        _, ll_b, ok, _ = _firth_newton(X, y, start, free)
        inner_ok.append(bool(ok))
        return 2.0 * (ll_hat - ll_b) - crit

    step = max(se_j, 0.1)
    inner = beta_hat[j]
    for k in range(40):
        outer = beta_hat[j] + direction * step * 2.0**k
        if excess(outer) > 0:
            lo, hi = sorted((inner, outer))
            return float(optimize.brentq(excess, lo, hi, xtol=1e-7))
        inner = outer
    return direction * math.inf


@_pinned
def firth_logit(X, y, names: Sequence[str] | None = None, alpha: float = 0.05, profile: bool = True,
                profile_cols: Sequence[int] | None = None, maxit: int = 100, tol: float = 1e-8,
                maxstep: float = 5.0) -> dict:
    """Firth's bias-reduced logistic regression (Firth 1993; Heinze & Schemper 2002).

    Maximises l(beta) + 0.5 log|X'WX| (Jeffreys-prior penalty) by Newton iterations on the modified score
    with step capping and step halving. Estimates are finite under (quasi-)complete separation. `X` is the
    full design matrix (add your own intercept column). Returns arrays: coef, se (from the inverse
    information at the penalized estimate), z, p_wald, ci_wald, cov, and - when `profile` - ci_pl (profile
    penalized-likelihood CI, asymmetric, the recommended interval under separation) and p_pl (penalized
    likelihood-ratio test of beta_j = 0). `profile_cols` restricts which columns get profile intervals
    (each costs about 20 constrained refits, so large models should select the terms they report).
    `profile_converged` is False if any constrained refit failed to converge (that interval may be too narrow)."""
    X = np.asarray(X, dtype=float)
    y = _as_binary(y, "y").astype(float)
    _check_design(X, y)
    n, p = X.shape
    names = list(names) if names is not None else [f"x{j}" for j in range(p)]
    if len(names) != p:
        raise ValueError(f"names has {len(names)} entries but X has {p} columns")
    beta, ll, ok, n_iter = _firth_newton(X, y, np.zeros(p), np.arange(p), maxit, tol, maxstep)
    eta = X @ beta
    pi = expit(eta)
    w = pi * (1 - pi)
    cov = np.linalg.inv((X.T * w) @ X)
    h = w * np.sum((X @ cov) * X, axis=1)
    se = np.sqrt(np.diag(cov))
    zc = stats.norm.ppf(1 - alpha / 2)
    z = beta / se
    out = {
        "names": names, "coef": beta, "se": se, "z": z, "p_wald": 2 * stats.norm.sf(np.abs(z)),
        "ci_wald": np.column_stack([beta - zc * se, beta + zc * se]), "cov": cov,
        "loglik": float(np.sum(y * eta - np.logaddexp(0.0, eta))), "loglik_pen": ll,
        "n": n, "converged": bool(ok), "n_iter": n_iter, "alpha": alpha,
        "max_abs_score": float(np.max(np.abs(X.T @ (y - pi + h * (0.5 - pi))))),
    }
    if profile:
        cols = range(p) if profile_cols is None else [int(j) for j in profile_cols]
        crit = stats.chi2.ppf(1 - alpha, 1)
        ci_pl = np.full((p, 2), np.nan)
        p_pl = np.full(p, np.nan)
        inner_ok: list[bool] = []
        for j in cols:
            ci_pl[j, 0] = _firth_profile_limit(X, y, beta, ll, j, -1.0, crit, se[j], inner_ok)
            ci_pl[j, 1] = _firth_profile_limit(X, y, beta, ll, j, +1.0, crit, se[j], inner_ok)
            start = beta.copy()
            start[j] = 0.0
            _, ll0, ok0, _ = _firth_newton(X, y, start, np.array([k for k in range(p) if k != j], dtype=int))
            inner_ok.append(bool(ok0))
            p_pl[j] = float(stats.chi2.sf(max(0.0, 2 * (ll - ll0)), 1))
        out["ci_pl"] = ci_pl
        out["p_pl"] = p_pl
        out["profile_converged"] = all(inner_ok)
    return out


# ----------------------------------------------------------------------------------------------
# Firth-type crossed random-intercept logistic model (APPROXIMATE)
# ----------------------------------------------------------------------------------------------


class _Glmm:
    """Pieces shared by the mixed-model fits: design, sparse indicator matrix Z, group bookkeeping."""

    def __init__(self, X: np.ndarray, y: np.ndarray, groups: Mapping[str, Sequence]):
        n = len(y)
        self.X, self.y = X, y
        self.names, self.levels, cols, offset = [], [], [], 0
        for name, values in groups.items():
            vals = np.asarray([str(v) for v in values])
            if len(vals) != n:
                raise ValueError(f"group '{name}' must have one entry per observation")
            levels, codes = np.unique(vals, return_inverse=True)
            self.names.append(name)
            self.levels.append(levels)
            cols.append(codes + offset)
            offset += len(levels)
        self.q = offset
        self.codes = np.column_stack(cols)  # (n, K): column of Z holding each row's level, per factor
        rows = np.repeat(np.arange(n), len(cols))
        self.Z = sparse.csr_matrix((np.ones(len(rows)), (rows, self.codes.ravel())), shape=(n, offset))
        self.sizes = np.array([len(lv) for lv in self.levels])
        self.comp_of_col = np.repeat(np.arange(len(self.names)), self.sizes)

    def zdot(self, v: np.ndarray) -> np.ndarray:
        """Z @ v for a vector or matrix v (gather-and-sum; cheaper than sparse @ dense)."""
        return v[self.codes].sum(axis=1)

    def dinv(self, sigma: np.ndarray) -> np.ndarray:
        return 1.0 / (sigma[self.comp_of_col] ** 2)

    def evaluate(self, beta: np.ndarray, u: np.ndarray, dinv: np.ndarray) -> dict:
        """Everything needed at (beta, u): Q = l - u'D^-1u/2 + 0.5 log|X'V^-1X| and its pieces."""
        X, y, Z = self.X, self.y, self.Z
        eta = X @ beta + self.zdot(u)
        pi = expit(eta)
        w = pi * expit(-eta)
        wx = X * w[:, None]
        xtwx = X.T @ wx
        ztwx = np.asarray(Z.T @ wx)
        ztwz = (Z.T @ sparse.diags(w) @ Z).toarray()
        huu = ztwz + np.diag(dinv)
        cho = cho_factor(huu)
        logdet_huu = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
        b = cho_solve(cho, ztwx)
        info = xtwx - ztwx.T @ b                       # = X'V^-1 X, V = W^-1 + Z D Z'
        sign, logdet_info = np.linalg.slogdet(info)
        ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        pen_u = 0.5 * float(u @ (dinv * u))
        q_val = ll - pen_u + 0.5 * logdet_info if sign > 0 else -math.inf
        return {"eta": eta, "pi": pi, "w": w, "ztwx": ztwx, "cho": cho, "b": b, "info": info,
                "logdet_huu": logdet_huu, "logdet_info": logdet_info, "ll": ll, "pen_u": pen_u, "Q": q_val}

    def score(self, st: dict, u: np.ndarray, dinv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact gradient of Q with respect to (beta, u) and (X'V^-1 X)^-1.

        dQ/d eta_i = y_i - pi_i + h_i (1/2 - pi_i), with h_i = W_i r_i' (X'V^-1 X)^-1 r_i and
        r_i = x_i - (random-effect projection of X)_i; for sigma -> 0 this is Firth's hat-value h_i."""
        info_inv = np.linalg.inv(st["info"])
        r = self.X - self.zdot(st["b"])
        h = st["w"] * np.sum((r @ info_inv) * r, axis=1)
        resid = self.y - st["pi"] + h * (0.5 - st["pi"])
        return self.X.T @ resid, np.asarray(self.Z.T @ resid).ravel() - dinv * u, info_inv

    def newton_step(self, st: dict, u: np.ndarray, dinv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Newton direction on Q (Fisher-type Hessian = Henderson mixed-model matrix). Returns (d_beta, d_u, info_inv)."""
        g_beta, g_u, info_inv = self.score(st, u, dinv)
        d_beta = info_inv @ (g_beta - st["ztwx"].T @ cho_solve(st["cho"], g_u))
        d_u = cho_solve(st["cho"], g_u - st["ztwx"] @ d_beta)
        return d_beta, d_u, info_inv

    def fit_mode(self, sigma: np.ndarray, start: tuple[np.ndarray, np.ndarray] | None = None, maxit: int = 100,
                 tol: float = 1e-8, maxstep: float = 5.0) -> dict:
        """Penalised joint mode (beta, u) for fixed standard deviations `sigma` (one per grouping factor)."""
        dinv = self.dinv(sigma)
        beta = np.zeros(self.X.shape[1]) if start is None else start[0].copy()
        u = np.zeros(self.q) if start is None else start[1].copy()
        st = self.evaluate(beta, u, dinv)
        converged, it = False, 0
        for it in range(1, maxit + 1):
            d_beta, d_u, info_inv = self.newton_step(st, u, dinv)
            d = np.concatenate([d_beta, d_u])
            biggest = float(np.max(np.abs(d)))
            if biggest > maxstep:
                d *= maxstep / biggest
            step, accepted = 1.0, False
            for _ in range(40):
                cb, cu = beta + step * d[: len(beta)], u + step * d[len(beta):]
                cand = self.evaluate(cb, cu, dinv)
                if cand["Q"] >= st["Q"] - 1e-10:
                    accepted = True
                    break
                step /= 2
            if not accepted:
                break
            beta, u, st = cb, cu, cand
            if float(np.max(np.abs(step * d))) < tol:
                converged = True
                break
        g_beta, g_u, info_inv = self.score(st, u, dinv)
        return {"beta": beta, "u": u, "state": st, "info_inv": info_inv, "converged": converged, "n_iter": it,
                "sigma": np.asarray(sigma, dtype=float),
                "max_abs_score": float(np.max(np.abs(np.concatenate([g_beta, g_u]))))}

    def laplace(self, fit: dict, reml: bool) -> float:
        """Laplace-approximate marginal log-likelihood of sigma at the penalised mode.
        = l - u'D^-1u/2 - 0.5 log|Huu| - sum_k n_k log sigma_k   [- 0.5 log|X'V^-1X| if reml]
        Integrating beta against the Jeffreys prior |X'V^-1X|^(1/2) at Laplace order cancels the
        log|X'V^-1X| terms, which is why the default (reml=False) is the Firth-consistent criterion."""
        st = fit["state"]
        val = st["ll"] - st["pen_u"] - 0.5 * st["logdet_huu"] - float(np.sum(self.sizes * np.log(fit["sigma"])))
        if reml:
            val -= 0.5 * st["logdet_info"]
        return val


_RE_APPROXIMATION = (
    "Firth-type penalized quasi-likelihood with Laplace-estimated variance components (NOT an exact penalized "
    "GLMM likelihood). For fixed random-intercept SDs sigma, (beta, u) maximise "
    "l(beta,u) - u'D^-1 u/2 + 0.5 log|X'V^-1 X|, V = W^-1 + Z D Z' (Jeffreys penalty of the working linear "
    "mixed model, with its exact gradient). sigma maximises the Laplace marginal likelihood evaluated at that "
    "mode (grid screen + Nelder-Mead in log sigma, floored). Standard errors are the Fisher-type "
    "sqrt(diag((X'V^-1 X)^-1)) at sigma-hat: they ignore the uncertainty in sigma-hat and the Laplace error "
    "(large with few binary observations per random-effect level), so CIs are somewhat too narrow, and Wald "
    "intervals for cells with (near-)zero counts are unreliable (compare firth_logit profile intervals). "
    "Because (beta, u) are fitted jointly at the mode (PQL-like), fixed effects are shrunk toward 0 relative to "
    "exact Gauss-Hermite maximum likelihood when clusters are small: measured slope shrinkage ~7% with 8 binary "
    "observations per cluster (sigma 1), ~1% with 72 per cluster (sigma 0.6)."
)


@_pinned
def firth_random_intercept_logit(X, y, groups: Mapping[str, Sequence], names: Sequence[str] | None = None,
                                 alpha: float = 0.05, sigma: Mapping[str, float] | None = None,
                                 sigma_floor: float = 0.05, sigma_max: float = 4.0, reml: bool = False,
                                 maxit: int = 100, tol: float = 1e-8, maxstep: float = 5.0) -> dict:
    """Approximate Firth penalized logistic model with crossed random intercepts,
    logit P(y_i = 1) = x_i'beta + sum_k u_{k, g_k(i)},  u_k ~ N(0, sigma_k^2),
    one intercept per level of every entry of `groups` (e.g. {"template_id": ..., "scenario_id": ...}).
    Nested factors work if their ids are globally unique (scenario ids that repeat across templates must be
    made unique by the caller). Factors with a single level are dropped and listed in `dropped_groups`.

    THIS IS AN APPROXIMATION of the spec's "Firth penalized random-intercept model": it is not an exact
    penalized-likelihood GLMM fit. What is computed (details in the returned `approximation` string):
      1. For given sigma, (beta, u) maximise l(beta,u) - sum_k |u_k|^2/(2 sigma_k^2) + 0.5 log|X'V^-1X|
         with V = W^-1 + Z D Z' (so the Jeffreys/Firth penalty acts on the fixed effects of the working
         mixed model; with sigma -> 0 this is exactly `firth_logit`). Newton with exact penalized score and
         step halving; finite under separation (all-zero cells), as for Firth GLMs.
      2. sigma is chosen by maximising the Laplace-approximate marginal likelihood at that mode
         (`reml=True` adds -0.5 log|X'V^-1X|, a REML-like variant that is less biased downward for sigma
         but not Firth-consistent). Search = coarse grid then Nelder-Mead on log sigma, bounded to
         [sigma_floor, sigma_max]; `at_floor` flags components stuck at the floor (variance not identified
         or ~0). `sigma={"scenario_id": 0.5}` fixes components instead of estimating them.
      3. se = sqrt(diag((X'V^-1X)^-1)) at sigma-hat, Wald z intervals. No profile intervals: for cells with
         (near-)zero counts the Wald interval is unreliable, so cross-check such terms with `firth_logit`
         profile intervals (fit_exp11_model(..., random_intercepts=False)).
    Joint-mode estimation is PQL-like and shrinks fixed effects toward 0 when clusters are small (checked
    against exact Gauss-Hermite ML: ~7% for a slope at 8 binary observations per cluster, ~1% at 72).
    Returns coef, se, z, p_wald, ci_wald, cov, names, sigma (dict), at_floor (dict), random_effects
    (dict group -> level -> mode), loglik (conditional), laplace_loglik, converged flags, n_groups,
    dropped_groups, and 'approximation': description string."""
    X = np.asarray(X, dtype=float)
    y = _as_binary(y, "y").astype(float)
    _check_design(X, y)
    p = X.shape[1]
    names = list(names) if names is not None else [f"x{j}" for j in range(p)]
    if len(names) != p:
        raise ValueError(f"names has {len(names)} entries but X has {p} columns")
    kept, dropped = {}, []
    for gname, vals in groups.items():
        if len(set(map(str, vals))) < 2:
            dropped.append(gname)
        else:
            kept[gname] = vals
    if not kept:
        raise ValueError("no grouping factor with >= 2 levels; use firth_logit instead")
    model = _Glmm(X, y, kept)
    unknown = [k for k in (sigma or {}) if k not in groups]
    if unknown:
        raise ValueError(f"sigma given for unknown grouping factor(s) {unknown}; groups are {list(groups)}")
    fixed = {k: float(v) for k, v in (sigma or {}).items() if k in kept}
    if any(v <= 0 for v in fixed.values()):
        raise ValueError("fixed sigma values must be positive")
    free_idx = [i for i, g in enumerate(model.names) if g not in fixed]
    base = np.array([fixed.get(g, 1.0) for g in model.names], dtype=float)
    lo_log, hi_log = math.log(sigma_floor), math.log(sigma_max)
    cache: dict[tuple, dict] = {}
    last = {"start": None}

    def fit_at(free_log: Sequence[float]) -> dict:
        s = base.copy()
        for i, v in zip(free_idx, free_log):
            s[i] = math.exp(min(max(v, lo_log), hi_log))
        key = tuple(np.round(s, 7))
        if key not in cache:
            f = model.fit_mode(s, last["start"], maxit, tol, maxstep)
            cache[key] = f
            if f["converged"]:
                last["start"] = (f["beta"], f["u"])
        return cache[key]

    if free_idx:
        def neg_crit(free_log: np.ndarray) -> float:
            return -model.laplace(fit_at(free_log), reml)

        grid = [lo_log, math.log(0.25), math.log(0.5), 0.0, math.log(2.0)]
        grid = [g for g in grid if lo_log <= g <= hi_log]
        if len(free_idx) > 2:
            grid = grid[1:4]
        best = min(itertools.product(grid, repeat=len(free_idx)), key=lambda t: neg_crit(np.array(t)))
        if len(free_idx) == 1:
            res = optimize.minimize_scalar(lambda v: neg_crit(np.array([v])), bounds=(lo_log, hi_log),
                                           method="bounded", options={"xatol": 1e-3})
            cands = [np.array([res.x]), np.array(best)]
            x_best = min(cands, key=neg_crit)
        else:
            x0 = np.array(best)
            simplex = np.vstack([x0] + [x0 + 0.5 * np.eye(len(x0))[i] for i in range(len(x0))])
            res = optimize.minimize(neg_crit, x0, method="Nelder-Mead",
                                    options={"xatol": 1e-3, "fatol": 1e-6, "initial_simplex": simplex})
            x_best = min([res.x, np.array(best)], key=neg_crit)
        final = fit_at(x_best)
    else:
        final = fit_at([])

    beta, u = final["beta"], final["u"]
    cov = final["info_inv"]
    se = np.sqrt(np.diag(cov))
    zc = stats.norm.ppf(1 - alpha / 2)
    z = beta / se
    sig = final["sigma"]
    re, offset = {}, 0
    for gname, levels in zip(model.names, model.levels):
        re[gname] = {str(lv): float(u[offset + i]) for i, lv in enumerate(levels)}
        offset += len(levels)
    at_floor = {g: bool(sig[i] <= sigma_floor * (1 + 1e-6)) for i, g in enumerate(model.names) if i in free_idx}
    return {
        "names": names, "coef": beta, "se": se, "z": z, "p_wald": 2 * stats.norm.sf(np.abs(z)),
        "ci_wald": np.column_stack([beta - zc * se, beta + zc * se]), "cov": cov,
        "sigma": {g: float(sig[i]) for i, g in enumerate(model.names)}, "at_floor": at_floor,
        "sigma_fixed": sorted(fixed), "random_effects": re,
        "n": len(y), "n_groups": {g: int(len(lv)) for g, lv in zip(model.names, model.levels)},
        "dropped_groups": dropped, "loglik": final["state"]["ll"],
        "laplace_loglik": model.laplace(final, reml), "variance_method": "laplace_reml" if reml else "laplace_ml",
        "converged": bool(final["converged"]), "n_iter": final["n_iter"], "alpha": alpha,
        "approximation": _RE_APPROXIMATION,
    }


# ----------------------------------------------------------------------------------------------
# design matrices and the Exp 1.1 model
# ----------------------------------------------------------------------------------------------

EXP11_FACTORS = ("arm", "cost_regime", "consequence", "boundary")
EXP11_REF_LEVELS = {"arm": "intact", "cost_regime": "low", "consequence": "harmless", "boundary": "implicit"}


def design_matrix_from_cells(records: Sequence[Mapping], factors: Sequence[str],
                             ref_levels: Mapping[str, object] | None = None,
                             interactions: int | None = None, intercept: bool = True) -> tuple[np.ndarray, list[str]]:
    """Treatment-coded design matrix from trial records (list of dicts).

    Columns: Intercept, main effects `factor[T.level]` for every non-reference level (reference first, other
    levels sorted by their string form), then all interactions up to order `interactions` (default: all,
    i.e. the full `f1 * f2 * ...` model), named like `a[T.x]:b[T.y]`. Interaction columns for level
    combinations never observed are all zero; the caller removes them (fit_exp11_model does).
    `ref_levels` defaults to the first sorted level per factor. Returns (X, names)."""
    ref_levels = dict(ref_levels or {})
    levels: dict[str, list[str]] = {}
    for f in factors:
        seen = sorted({str(r[f]) for r in records})
        ref = str(ref_levels.get(f, seen[0]))
        if ref not in seen:
            raise ValueError(f"reference level '{ref}' of '{f}' does not occur in the records")
        levels[f] = [ref] + [lv for lv in seen if lv != ref]
    n = len(records)
    indicator: dict[tuple[str, str], np.ndarray] = {}
    for f in factors:
        vals = np.asarray([str(r[f]) for r in records])
        for lv in levels[f][1:]:
            indicator[(f, lv)] = (vals == lv).astype(float)
    cols = [np.ones(n)] if intercept else []
    names = ["Intercept"] if intercept else []
    max_order = len(factors) if interactions is None else int(interactions)
    for order in range(1, max_order + 1):
        for combo in itertools.combinations(factors, order):
            for lvs in itertools.product(*[levels[f][1:] for f in combo]):
                col = np.ones(n)
                for f, lv in zip(combo, lvs):
                    col = col * indicator[(f, lv)]
                cols.append(col)
                names.append(":".join(f"{f}[T.{lv}]" for f, lv in zip(combo, lvs)))
    return np.column_stack(cols), names


def _drop_dependent_columns(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str], list[dict]]:
    """Drop all-zero columns and columns linearly dependent on earlier ones (so later, higher-order
    interaction terms go first). Returns (X, names, [{"term", "reason"}])."""
    gram = X.T @ X
    kept: list[int] = []
    dropped: list[dict] = []
    for j in range(X.shape[1]):
        if gram[j, j] == 0:
            dropped.append({"term": names[j], "reason": "all zero (level combination never observed)"})
            continue
        if kept:
            g_k = gram[np.ix_(kept, kept)]
            resid = gram[j, j] - gram[kept, j] @ np.linalg.solve(g_k, gram[kept, j])
        else:
            resid = gram[j, j]
        if resid < 1e-9 * gram[j, j]:
            dropped.append({"term": names[j], "reason": "collinear with earlier columns"})
        else:
            kept.append(j)
    return X[:, kept], [names[j] for j in kept], dropped


def contrast_from_fit(fit: Mapping, weights, alpha: float = 0.05) -> dict:
    """Wald estimate, SE, CI and p for the linear combination w'beta of a fitted model's coefficients
    (needs fit['coef'], fit['cov'], fit['names']). `weights` is a {term: weight} dict or a full vector."""
    names = list(fit["names"])
    if isinstance(weights, Mapping):
        w = np.zeros(len(names))
        for term, val in weights.items():
            w[names.index(term)] = val
    else:
        w = np.asarray(weights, dtype=float)
    est = float(w @ np.asarray(fit["coef"]))
    se = float(math.sqrt(max(0.0, w @ np.asarray(fit["cov"]) @ w)))
    zc = stats.norm.ppf(1 - alpha / 2)
    z = est / se if se > 0 else float("nan")
    return {"estimate": est, "se": se, "ci_low": float(est - zc * se), "ci_high": float(est + zc * se), "z": z,
            "p": float(2 * stats.norm.sf(abs(z))) if se > 0 else float("nan")}


def fit_exp11_model(records: Sequence[Mapping], outcome: str | Callable[[Mapping], object] = "direct_taken",
                    factors: Sequence[str] = EXP11_FACTORS, ref_levels: Mapping[str, object] | None = None,
                    interactions: int | None = None, group_keys: Sequence[str] = ("template_id", "scenario_id"),
                    random_intercepts: bool = True, alpha: float = 0.05, profile_cols: Sequence[str] | None = None,
                    **fit_kwargs) -> dict:
    """The Exp 1.1 model  outcome ~ arm * cost_regime * consequence * boundary  [+ (1|template_id) + (1|scenario_id)].

    `records` are episode records (filter to the family / arms / cot_mode you want first). `outcome` is a
    record key or a function record -> 0/1/bool; None or NaN drops the row (counted in `n_dropped_missing`);
    any other value (a string, 0.7, 2) raises. With `random_intercepts=True` this is `firth_random_intercept_logit` (approximate,
    Wald CIs); with False it is the exact `firth_logit` with profile CIs for the terms in `profile_cols`
    (default: all) and no random effects. Unobserved / collinear columns are dropped and reported in
    `dropped_columns`. Reference levels default to EXP11_REF_LEVELS. Returns JSON-friendly `terms`
    (estimate, se, ci, odds ratio, p) plus names / coef / cov for `contrast_from_fit`."""
    refs = {**EXP11_REF_LEVELS, **(ref_levels or {})}
    get = outcome if callable(outcome) else (lambda r, k=outcome: r.get(k))
    rows, ys = [], []
    for r in records:
        v = get(r)
        if v is None or (isinstance(v, (float, np.floating)) and math.isnan(v)):
            continue
        rows.append(r)
        ys.append(v)
    n_dropped = len(records) - len(rows)
    if not rows:
        raise ValueError("no records with an outcome")
    y = _as_binary(ys, "outcome")  # raises on anything but 0/1/bool: a 0.7 must never be truncated to 0
    X, names = design_matrix_from_cells(rows, factors, refs, interactions)
    X, names, dropped = _drop_dependent_columns(X, names)
    if random_intercepts:
        fit = firth_random_intercept_logit(X, y, {g: [r[g] for r in rows] for g in group_keys}, names=names,
                                           alpha=alpha, **fit_kwargs)
        lows, highs, ci_kind = fit["ci_wald"][:, 0], fit["ci_wald"][:, 1], "wald"
        pvals = fit["p_wald"]
    else:
        cols = None if profile_cols is None else [names.index(t) for t in profile_cols]
        fit = firth_logit(X, y, names=names, alpha=alpha, profile_cols=cols, **fit_kwargs)
        lows, highs, ci_kind, pvals = fit["ci_pl"][:, 0], fit["ci_pl"][:, 1], "profile", fit["p_pl"]
    terms = []
    for j, term in enumerate(names):
        est = float(fit["coef"][j])
        lo, hi = float(lows[j]), float(highs[j])
        terms.append({"term": term, "estimate": est, "se": float(fit["se"][j]), "ci_low": lo, "ci_high": hi,
                      "odds_ratio": math.exp(min(max(est, -50), 50)),
                      "or_ci_low": math.exp(min(max(lo, -50), 50)) if not math.isnan(lo) else float("nan"),
                      "or_ci_high": math.exp(min(max(hi, -50), 50)) if not math.isnan(hi) else float("nan"),
                      "p": float(pvals[j]), "p_wald": float(fit["p_wald"][j])})
    out = {"terms": terms, "ci_method": ci_kind, "n": len(y), "n_events": int(y.sum()),
           "n_dropped_missing": n_dropped, "dropped_columns": dropped, "outcome": outcome if isinstance(outcome, str) else "callable",
           "ref_levels": {f: str(refs.get(f)) for f in factors}, "factors": list(factors),
           "names": names, "coef": fit["coef"], "cov": fit["cov"], "converged": fit["converged"],
           "random_intercepts": random_intercepts}
    if random_intercepts:
        out.update(sigma=fit["sigma"], at_floor=fit["at_floor"], n_groups=fit["n_groups"],
                   dropped_groups=fit["dropped_groups"], approximation=fit["approximation"],
                   variance_method=fit["variance_method"])
    else:
        out["approximation"] = None
        out["profile_converged"] = fit.get("profile_converged")
    return out
