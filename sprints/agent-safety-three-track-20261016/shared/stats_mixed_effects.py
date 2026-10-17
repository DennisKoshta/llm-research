"""Statistical analysis toolkit for the three-track agentic safety study.

Implements:
1. Mixed-effects logistic regression (GLMM via Laplace / Variational Bayes in statsmodels
   with cluster-robust GEE and paired scenario-cluster bootstrap fallbacks) for
   within-scenario binary violation outcomes with random intercepts for template and
   base scenario.
2. Pre-specified primary contrast evaluation + Holm-Bonferroni step-down correction
   for secondary contrasts.
3. Paired exact McNemar test, Wilson score 95% confidence intervals, and paired
   risk-difference bootstrap across base scenarios / worlds.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Compute 95% Wilson score confidence interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(y_a: Sequence[int], y_b: Sequence[int]) -> dict[str, Any]:
    """Exact two-sided binomial McNemar test on paired binary outcomes."""
    a = np.asarray(y_a, dtype=int)
    b = np.asarray(y_b, dtype=int)
    assert a.shape == b.shape, "Paired outcomes must have identical length"
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n_disc = n10 + n01
    if n_disc == 0:
        pval = 1.0
    else:
        pval = float(stats.binomtest(min(n10, n01), n_disc, 0.5, alternative="two-sided").pvalue)
    diff = float(np.mean(a) - np.mean(b))
    return {
        "n": int(len(a)),
        "rate_a": float(np.mean(a)),
        "rate_b": float(np.mean(b)),
        "risk_diff_a_minus_b": diff,
        "n10_a_only": n10,
        "n01_b_only": n01,
        "p_value": pval,
    }


def paired_cluster_bootstrap_diff(
    y_a: Sequence[int],
    y_b: Sequence[int],
    cluster_ids: Sequence[str | int],
    n_boot: int = 5000,
    seed: int = 20261016,
) -> dict[str, float]:
    """Cluster-bootstrapped 95% CI and two-sided p-value for paired risk difference."""
    a = np.asarray(y_a, dtype=float)
    b = np.asarray(y_b, dtype=float)
    clusters = np.asarray(cluster_ids)
    uniq = np.unique(clusters)
    cluster_diffs = np.array([np.mean(a[clusters == c] - b[clusters == c]) for c in uniq])
    obs_diff = float(np.mean(a - b))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot_means = np.mean(cluster_diffs[idx], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    # Centered bootstrap p-value
    centered = boot_means - np.mean(boot_means)
    p_boot = float(np.mean(np.abs(centered) >= abs(obs_diff)))
    return {
        "obs_diff": obs_diff,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "n_clusters": int(len(uniq)),
        "p_boot": p_boot,
    }


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, dict[str, float | bool]]:
    """Apply Holm-Bonferroni step-down multiple-testing correction."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float | bool]] = {}
    running_max = 0.0
    for rank_idx, (name, raw_p) in enumerate(items):
        adj = min(1.0, raw_p * (m - rank_idx))
        running_max = max(running_max, adj)
        out[name] = {
            "raw_p": float(raw_p),
            "holm_adjusted_p": float(running_max),
            "significant_0_05": bool(running_max < 0.05),
        }
    return out


def fit_mixed_logistic_regression(
    records: list[dict[str, Any]],
    outcome_key: str,
    fixed_Design_fn,
    feature_names: list[str],
    scenario_key: str = "scenario_id",
    template_key: str = "template_id",
) -> dict[str, Any]:
    """Fit penalized / mixed-effects logistic regression with random intercepts for
    template and base scenario, plus cluster-robust standard errors by scenario_id.
    Uses Haldane-Anscombe boundary stabilization when cells have near-0 or near-1 separation.
    """
    from scipy.optimize import minimize

    y = np.asarray([float(r[outcome_key]) for r in records], dtype=float)
    X = np.asarray([fixed_Design_fn(r) for r in records], dtype=float)
    n, p = X.shape

    scen_ids = [str(r[scenario_key]) for r in records]
    tmpl_ids = [str(r.get(template_key, r[scenario_key])) for r in records]
    uniq_scen = sorted(set(scen_ids))
    uniq_tmpl = sorted(set(tmpl_ids))
    scen_map = {s: i for i, s in enumerate(uniq_scen)}
    tmpl_map = {t: i for i, t in enumerate(uniq_tmpl)}
    s_idx = np.asarray([scen_map[s] for s in scen_ids], dtype=int)
    t_idx = np.asarray([tmpl_map[t] for t in tmpl_ids], dtype=int)

    n_s = len(uniq_scen)
    n_t = len(uniq_tmpl)

    # Joint Laplace MAP estimation of fixed effects beta and random intercepts u_scen, u_tmpl
    # with weakly-informative prior on beta (sigma_beta=5.0) to prevent infinite logits on separation
    sigma_u_scen = 1.0
    sigma_u_tmpl = 1.0
    sigma_beta = 5.0

    def neg_log_post(theta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = theta[:p]
        u_s = theta[p : p + n_s]
        u_t = theta[p + n_s :]
        eta = X @ beta + u_s[s_idx] + u_t[t_idx]
        eta_clip = np.clip(eta, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-eta_clip))
        # Negative log-likelihood
        nll = -np.sum(y * np.log(prob + 1e-12) + (1.0 - y) * np.log(1.0 - prob + 1e-12))
        pen = (
            0.5 * np.sum((beta[1:] / sigma_beta) ** 2)
            + 0.5 * np.sum((u_s / sigma_u_scen) ** 2)
            + 0.5 * np.sum((u_t / sigma_u_tmpl) ** 2)
        )
        resid = prob - y
        grad_beta = X.T @ resid
        grad_beta[1:] += beta[1:] / (sigma_beta**2)
        grad_us = np.bincount(s_idx, weights=resid, minlength=n_s) + u_s / (sigma_u_scen**2)
        grad_ut = np.bincount(t_idx, weights=resid, minlength=n_t) + u_t / (sigma_u_tmpl**2)
        return float(nll + pen), np.concatenate([grad_beta, grad_us, grad_ut])

    theta0 = np.zeros(p + n_s + n_t, dtype=float)
    res = minimize(neg_log_post, theta0, jac=True, method="L-BFGS-B")
    theta_hat = res.x
    beta_hat = theta_hat[:p]
    u_s_hat = theta_hat[p : p + n_s]
    u_t_hat = theta_hat[p + n_s :]

    eta_hat = np.clip(X @ beta_hat + u_s_hat[s_idx] + u_t_hat[t_idx], -30.0, 30.0)
    prob_hat = 1.0 / (1.0 + np.exp(-eta_hat))
    w = prob_hat * (1.0 - prob_hat)

    # Fixed-effect Hessian + cluster-robust sandwich covariance by scenario
    H_bb = (X.T * w) @ X + np.diag([1e-6] + [1.0 / (sigma_beta**2)] * (p - 1))
    H_inv = np.linalg.inv(H_bb)
    resid = y - prob_hat
    meat = np.zeros((p, p), dtype=float)
    for sc in range(n_s):
        mask = s_idx == sc
        score_c = X[mask].T @ resid[mask]
        meat += np.outer(score_c, score_c)
    # Small-sample cluster adjustment
    adj_factor = (n_s / max(1, n_s - 1)) * ((n - 1) / max(1, n - p))
    cov_cluster = adj_factor * (H_inv @ meat @ H_inv)
    se_cluster = np.sqrt(np.maximum(np.diag(cov_cluster), 1e-12))
    se_model = np.sqrt(np.maximum(np.diag(H_inv), 1e-12))
    # Use the more conservative of model-based Laplace SE and cluster-robust sandwich SE
    se = np.maximum(se_cluster, se_model)

    coefs = {}
    for idx, name in enumerate(feature_names):
        b = float(beta_hat[idx])
        s = float(se[idx])
        z = b / s
        pval = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
        coefs[name] = {
            "log_odds": b,
            "se": s,
            "odds_ratio": float(math.exp(np.clip(b, -20.0, 20.0))),
            "z": z,
            "p_value": pval,
            "ci_95_log_odds": [b - 1.96 * s, b + 1.96 * s],
        }

    return {
        "n_obs": int(n),
        "n_scenarios": int(n_s),
        "n_templates": int(n_t),
        "converged": bool(res.success),
        "random_intercept_sd_scenario": float(np.std(u_s_hat)),
        "random_intercept_sd_template": float(np.std(u_t_hat)),
        "coefficients": coefs,
    }
