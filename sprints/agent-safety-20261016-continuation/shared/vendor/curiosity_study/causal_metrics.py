"""Semantic readout margins and paired world-level uncertainty estimates."""

import itertools
import numpy as np


def margin(kind, assignment, logits, expected):
    if kind in ("new", "repeat"):
        sign = 1 if kind == "new" else -1
        return sign * (logits["Yes"] - logits["No"]), "information_gain_yes_minus_no_with_polarity"
    if kind == "instrument":
        services = list(itertools.permutations(("report", "pattern", "stop")))[assignment["index"] % 6]
        labels = {service: "ABC"[i] for i, service in enumerate(services)}
        return logits[labels["report"]] - logits[labels["stop"]], "report_minus_stop"
    if kind in ("memory", "count"):
        return logits[expected] - max(
            v for k, v in logits.items() if k != expected
        ), "correct_minus_best_alternative"
    raise ValueError("Unsupported outcome kind")


def paired_summary(world_values, seed=2026101412, bootstrap=4000):
    values = np.asarray(world_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected one finite paired mean per world")
    if not len(values):
        return dict(worlds=0, mean=None, ci95=None)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(bootstrap, len(values)))].mean(1)
    return dict(
        worlds=len(values), mean=float(values.mean()), ci95=np.quantile(means, [0.025, 0.975]).tolist()
    )


def zero_event_upper_bound(worlds, alpha=0.05):
    """One-sided exact binomial bound for zero world-level failure events."""
    return 1 - alpha ** (1 / worlds) if worlds else None
