"""World-paired factorial contrasts with explicit missingness and split boundaries."""

from collections import defaultdict
import numpy as np

CELL_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
GAIN_WEIGHTS = np.array([-0.5, 0.5, 0.5, -0.5])
KNOWN_WEIGHTS = np.array([-0.5, -0.5, 0.5, 0.5])
QUERY_WEIGHTS = np.array([-0.5, 0.5, -0.5, 0.5])


def group_worlds(assignments):
    groups = defaultdict(dict)
    for row, a in enumerate(assignments):
        key = (a["family"], a["index"], a["kind"])
        cell = (a["known"][0], a["query"])
        if cell in groups[key]:
            raise ValueError("Duplicate world cell")
        groups[key][cell] = row
    if any(set(cells) != set(CELL_ORDER) for cells in groups.values()):
        raise ValueError("Incomplete factorial assignment")
    return {key: np.array([cells[cell] for cell in CELL_ORDER]) for key, cells in groups.items()}


def world_summaries(assignments, vectors):
    """Return whole-world contrasts; any missing cell makes that world-position missing."""
    groups = group_worlds(assignments)
    result = {}
    for key, indices in groups.items():
        values = vectors[indices].astype(np.float64)
        complete = np.isfinite(values).all(axis=(0, -1))
        summaries = {}
        for name, weights in [
            ("gain", GAIN_WEIGHTS),
            ("known", KNOWN_WEIGHTS),
            ("query", QUERY_WEIGHTS),
            ("center", np.full(4, 0.25)),
        ]:
            summaries[name] = np.einsum("n,n...->...", weights, values)
            summaries[name][~complete] = np.nan
        result[key] = summaries
    return groups, result


def fit_source(summaries, families, kinds, indices=range(50), answer_contrast=False):
    """Equal family/kind weighting and paired availability across source polarities."""
    family_values, counts = [], []
    for family in families:
        worlds = []
        for index in indices:
            selected = [summaries[(family, index, kind)] for kind in kinds]
            both = np.logical_and.reduce([np.isfinite(s["gain"]).all(-1) for s in selected])
            combined = {}
            for name in ("gain", "known", "query", "center"):
                stack = np.stack([s[name] for s in selected])
                weights = (
                    np.array([1, -1]) / 2
                    if answer_contrast and name == "gain"
                    else np.full(len(kinds), 1 / len(kinds))
                )
                combined[name] = np.einsum("n,n...->...", weights, stack)
                combined[name][~both] = np.nan
            worlds.append(combined)
        count = np.isfinite(np.stack([w["gain"] for w in worlds])).all(-1).sum(0)
        averaged = {}
        for name in ("gain", "known", "query", "center"):
            summed = np.nansum(np.stack([w[name] for w in worlds]), axis=0)
            averaged[name] = np.divide(
                summed, count[..., None], out=np.full_like(summed, np.nan), where=count[..., None] > 0
            )
        counts.append(count)
        family_values.append(averaged)
    result = {name: np.stack([f[name] for f in family_values]).mean(0) for name in family_values[0]}
    result["n_worlds_by_family"] = np.stack(counts)
    norm = np.linalg.norm(result["gain"], axis=-1)
    result["norm"] = norm
    result["unit"] = np.divide(
        result["gain"], norm[..., None], out=np.full_like(result["gain"], np.nan), where=norm[..., None] > 0
    )
    result["threshold"] = np.sum(result["center"] * result["unit"], axis=-1)
    return result


def auc(scores, labels):
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    difference = positive[:, None] - negative[None, :]
    return float(np.mean(difference > 0) + 0.5 * np.mean(difference == 0))


def direction_geometry(left, right):
    """Compare complete vectors; zero or missing directions have no defined angle."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim < 1:
        raise ValueError("Expected equally shaped vectors with a final feature axis")
    left_norm = np.linalg.norm(left, axis=-1)
    right_norm = np.linalg.norm(right, axis=-1)
    complete = np.isfinite(left).all(-1) & np.isfinite(right).all(-1)
    comparable = complete & (left_norm > 0) & (right_norm > 0)
    cosine = np.divide(
        np.sum(left * right, axis=-1),
        left_norm * right_norm,
        out=np.full_like(left_norm, np.nan),
        where=comparable,
    )
    cosine = np.clip(cosine, -1, 1)
    return dict(
        left_norm=left_norm,
        right_norm=right_norm,
        complete=complete,
        comparable=comparable,
        cosine=cosine,
        angle_degrees=np.degrees(np.arccos(cosine)),
    )


def projection_metrics(world_scores, threshold, seed=2026101409, bootstrap=2000):
    """One independent resampling unit is one complete four-cell world."""
    scores = np.asarray(world_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError("Expected worlds by four factorial cells")
    scores = scores[np.isfinite(scores).all(1)]
    if not len(scores):
        return dict(
            worlds=0, auc=None, accuracy=None, mean_factorial_gap=None, gap_ci95=None, world_concordance=None
        )
    gaps = scores @ GAIN_WEIGHTS
    rng = np.random.default_rng(seed)
    means = gaps[rng.integers(len(gaps), size=(bootstrap, len(gaps)))].mean(1)
    labels = np.tile([0, 1, 1, 0], len(scores))
    return dict(
        worlds=len(scores),
        auc=auc(scores.ravel(), labels),
        accuracy=float(np.mean((scores.ravel() > threshold) == labels)),
        mean_factorial_gap=float(gaps.mean()),
        gap_ci95=np.quantile(means, [0.025, 0.975]).tolist(),
        world_concordance=float(np.mean(gaps > 0) + 0.5 * np.mean(gaps == 0)),
    )
