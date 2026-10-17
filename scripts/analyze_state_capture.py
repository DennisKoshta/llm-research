"""Fit independent passive-state contrasts and report development generalization."""

import argparse
import hashlib
from pathlib import Path
import numpy as np
from curiosity_study.artifacts import RunStore, read, write_once, digest


def auc(scores, labels):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        raise ValueError("Both classes required")
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def load_capture(root):
    m = read(root / "manifest.json")
    rs = RunStore(root, m).completed()
    aa = m["assignments"] + m["diagnostics"]
    assert set(rs) == {a["trial_id"] for a in aa}
    assert len(list((root / "attempts").glob("*/*.json"))) + sum(
        len(read(p)["records"]) for p in (root / "cohorts").glob("*/*.json")
    ) == len(aa)
    loaded = {}
    vectors = []
    for a in aa:
        r = rs[a["trial_id"]]
        assert r["assignment"] == a and r["measurement"]["hooks_remaining"] == 0
        assert r["task"]["uncertainty_bits"] == 1
        assert r["task"]["information_bits"] == int(a["query"] not in a["known"])
        meta = r["activation_artifact"]
        relative = meta["path"]
        if relative not in loaded:
            file = root / relative
            assert hashlib.sha256(file.read_bytes()).hexdigest() == meta["sha256"]
            with np.load(file, allow_pickle=False) as z:
                loaded[relative] = {k: z[k] for k in z.files}
        z = loaded[relative]
        assert list(z["layers"]) == meta["layers"] and list(z["positions"]) == meta["positions"]
        assert z["trial_ids"][meta["row"]] == a["trial_id"]
        vectors.append(z["hidden"][meta["row"]])
    x = np.stack(vectors).astype(np.float64)
    assert np.isfinite(x).all()
    return m, rs, aa, x, meta["layers"], meta["positions"]


def analyze(root, out):
    m, rs, aa, x, layers, positions = load_capture(root)
    train = np.array([a["split"] == "train" for a in aa])
    development = np.array([a["split"] == "development" for a in aa])
    y = np.array([int(a["query"] not in a["known"]) for a in aa])
    families = np.array([a["family"] for a in aa])
    known = np.array([a["known"][0] for a in aa])
    query = np.array([a["query"] for a in aa])
    sources = ["pooled", "objects", "machines"]
    units = []
    centers = []
    norms = []
    control_vectors = []
    metrics = []
    for source in sources:
        fit = train & ((families == source) if source != "pooled" else True)
        positive = x[fit & (y == 1)].mean(0)
        negative = x[fit & (y == 0)].mean(0)
        contrast = positive - negative
        norm = np.linalg.norm(contrast, axis=-1)
        assert np.all(norm > 0)
        unit = contrast / norm[..., None]
        center = np.sum((positive + negative) * unit / 2, axis=-1)
        units.append(unit)
        norms.append(norm)
        centers.append(center)
        controls = []
        for labels in (known, query):
            d = x[fit & (labels == 1)].mean(0) - x[fit & (labels == 0)].mean(0)
            controls.append(d / np.linalg.norm(d, axis=-1)[..., None])
        control_vectors.append(np.stack(controls))
        scores = np.einsum("nlpd,lpd->nlp", x, unit)
        for family in ("objects", "machines"):
            test = development & (families == family)
            for li, layer in enumerate(layers):
                for pi, position in enumerate(positions):
                    s = scores[test, li, pi]
                    labels = y[test]
                    pairs = {}
                    for idx in np.where(test)[0]:
                        a = aa[idx]
                        pairs.setdefault((a["index"], a["query"]), {})[int(y[idx])] = scores[idx, li, pi]
                    gaps = np.array([v[1] - v[0] for v in pairs.values()])
                    metrics.append(
                        dict(
                            source=source,
                            target=family,
                            layer=layer,
                            position=position,
                            n=int(test.sum()),
                            auc=auc(s, labels),
                            accuracy=float(np.mean((s > center[li, pi]) == labels)),
                            paired_concordance=float(np.mean(gaps > 0) + 0.5 * np.mean(gaps == 0)),
                            mean_paired_gap=float(gaps.mean()),
                            training_contrast_norm=float(norm[li, pi]),
                        )
                    )
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "directions.npz"
    with destination.open("xb") as f:
        np.savez_compressed(
            f,
            unit=np.array(units),
            center=np.array(centers),
            contrast_norm=np.array(norms),
            controls=np.array(control_vectors),
            sources=np.array(sources),
            layers=np.array(layers),
            positions=np.array(positions),
        )
    context_ids = {split: set() for split in ("train", "development")}
    world_ids = {split: set() for split in ("train", "development")}
    for a in aa:
        r = rs[a["trial_id"]]
        t = r["task"]
        split = a["split"]
        context_ids[split].add(r["measurement"]["prompt_sha256"])
        world_ids[split].add(
            digest(
                dict(
                    family=a["family"],
                    attributes=t["attributes"],
                    values=[sorted(v) for v in t["values"]],
                    truth=t["candidates"][t["truth"]],
                )
            )
        )
    summary = dict(
        manifest_sha256=digest(m),
        complete=len(rs),
        layers=layers,
        positions=positions,
        direction_file_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        metrics=metrics,
        duplicate_audit=dict(
            exact_prompt_overlap=len(context_ids["train"] & context_ids["development"]),
            semantic_world_overlap=len(world_ids["train"] & world_ids["development"]),
            unique_semantic_worlds={k: len(v) for k, v in world_ids.items()},
            interpretation="Development reuses the development vocabulary; held-out evaluation uses disjoint vocabularies and numeric ranges.",
        ),
        selection="No downstream behavior used; all layers, positions, and cross-family results retained.",
    )
    write_once(out / "analysis.json", summary)
    lines = [
        "# Passive information-state representation development",
        "",
        "Training and development data only. The held-out corpus has not been evaluated. No causal or experience claim follows from decoding.",
        "",
        "| Source | Target | Layer | Position | AUC | Accuracy | Paired concordance |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for c in metrics:
        lines.append(
            f"| {c['source']} | {c['target']} | {c['layer']} | {c['position']} | {c['auc']:.3f} | {c['accuracy']:.3f} | {c['paired_concordance']:.3f} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print({k: v for k, v in summary.items() if k != "metrics"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    analyze(args.run, args.out)
