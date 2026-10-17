"""Analyze all frozen layers/positions without selecting on intervention outcomes."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import numpy as np
from curiosity_study.artifacts import read, write_once, digest, RunStore
from curiosity_study.representation import world_summaries, fit_source, projection_metrics, direction_geometry
from curiosity_study.thinking_information import task_module


def load_collection(root):
    manifest = read(root / "manifest.json")
    children = (
        read(root / "children.json")
        if (root / "children.json").exists()
        else [dict(path=str(root), manifest_sha256=digest(manifest))]
    )
    records, locations = {}, {}
    for child in children:
        path = Path(child["path"])
        m = read(path / "manifest.json")
        assert digest(m) == child["manifest_sha256"]
        if path != root:
            assert m["parent_manifest_sha256"] == digest(manifest)
        rs = RunStore(path, m).completed()
        assert set(rs) == {a["trial_id"] for a in m["diagnostics"]}
        assert not records.keys() & rs.keys()
        records.update(rs)
        locations.update({key: path for key in rs})
    aa = manifest["diagnostics"]
    assert set(records) == {a["trial_id"] for a in aa}
    first = records[aa[0]["trial_id"]]["activation_artifact"]
    layers, positions = first["layers"], first["positions"]
    by_file = defaultdict(list)
    for i, a in enumerate(aa):
        r = records[a["trial_id"]]
        assert r["assignment"] == a
        assert digest(r["task"]) == digest(task_module(a).task(a))
        meta = r["activation_artifact"]
        by_file[locations[a["trial_id"]] / meta["path"]].append((i, a, meta))
    x = None
    for file, selected in by_file.items():
        sha = hashlib.sha256(file.read_bytes()).hexdigest()
        with np.load(file, allow_pickle=False) as z:
            h = z["hidden"]
            if x is None:
                x = np.empty((len(aa), *h.shape[1:]), dtype=np.float32)
            assert list(z["layers"]) == layers and list(z["positions"]) == positions
            for i, a, meta in selected:
                assert sha == meta["sha256"]
                assert z["trial_ids"][meta["row"]] == a["trial_id"]
                assert np.array_equal(np.isfinite(h[meta["row"]]).all(-1), z["available"][meta["row"]])
                x[i] = h[meta["row"]]
    return manifest, aa, records, locations, x, layers, positions


def analyze(root, out):
    manifest, aa, records, locations, x, layers, positions = load_collection(root)
    assert all(a["representation_split"] == ("extraction" if a["index"] < 50 else "development") for a in aa)
    out.mkdir(parents=True, exist_ok=True)
    groups, summaries = world_summaries(aa, x)
    sources, fitted = [], []
    for family in ("pooled", "objects", "machines"):
        families = ["objects", "machines"] if family == "pooled" else [family]
        for kind in ("new", "repeat", "balanced", "answer"):
            kinds = ["new", "repeat"] if kind in ("balanced", "answer") else [kind]
            sources.append(f"{family}:{kind}")
            fitted.append(fit_source(summaries, families, kinds, answer_contrast=kind == "answer"))
    complete_world_availability = []
    for family in ("objects", "machines"):
        for kind in ("new", "repeat", "memory", "count", "instrument"):
            for split, indices in (("extraction", range(50)), ("development", range(50, 100))):
                complete = np.stack(
                    [np.isfinite(summaries[(family, index, kind)]["gain"]).all(-1) for index in indices]
                ).sum(0)
                for li, layer in enumerate(layers):
                    for pi, position in enumerate(positions):
                        complete_world_availability.append(
                            dict(
                                family=family,
                                kind=kind,
                                split=split,
                                layer=layer,
                                position=position,
                                assigned_worlds=len(indices),
                                complete_worlds=int(complete[li, pi]),
                            )
                        )
    by_source = dict(zip(sources, fitted))
    comparisons = []
    for family in ("pooled", "objects", "machines"):
        balanced, answer = f"{family}:balanced", f"{family}:answer"
        comparisons.extend(
            [
                (balanced, "gain", answer, "gain"),
                (balanced, "gain", balanced, "known"),
                (balanced, "gain", balanced, "query"),
                (f"{family}:new", "gain", f"{family}:repeat", "gain"),
            ]
        )
    comparisons.append(("objects:balanced", "gain", "machines:balanced", "gain"))
    geometry = []
    for left, left_component, right, right_component in comparisons:
        g = direction_geometry(by_source[left][left_component], by_source[right][right_component])
        for li, layer in enumerate(layers):
            for pi, position in enumerate(positions):
                row = dict(
                    left_source=left,
                    left_component=left_component,
                    right_source=right,
                    right_component=right_component,
                    layer=layer,
                    position=position,
                    left_source_world_counts=by_source[left]["n_worlds_by_family"][:, li, pi].tolist(),
                    right_source_world_counts=by_source[right]["n_worlds_by_family"][:, li, pi].tolist(),
                    complete=bool(g["complete"][li, pi]),
                    comparable=bool(g["comparable"][li, pi]),
                )
                for key in ("left_norm", "right_norm", "cosine", "angle_degrees"):
                    value = float(g[key][li, pi])
                    row[key] = value if np.isfinite(value) else None
                geometry.append(row)
    units = np.stack([f["unit"] for f in fitted])
    score = np.empty((len(aa), len(sources), len(layers), len(positions)), dtype=np.float32)
    for li in range(len(layers)):
        for pi in range(len(positions)):
            score[:, :, li, pi] = x[:, li, pi].astype(np.float64) @ units[:, li, pi].T
    del x, summaries
    metrics = []
    for si, (name, fit) in enumerate(zip(sources, fitted)):
        for family in ("objects", "machines"):
            for kind in ("new", "repeat", "memory", "count", "instrument"):
                indices = [groups[(family, index, kind)] for index in range(50, 100)]
                for li, layer in enumerate(layers):
                    for pi, position in enumerate(positions):
                        ws = score[np.array(indices), si, li, pi]
                        metric = projection_metrics(ws, fit["threshold"][li, pi])
                        metrics.append(
                            dict(
                                source=name,
                                target_family=family,
                                target_kind=kind,
                                layer=layer,
                                position=position,
                                assigned_worlds=50,
                                **metric,
                            )
                        )
    path = out / "directions.npz"
    with path.open("xb") as f:
        np.savez_compressed(
            f,
            sources=np.array(sources),
            layers=np.array(layers),
            positions=np.array(positions),
            unit=units,
            contrast=np.stack([v["gain"] for v in fitted]),
            contrast_norm=np.stack([v["norm"] for v in fitted]),
            center=np.stack([v["center"] for v in fitted]),
            threshold=np.stack([v["threshold"] for v in fitted]),
            known_contrast=np.stack([v["known"] for v in fitted]),
            query_contrast=np.stack([v["query"] for v in fitted]),
        )
    with (out / "scores.npz").open("xb") as f:
        np.savez_compressed(
            f,
            scores=score,
            trial_ids=np.array([a["trial_id"] for a in aa]),
            sources=np.array(sources),
            layers=np.array(layers),
            positions=np.array(positions),
        )
    world_ids = defaultdict(set)
    prompt_ids = defaultdict(set)
    availability = defaultdict(lambda: dict(assigned=0, available=0))
    for a in aa:
        r = records[a["trial_id"]]
        task = r["task"]
        split = a["representation_split"]
        # Include numeric input identities, but ignore candidate row ordering.
        wid = digest(
            dict(
                family=a["family"],
                attributes=task["attributes"],
                values=[sorted(v) for v in task["values"]],
                truth=task["candidates"][task["truth"]],
            )
        )
        world_ids[(a["family"], split)].add(wid)
        prompt_ids[split].add(r["inference"]["prompt_sha256"])
        for pi, position in enumerate(positions):
            c = availability[(a["family"], a["kind"], int(task["information_bits"]), split, position)]
            c["assigned"] += 1
            c["available"] += all(row[pi] for row in r["activation_artifact"]["available"])
    audit = dict(
        exact_prompt_overlap=len(prompt_ids["extraction"] & prompt_ids["development"]),
        semantic_worlds={
            family: dict(
                extraction=len(world_ids[(family, "extraction")]),
                development=len(world_ids[(family, "development")]),
                overlap=len(world_ids[(family, "extraction")] & world_ids[(family, "development")]),
            )
            for family in ("objects", "machines")
        },
    )
    result = dict(
        manifest_sha256=digest(manifest),
        model=manifest["model"],
        assigned=len(aa),
        layers=layers,
        positions=positions,
        directions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        development_only=True,
        source_world_counts={name: f["n_worlds_by_family"].tolist() for name, f in zip(sources, fitted)},
        direction_geometry=geometry,
        complete_world_availability=complete_world_availability,
        duplicate_audit=audit,
        metrics=metrics,
        availability=[
            dict(family=k[0], kind=k[1], gain=k[2], split=k[3], position=k[4], **v)
            for k, v in sorted(availability.items())
        ],
    )
    write_once(out / "analysis.json", result)
    lines = [
        "# Information representation development results",
        "",
        "Extraction and development only. All four layers and seven positions are retained. Missing cells exclude the entire world at that position; no correctness filtering is applied. Bootstrap intervals resample whole worlds. AUC and accuracy describe development performance and are not confirmatory significance tests.",
        "",
        "The JSON artifact reports complete four-cell world availability and direction geometry at every layer/position. Balanced polarity contrasts need not be orthogonal to answer or attribute contrasts; their cosines and norms are diagnostics, not proof that the information component is independent of those alternatives. Missing or zero-length directions have no defined cosine or angle.",
        "",
        "## Information and answer direction geometry",
        "",
        "| Source families | Layer | Position | Information norm | Answer norm | Cosine | Angle (degrees) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for c in geometry:
        if c["left_source"] != "pooled:balanced" or c["right_source"] != "pooled:answer":
            continue
        lines.append(
            f"| Pooled | {c['layer']} | {c['position']} | {c['left_norm']} | {c['right_norm']} | {c['cosine']} | {c['angle_degrees']} |"
        )
    lines.extend(
        [
            "",
            "## Balanced information direction across question polarities",
            "",
            "| Target | Question | Layer | Position | Worlds | AUC | Mean paired gap | 95% interval |",
            "|---|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for c in metrics:
        if c["source"] != "pooled:balanced" or c["position"] not in (
            "record",
            "prefill",
            "think_end",
            "decision",
        ):
            continue
        ci = c["gap_ci95"]
        lines.append(
            f"| {c['target_family']} | {c['target_kind']} | {c['layer']} | {c['position']} | {c['worlds']} | {c['auc']} | {c['mean_factorial_gap']} | {ci} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(
        dict(
            assigned=len(aa),
            metrics=len(metrics),
            duplicate_audit=audit,
            directions_sha256=result["directions_sha256"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    analyze(args.run, args.out)
