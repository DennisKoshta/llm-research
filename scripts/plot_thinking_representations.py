"""Export all-site descriptive representation figures without refitting any axis."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from curiosity_study.artifacts import digest, read, write_once  # noqa: E402
from plot_thinking_validation import csv_file, file_sha, require, save, style  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402

FAMILIES = ("objects", "machines")
KINDS = ("new", "repeat", "memory", "count", "instrument")
LAYERS = (15, 31, 47, 57)
POSITIONS = ("record", "prefill", "token_32", "token_128", "token_512", "think_end", "decision")
POSITION_LABELS = (
    "Record",
    "Prompt\nend",
    "Token\n32",
    "Token\n128",
    "Token\n512",
    "Think\nend",
    "Answer\nboundary",
)
SOURCE = "pooled:balanced"


def flatten(row):
    return {
        key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
        for key, value in row.items()
    }


def derive_rows(analysis, manifest, audit, directions, scores):
    """Recompute descriptive metrics from saved scores, with the saved thresholds."""
    require(analysis["development_only"] is True, "Expected a development-only analysis")
    require(
        analysis["manifest_sha256"] == audit["manifest_sha256"] == digest(manifest),
        "Analysis, audit, and manifest binding differ",
    )
    require(
        audit["complete"] is True and analysis["assigned"] == audit["assigned"] == 4000,
        "Require the complete validation",
    )
    require(
        tuple(analysis["layers"]) == LAYERS and tuple(analysis["positions"]) == POSITIONS,
        "Unexpected planned sites",
    )
    require(analysis["model"] == manifest["model"], "Model binding differs")
    assignments = manifest["diagnostics"]
    require(digest(assignments) == manifest["assignment_sha256"], "Assignment binding differs")
    require(
        scores["trial_ids"].tolist() == [a["trial_id"] for a in assignments],
        "Saved score row identities differ",
    )
    sources = directions["sources"].tolist()
    expected_sources = {
        f"{family}:{kind}"
        for family in ("pooled", *FAMILIES)
        for kind in ("new", "repeat", "balanced", "answer")
    }
    require(set(sources) == expected_sources and len(sources) == 12, "Missing or extra fitted sources")
    for artifact in (directions, scores):
        require(artifact["sources"].tolist() == sources, "Source order differs between artifacts")
        require(
            artifact["layers"].tolist() == list(LAYERS) and artifact["positions"].tolist() == list(POSITIONS),
            "Artifact site order differs",
        )
    require(scores["scores"].shape == (4000, 12, 4, 7), "Unexpected score array shape")
    groups = defaultdict(dict)
    for i, a in enumerate(assignments):
        require(
            a["representation_split"] == ("extraction" if a["index"] < 50 else "development"),
            "Unexpected source split",
        )
        key = a["family"], a["kind"], a["index"]
        cell = a["known"][0], a["query"]
        require(cell not in groups[key], "Duplicate factorial assignment")
        groups[key][cell] = i
    cells = ((0, 0), (0, 1), (1, 0), (1, 1))
    require(all(set(g) == set(cells) for g in groups.values()), "Incomplete factorial assignments")
    target_indices = {
        (family, kind): np.array(
            [[groups[(family, kind, index)][cell] for cell in cells] for index in range(50, 100)]
        )
        for family in FAMILIES
        for kind in KINDS
    }
    availability = {
        (r["family"], r["kind"], r["split"], r["layer"], r["position"]): r
        for r in analysis["complete_world_availability"]
    }
    require(
        len(availability) == len(analysis["complete_world_availability"]) == 560,
        "Complete-world availability coverage differs",
    )
    source_counts = []
    for source in sources:
        family = source.split(":")[0]
        families = FAMILIES if family == "pooled" else (family,)
        counts = np.array(analysis["source_world_counts"][source])
        require(
            counts.shape == (len(families), 4, 7) and np.isin(counts, np.arange(51)).all(),
            "Invalid extraction world counts",
        )
        for fi, source_family in enumerate(families):
            for li, layer in enumerate(LAYERS):
                for pi, position in enumerate(POSITIONS):
                    source_counts.append(
                        dict(
                            source=source,
                            family=source_family,
                            layer=layer,
                            position=position,
                            assigned_extraction_worlds=50,
                            complete_extraction_worlds=int(counts[fi, li, pi]),
                        )
                    )
    rows, seen = [], set()
    for r in analysis["metrics"]:
        source, family, kind, layer, position = (
            r[k] for k in ("source", "target_family", "target_kind", "layer", "position")
        )
        key = source, family, kind, layer, position
        require(
            key not in seen and source in sources and (family, kind) in target_indices,
            "Duplicate or unexpected metric row",
        )
        seen.add(key)
        si, li, pi = sources.index(source), LAYERS.index(layer), POSITIONS.index(position)
        unit = directions["unit"][si, li, pi]
        axis_available = bool(np.isfinite(unit).all())
        threshold = float(directions["threshold"][si, li, pi])
        world_scores = scores["scores"][target_indices[(family, kind)], si, li, pi].astype(float)
        complete = np.isfinite(world_scores).all(1)
        observed = world_scores[complete]
        worlds = len(observed)
        target = availability[(family, kind, "development", layer, position)]
        require(
            target["assigned_worlds"] == r["assigned_worlds"] == 50 and r["worlds"] == worlds,
            "Complete scored-world counts differ",
        )
        require(
            worlds == (target["complete_worlds"] if axis_available else 0),
            "Target availability and axis availability do not explain scored worlds",
        )
        if worlds:
            require(np.isfinite(threshold), "Available projections require a finite frozen threshold")
            positive = observed[:, [1, 2]].ravel()
            negative = observed[:, [0, 3]].ravel()
            comparisons = positive[:, None] - negative[None, :]
            auc = float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))
            accuracy = float(np.mean((observed > threshold) == np.array([0, 1, 1, 0])))
            gaps = observed @ np.array([-0.5, 0.5, 0.5, -0.5])
            for field, expected in (
                ("auc", auc),
                ("accuracy", accuracy),
                ("mean_factorial_gap", float(gaps.mean())),
                ("world_concordance", float(np.mean(gaps > 0) + 0.5 * np.mean(gaps == 0))),
            ):
                require(
                    np.isclose(r[field], expected, rtol=1e-7, atol=1e-9),
                    f"Saved metric does not reproduce: {key}, {field}",
                )
            require(
                len(r["gap_ci95"]) == 2
                and np.isfinite(r["gap_ci95"]).all()
                and r["gap_ci95"][0] <= r["gap_ci95"][1],
                "Malformed source bootstrap interval",
            )
        else:
            require(
                all(
                    r[field] is None
                    for field in ("auc", "accuracy", "mean_factorial_gap", "world_concordance", "gap_ci95")
                ),
                "Unavailable metric must remain missing",
            )
        row = {k: v for k, v in r.items() if k != "gap_ci95"}
        row.update(
            gap_ci95_low=None if r["gap_ci95"] is None else r["gap_ci95"][0],
            gap_ci95_high=None if r["gap_ci95"] is None else r["gap_ci95"][1],
            source_axis_available=axis_available,
            fixed_extraction_threshold=threshold if np.isfinite(threshold) else None,
            target_complete_worlds=target["complete_worlds"],
            scored_worlds_missing=50 - worlds,
            metric_status="available"
            if worlds
            else "source_axis_unavailable"
            if not axis_available
            else "no_complete_target_worlds",
        )
        rows.append(row)
    require(
        len(rows) == 3360
        and seen
        == {
            (s, f, k, layer, p)
            for s in sources
            for f in FAMILIES
            for k in KINDS
            for layer in LAYERS
            for p in POSITIONS
        },
        "All 3,360 planned metric rows must be retained",
    )
    require(
        len(analysis["direction_geometry"]) == 364, "All planned direction-geometry rows must be retained"
    )
    main = [r for r in rows if r["source"] == SOURCE and r["target_kind"] in ("new", "repeat")]
    require(len(main) == 112, "Main figure must retain every pooled balanced target/site")
    return rows, main, source_counts


def label_color(cmap, value):
    rgb = np.array(cmap(value)[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
    return "white" if 1.05 / (luminance + 0.05) >= 4.5 else "black"


def plot_metric(rows, analysis, audit, out, metric):
    lookup = {(r["target_family"], r["target_kind"], r["layer"], r["position"]): r for r in rows}
    cmap = LinearSegmentedColormap.from_list("development_blue", ["#F8FAFC", "#315F91"])
    cmap.set_bad("#E5E5E5")
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 11.1), sharex=True, sharey=True)
    for ri, family in enumerate(FAMILIES):
        for ci, kind in enumerate(("new", "repeat")):
            ax = axes[ri, ci]
            values = np.array(
                [
                    [
                        np.nan
                        if lookup[(family, kind, layer, pos)][metric] is None
                        else lookup[(family, kind, layer, pos)][metric]
                        for pos in POSITIONS
                    ]
                    for layer in LAYERS
                ]
            )
            im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for y, layer in enumerate(LAYERS):
                for x, position in enumerate(POSITIONS):
                    row = lookup[(family, kind, layer, position)]
                    if row[metric] is None:
                        ax.add_patch(
                            Rectangle(
                                (x - 0.5, y - 0.5),
                                1,
                                1,
                                facecolor="none",
                                edgecolor="#BBBBBB",
                                hatch="//",
                                linewidth=0,
                            )
                        )
                        text = f"NA\n{'no axis' if not row['source_axis_available'] else 'no worlds'}\nn={row['worlds']}/50"
                        color = "black"
                    else:
                        text = f"{row[metric]:.3f}\nn={row['worlds']}/50"
                        color = label_color(cmap, row[metric])
                    ax.text(x, y, text, ha="center", va="center", fontsize=9, color=color)
            ax.set_title(
                f"{family.title()} · {'New-information question' if kind == 'new' else 'Already-known question'}",
                loc="left",
                fontweight="bold",
                pad=13,
            )
            ax.set_xticks(range(7), POSITION_LABELS)
            ax.set_yticks(range(4), [str(layer) for layer in LAYERS])
            ax.tick_params(axis="both", length=0, pad=8, labelbottom=True, labelleft=True)
            if ci == 0:
                ax.set_ylabel("Layer index")
            ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.3)
            ax.tick_params(which="minor", length=0)
            ax.spines[:].set_visible(False)
    title = (
        "Information-gain AUC in development worlds"
        if metric == "auc"
        else "Fixed-threshold accuracy in development worlds"
    )
    fig.suptitle(title, x=0.04, y=0.985, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.04,
        0.947,
        "Qwen3-32B · pooled polarity-balanced axis · extraction indices 0–49; development indices 50–99",
        fontsize=11,
        color="#444444",
    )
    fig.text(
        0.04,
        0.920,
        "All four layers and seven positions retained; axis sign and classification threshold are unchanged from extraction.",
        fontsize=10,
        color="#444444",
    )
    cax = fig.add_axes([0.29, 0.227, 0.42, 0.017])
    colorbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.25, 0.5, 0.75, 1])
    colorbar.ax.set_xticklabels(["0.00", "0.25", "0.50\nchance", "0.75", "1.00"])
    colorbar.ax.axvline(0.5, color="#222222", linewidth=1.2)
    colorbar.outline.set_visible(False)
    overlap = analysis["duplicate_audit"]["semantic_worlds"]
    status = "met" if audit["gates"]["passed"] else "not met (count validity)"
    footnotes = [
        f"Development only. Original whole-instrument gate: {status}. Semantic-world overlap: objects {overlap['objects']['overlap']}; machines {overlap['machines']['overlap']}.",
        f"There are {analysis['duplicate_audit']['exact_prompt_overlap']} exact-prompt overlaps between extraction and development. This is not a fresh confirmation sample.",
        "Record precedes the question: its context is identical across task kinds, so cross-kind decoding is not independent replication.",
        "n = complete scored four-cell worlds / 50 assigned. No axis = no complete paired extraction worlds. Available world sets can differ by position.",
        "These are descriptive projections. They establish no causal mechanism, motivational state, or subjective experience.",
    ]
    for y, note in zip((0.140, 0.111, 0.082, 0.053, 0.024), footnotes):
        fig.text(0.04, y, note, fontsize=9.5, color="#444444")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.855, bottom=0.32, hspace=0.48, wspace=0.13)
    save(fig, out, f"pooled-balanced-{metric}")


def export(analysis_dir, run, audit_path, out):
    analysis_path = analysis_dir / "analysis.json"
    analysis, manifest, audit = read(analysis_path), read(run / "manifest.json"), read(audit_path)
    direction_path, score_path = analysis_dir / "directions.npz", analysis_dir / "scores.npz"
    require(
        file_sha(direction_path) == analysis["directions_sha256"], "Fitted direction artifact hash differs"
    )
    with np.load(direction_path, allow_pickle=False) as z:
        directions = {key: z[key] for key in ("sources", "layers", "positions", "unit", "threshold")}
    with np.load(score_path, allow_pickle=False) as z:
        scores = {key: z[key] for key in z.files}
    rows, main, source_counts = derive_rows(analysis, manifest, audit, directions, scores)
    frozen = read(run / "freeze.json")
    require(frozen["manifest_sha256"] == digest(manifest), "Frozen source collection differs")
    require(not out.exists(), "Choose a new figure directory; existing exports are not overwritten")
    out.mkdir(parents=True)
    style()
    plot_metric(main, analysis, audit, out, "auc")
    plot_metric(main, analysis, audit, out, "accuracy")
    for name, values in (
        ("all-development-metrics.csv", rows),
        ("plotted-pooled-balanced-metrics.csv", main),
        ("direction-geometry.csv", [flatten(r) for r in analysis["direction_geometry"]]),
        ("complete-world-availability.csv", analysis["complete_world_availability"]),
        ("branch-availability.csv", analysis["availability"]),
        ("source-extraction-world-counts.csv", source_counts),
    ):
        csv_file(out / name, values)
    source_files = {
        "analysis": analysis_path,
        "directions": direction_path,
        "scores": score_path,
        "source_manifest": run / "manifest.json",
        "validation_audit": audit_path,
        "plot_script": Path(__file__),
        "shared_plot_style": Path(__file__).with_name("plot_thinking_validation.py"),
    }
    write_once(
        out / "figure-provenance.json",
        dict(
            schema_version=1,
            development_only=True,
            source_files={
                name: dict(path=str(path.resolve()), sha256=file_sha(path))
                for name, path in source_files.items()
            },
            manifest_sha256=digest(manifest),
            analysis_sha256=digest(analysis),
            audit_sha256=digest(audit),
            frozen_source_sha256=frozen["source_sha256"],
            frozen_archive_sha256=frozen["archive_sha256"],
            directions_sha256=analysis["directions_sha256"],
            model=analysis["model"],
            matplotlib_version=matplotlib.__version__,
            numpy_version=np.__version__,
            plotted_source=SOURCE,
            plotted_target_families=list(FAMILIES),
            plotted_target_kinds=["new", "repeat"],
            all_sites_retained=True,
            original_instrument_gate_passed=audit["gates"]["passed"],
            duplicate_audit=analysis["duplicate_audit"],
            row_counts=dict(
                all_metrics=len(rows),
                plotted_metrics=len(main),
                direction_geometry=len(analysis["direction_geometry"]),
                complete_world_availability=len(analysis["complete_world_availability"]),
                source_extraction_world_counts=len(source_counts),
            ),
            checks=[
                "all 3,360 AUC, accuracy, factorial-gap and concordance values recomputed from saved scores using the saved extraction thresholds",
                "score trial identities and all source/layer/position dimensions reconciled",
                "all 112 plotted rows retain their complete four-cell world denominator",
                "unavailable source axes remain missing rather than zero-valued",
                "no fitting, recentering, threshold changes or favorable-site selection performed",
            ],
            interval_policy="AUC and threshold accuracy are descriptive and have no plotted confidence intervals. CSV retains original pointwise whole-world bootstrap gap intervals without treating them as confirmatory or multiplicity-adjusted.",
            missingness_policy="Scored-world counts require an available source axis and all four target cells. CSV distinguishes raw target complete-world availability from a missing source axis.",
            caveats=[
                "Original whole-instrument validation did not pass its count-validity gate",
                "Extraction/development semantic and exact-prompt overlap is reported",
                "Prequestion record context is identical across task kinds; these are not independent replications",
                "Variable token positions can use different world sets",
                "No causal mechanism, motivational state, or subjective experience claim",
            ],
            output_sha256={p.name: file_sha(p) for p in sorted(out.iterdir()) if p.is_file()},
            visual_review="pending rendered inspection; recorded separately after export",
        ),
    )
    print(
        dict(
            figures=2,
            exported_metrics=len(rows),
            geometry_rows=len(analysis["direction_geometry"]),
            out=str(out.resolve()),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.analysis, args.run, args.audit, args.out)
