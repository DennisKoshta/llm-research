"""Export all-site relational transfer figures from frozen axes and saved scores.

No model execution, fitting, threshold calibration, or favorable-site selection.
The denominators are raw assigned index-worlds, not deduplicated semantic worlds.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from curiosity_study.artifacts import digest, read, write_once  # noqa: E402
from plot_thinking_validation import csv_file, file_sha, require, save, style  # noqa: E402
from plot_thinking_representations import (  # noqa: E402
    FAMILIES,
    KINDS,
    LAYERS,
    POSITIONS,
    POSITION_LABELS,
    SOURCE,
    label_color,
)

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402

PLOTTED_KINDS = ("new", "repeat", "instrument")
KIND_LABELS = {
    "new": "New-information question",
    "repeat": "Already-known question",
    "instrument": "Goal-directed choice",
}
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
QUANTITY = "information_bits"


def derive_rows(analysis, manifest, audit, directions, scores):
    """Independently reproduce every saved metric, including its existing gap CI."""
    require(
        analysis["descriptive_only"] is True and analysis["refitted_on_transfer"] is False,
        "Only descriptive transfer without refitting is supported",
    )
    require(
        analysis["task_revision"] == manifest["task_revision"] == "relational-information-v1",
        "This exporter must not mix target relevance and full-world information gain",
    )
    require(
        analysis["phase"] == manifest["phase"] == audit["phase"] == "development_validation"
        and analysis["inference_scope"] == manifest["inference_scope"] == "development",
        "The collection must remain explicitly development-only",
    )
    require(
        analysis["manifest_sha256"] == audit["manifest_sha256"] == digest(manifest),
        "Analysis, audit, and source manifest do not match",
    )
    require(
        analysis["source_audit_sha256"] == digest(audit)
        and audit["complete"] is True
        and audit["gates"]["passed"] is True,
        "Require the bound passing global development audit",
    )
    require(analysis["assigned"] == audit["assigned"] == 800, "Require all 800 source assignments")
    require(analysis["model"] == manifest["model"], "Transfer model binding differs")
    require(
        analysis["transfer_estimand"]
        == dict(
            information_quantity=QUANTITY,
            construct="expected_full_world_information_gain",
            cell_order=[list(c) for c in CELLS],
            gain_labels_bits=[0, 1, 1, 0],
        ),
        "Unexpected transfer information estimand",
    )
    require(
        analysis["validation_criteria"] == manifest["validation_criteria"]
        and manifest["validation_criteria"]["minimum_distinct_items_per_cell"] == 20,
        "Expected the prospectively declared 20-index-world development gate",
    )
    assignments = manifest["diagnostics"]
    require(
        len(assignments) == 800 and digest(assignments) == manifest["assignment_sha256"],
        "Source assignment binding differs",
    )
    require(
        scores["trial_ids"].tolist() == [a["trial_id"] for a in assignments],
        "Saved score identities differ from the frozen assignment order",
    )
    require(
        str(scores["information_quantity"]) == QUANTITY
        and scores["contrast_gain_labels_bits"].tolist() == [0, 1, 1, 0],
        "Saved scores have a different outcome quantity",
    )
    sources = directions["sources"].tolist()
    require(
        set(sources)
        == {
            f"{family}:{kind}"
            for family in ("pooled", *FAMILIES)
            for kind in ("new", "repeat", "balanced", "answer")
        }
        and len(sources) == 12,
        "Require all 12 original fitted sources",
    )
    for artifact in (directions, scores):
        require(
            artifact["sources"].tolist() == sources
            and artifact["layers"].tolist() == list(LAYERS)
            and artifact["positions"].tolist() == list(POSITIONS),
            "Score/direction dimensions differ",
        )
    require(scores["scores"].shape == (800, 12, 4, 7), "Unexpected score shape")
    groups = defaultdict(dict)
    for row, a in enumerate(assignments):
        require(
            a["task_revision"] == "relational-information-v1" and a["template"] == "table",
            "Unexpected source task or rendering",
        )
        key = a["family"], a["seed"], a["index"], a["kind"], a["template"]
        cell = a["correlated"], a["query"]
        require(cell in CELLS and cell not in groups[key], "Invalid or duplicate factorial cell")
        groups[key][cell] = row
    require(
        len(groups) == 200 and all(set(rows) == set(CELLS) for rows in groups.values()),
        "Require every assigned four-cell index-world",
    )
    targets = defaultdict(list)
    for (family, seed, index, kind, template), rows in groups.items():
        targets[(family, kind, template)].append([rows[cell] for cell in CELLS])
    require(
        set(targets) == {(family, kind, "table") for family in FAMILIES for kind in KINDS}
        and all(len(rows) == 20 for rows in targets.values()),
        "Expected 20 index-worlds per family/kind",
    )
    availability, available_axes = {}, np.isfinite(directions["unit"]).all(-1)
    for (family, kind, template), indices in targets.items():
        for li, layer in enumerate(LAYERS):
            for pi, position in enumerate(POSITIONS):
                axes = np.flatnonzero(available_axes[:, li, pi])
                finite = None
                for si in axes:
                    candidate = np.isfinite(scores["scores"][np.array(indices), si, li, pi])
                    require(
                        finite is None or np.array_equal(candidate, finite),
                        "Finite source axes disagree about target score availability",
                    )
                    finite = candidate
                availability[(family, kind, template, layer, position)] = dict(
                    target_family=family,
                    target_kind=kind,
                    template=template,
                    layer=layer,
                    position=position,
                    assigned_index_worlds=20,
                    assigned_branches=80,
                    target_complete_index_worlds=None if finite is None else int(finite.all(1).sum()),
                    target_available_branches=None if finite is None else int(finite.sum()),
                    target_availability_identifiable_from_saved_scores=finite is not None,
                )
    rows, seen, max_errors = [], set(), defaultdict(float)
    for recorded in analysis["metrics"]:
        source, family, kind, template, layer, position = (
            recorded[k] for k in ("source", "target_family", "target_kind", "template", "layer", "position")
        )
        key = source, family, kind, template, layer, position
        require(
            key not in seen and source in sources and (family, kind, template) in targets,
            "Duplicate or unexpected metric identity",
        )
        seen.add(key)
        si, li, pi = sources.index(source), LAYERS.index(layer), POSITIONS.index(position)
        available = bool(available_axes[si, li, pi])
        threshold = float(directions["threshold"][si, li, pi])
        original = scores["scores"][np.array(targets[(family, kind, template)]), si, li, pi]
        complete = np.isfinite(original).all(1)
        values = original[complete]
        n = len(values)
        require(
            recorded["assigned_worlds"] == 20
            and recorded["worlds"] == n
            and recorded["source_axis_available"] == available
            and recorded["information_quantity"] == QUANTITY,
            "Recorded counts, source availability, or outcome quantity differ",
        )
        if not available:
            require(np.isnan(original).all() and n == 0, "Missing source axes must retain missing scores")
        expected = dict(
            auc=None, accuracy=None, mean_factorial_gap=None, world_concordance=None, gap_ci95=None
        )
        if n:
            require(
                available and np.isfinite(threshold), "Scored worlds require a finite frozen axis/threshold"
            )
            positive, negative = values[:, [1, 2]].ravel(), values[:, [0, 3]].ravel()
            difference = positive[:, None] - negative[None, :]
            gaps = (values[:, 1] + values[:, 2] - values[:, 0] - values[:, 3]) / 2
            # Reproduce the saved descriptive interval with its original ordering,
            # seed and 2,000 draws. It is not a semantic-deduplicated confidence claim.
            indices = np.random.default_rng(2026101409).integers(n, size=(2000, n))
            expected.update(
                auc=float(np.mean(difference > 0) + 0.5 * np.mean(difference == 0)),
                accuracy=float(np.mean((values > threshold) == np.array([0, 1, 1, 0]))),
                mean_factorial_gap=float(gaps.mean()),
                world_concordance=float(np.mean(gaps > 0) + 0.5 * np.mean(gaps == 0)),
                gap_ci95=np.quantile(gaps[indices].mean(1), [0.025, 0.975]).tolist(),
            )
        for field, value in expected.items():
            if value is None:
                require(recorded[field] is None, f"Unavailable metric is not missing: {key}, {field}")
            else:
                require(
                    np.allclose(recorded[field], value, rtol=1e-9, atol=1e-11),
                    f"Saved metric does not reproduce: {key}, {field}",
                )
                max_errors[field] = max(
                    max_errors[field], float(np.max(np.abs(np.asarray(recorded[field]) - value)))
                )
        target = availability[(family, kind, template, layer, position)]
        row = {k: v for k, v in recorded.items() if k != "gap_ci95"}
        row.update(
            gap_ci95_low=None if expected["gap_ci95"] is None else expected["gap_ci95"][0],
            gap_ci95_high=None if expected["gap_ci95"] is None else expected["gap_ci95"][1],
            denominator_basis="raw assigned seed/index worlds; not semantic-deduplicated",
            scored_index_worlds_missing=20 - n,
            finite_scored_branches=int(np.isfinite(original).sum()),
            target_complete_index_worlds=target["target_complete_index_worlds"],
            fixed_extraction_threshold=threshold if np.isfinite(threshold) else None,
            metric_status="available"
            if n
            else "source_axis_unavailable"
            if not available
            else "no_complete_target_worlds",
        )
        rows.append(row)
    require(
        len(rows) == 3360
        and seen
        == {
            (source, family, kind, "table", layer, position)
            for source in sources
            for family in FAMILIES
            for kind in KINDS
            for layer in LAYERS
            for position in POSITIONS
        },
        "All 3,360 metric rows must be retained",
    )
    main = [r for r in rows if r["source"] == SOURCE and r["target_kind"] in PLOTTED_KINDS]
    require(len(main) == 168, "All 168 plotted family/kind/sites must remain visible")
    return rows, main, list(availability.values()), dict(max_errors)


def plot_metric(rows, out, metric):
    lookup = {(r["target_family"], r["target_kind"], r["layer"], r["position"]): r for r in rows}
    cmap = LinearSegmentedColormap.from_list("transfer_blue", ["#F8FAFC", "#315F91"])
    cmap.set_bad("#E5E5E5")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11.1), sharex=True, sharey=True)
    for ri, family in enumerate(FAMILIES):
        for ci, kind in enumerate(PLOTTED_KINDS):
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
                    value = row[metric]
                    if value is None:
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
                        label = "NA\nno axis" if not row["source_axis_available"] else "NA\nno worlds"
                        color = "black"
                    else:
                        label = f"{value:.1%}" if metric == "accuracy" else f"{value:.3f}"
                        color = label_color(cmap, value)
                    ax.text(
                        x,
                        y,
                        f"{label}\nn={row['worlds']}/20",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=color,
                    )
            ax.set_title(f"{family.title()} · {KIND_LABELS[kind]}", loc="left", fontweight="bold", pad=13)
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
    title = "Information-gain AUC" if metric == "auc" else "Fixed-threshold accuracy"
    fig.suptitle(
        f"{title} · relational transfer", x=0.035, y=0.985, ha="left", fontsize=20, fontweight="bold"
    )
    fig.text(
        0.035,
        0.947,
        "Qwen3-32B · frozen pooled polarity-balanced axis · separate relational development collection",
        fontsize=12,
        color="#444444",
    )
    fig.text(
        0.035,
        0.918,
        "All four layers and seven positions; original axis sign and extraction threshold retained. Relational development gate passed (800 assigned trials).",
        fontsize=11,
        color="#444444",
    )
    cax = fig.add_axes([0.34, 0.217, 0.32, 0.018])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.25, 0.5, 0.75, 1])
    bar.ax.set_xticklabels(
        ["0%", "25%", "50%\nchance", "75%", "100%"]
        if metric == "accuracy"
        else ["0.00", "0.25", "0.50\nchance", "0.75", "1.00"]
    )
    bar.ax.axvline(0.5, color="#222222", linewidth=1.2)
    bar.outline.set_visible(False)
    notes = [
        "Development only. n = complete scored raw index-worlds (all four cells) / 20 assigned per family and kind; these are not semantic-deduplicated independent counts.",
        "Hatching marks unavailable metrics. No axis = no available frozen extraction axis; this does not imply zero target activations. World availability may vary by position.",
        "Record context is identical across question kinds; early cross-kind readouts are not independent replications. Rank transfer and fixed-threshold transfer measure different behavior.",
        "Think-end and answer-boundary states condition on generated reasoning and answer policy. No favorable-site selection, causal-mechanism claim, or subjective-experience claim.",
        "AUC and accuracy are descriptive point estimates without plotted intervals. The full CSV retains original descriptive gap intervals, not confirmatory confidence claims.",
    ]
    for y, note in zip((0.143, 0.114, 0.085, 0.056, 0.027), notes):
        fig.text(0.035, y, note, fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.855, bottom=0.305, hspace=0.48, wspace=0.16)
    save(fig, out, f"pooled-balanced-transfer-{metric}")


def export(analysis_dir, run, audit_path, direction_path, out):
    analysis_path, score_path = analysis_dir / "analysis.json", analysis_dir / "scores.npz"
    analysis, manifest, audit = read(analysis_path), read(run / "manifest.json"), read(audit_path)
    source_analysis_path = direction_path.parent / "analysis.json"
    source_analysis = read(source_analysis_path)
    require(
        file_sha(direction_path) == analysis["directions_sha256"] == source_analysis["directions_sha256"],
        "Original fitted direction artifact differs",
    )
    require(
        digest(source_analysis) == analysis["source_analysis_sha256"]
        and source_analysis["manifest_sha256"] == analysis["direction_extraction_manifest_sha256"]
        and source_analysis["model"] == analysis["model"],
        "Original extraction analysis binding differs",
    )
    with np.load(direction_path, allow_pickle=False) as z:
        directions = {key: z[key] for key in ("sources", "layers", "positions", "unit", "threshold")}
    with np.load(score_path, allow_pickle=False) as z:
        scores = {key: z[key] for key in z.files}
    rows, main, availability, errors = derive_rows(analysis, manifest, audit, directions, scores)
    frozen = read(run / "freeze.json")
    require(frozen["manifest_sha256"] == digest(manifest), "Transfer source freeze binding differs")
    require(not out.exists(), "Choose a new directory; figure exports are immutable")
    out.mkdir(parents=True)
    style()
    for metric in ("auc", "accuracy"):
        plot_metric(main, out, metric)
    for name, values in (
        ("all-transfer-metrics.csv", rows),
        ("plotted-pooled-balanced-metrics.csv", main),
        ("target-score-availability.csv", availability),
    ):
        csv_file(out / name, values)
    paths = dict(
        analysis=analysis_path,
        scores=score_path,
        directions=direction_path,
        extraction_analysis=source_analysis_path,
        source_manifest=run / "manifest.json",
        source_freeze=run / "freeze.json",
        validation_audit=audit_path,
        plot_script=Path(__file__),
        shared_plot_style=Path(__file__).with_name("plot_thinking_validation.py"),
        shared_plot_definitions=Path(__file__).with_name("plot_thinking_representations.py"),
    )
    write_once(
        out / "figure-provenance.json",
        dict(
            schema_version=1,
            development_only=True,
            refitted_on_transfer=False,
            all_sites_retained=True,
            source_files={
                key: dict(path=str(path.resolve()), sha256=file_sha(path)) for key, path in paths.items()
            },
            manifest_sha256=digest(manifest),
            analysis_sha256=digest(analysis),
            audit_sha256=digest(audit),
            frozen_source_sha256=frozen["source_sha256"],
            frozen_archive_sha256=frozen["archive_sha256"],
            directions_sha256=analysis["directions_sha256"],
            model=analysis["model"],
            source_phase=manifest["phase"],
            global_development_gate_passed=audit["gates"]["passed"],
            validation_criteria=manifest["validation_criteria"],
            transfer_estimand=analysis["transfer_estimand"],
            plotted_source=SOURCE,
            plotted_target_families=list(FAMILIES),
            plotted_target_kinds=list(PLOTTED_KINDS),
            layers=list(LAYERS),
            positions=list(POSITIONS),
            row_counts=dict(
                all_metrics=len(rows), plotted_metrics=len(main), target_availability=len(availability)
            ),
            independently_recomputed_metrics=[
                "auc",
                "accuracy",
                "mean_factorial_gap",
                "world_concordance",
                "gap_ci95",
            ],
            maximum_absolute_recomputation_error=errors,
            resampling_reproduction=dict(
                seed=2026101409, draws=2000, unit="raw complete four-cell index-world"
            ),
            denominator_policy="20 raw assigned index-worlds per family/kind; no semantic deduplication or independence claim",
            missingness_policy="Missing source axes remain NA. Raw target availability is unknown where every frozen source axis is absent, not zero.",
            interval_policy="AUC/accuracy have no plotted intervals. Reproduced gap intervals remain descriptive raw-index bootstrap results, not confirmatory or semantic-deduplicated intervals.",
            interpretation="Frozen axes and thresholds; no recalibration, sign selection or site selection. Prequestion contexts repeat across kinds. Late states condition on generated reasoning and answer policy. No causal or subjective-experience inference.",
            matplotlib_version=matplotlib.__version__,
            numpy_version=np.__version__,
            output_sha256={path.name: file_sha(path) for path in sorted(out.iterdir()) if path.is_file()},
            visual_review="pending rendered inspection; separate review receipt follows",
        ),
    )
    print(
        dict(
            out=str(out.resolve()),
            figures=2,
            all_metrics=len(rows),
            plotted_metrics=len(main),
            maximum_absolute_recomputation_error=errors,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.analysis, args.run, args.audit, args.directions, args.out)
