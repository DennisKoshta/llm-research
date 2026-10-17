"""Descriptive figures from locked voluntary-resolution outputs; no refitting."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_concept_candidates import LABELS

FAMILIES = ("objects", "machines", "pooled")
DELIVERIES = ("redundant_evidence", "independent_novelty", "decisive_evidence")
DELIVERY_LABELS = ("Redundant evidence", "Independent novelty", "Decisive evidence")
CATEGORIES = ("report", "repeat", "pattern", "stop", "invalid", "execution_failure", "missing")
COLORS = ("#176B86", "#90BCC5", "#BA8DCC", "#89949E", "#D77255", "#B63737", "#E9CA5B")


def save(fig, out, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=180)
    plt.close(fig)


def plot(source, out):
    source, out = Path(source), Path(out)
    if out.exists():
        raise ValueError("Figure exports are immutable")
    data = json.loads(source.read_text())
    if not (data["complete"] and data["integrity_passed"]):
        raise ValueError("Require complete, integrity-passed analysis")
    out.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    status = "passed" if data["readiness_passed"] else "FAILED; no progression under this revision"
    figure_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
    for ax, family in zip(axes, FAMILIES[:2]):
        cells = [
            next(
                r
                for r in data["denominators"]
                if r["family"] == family and r["delivery"] == delivery and r["menu_layout"] == "averaged"
            )
            for delivery in DELIVERIES
        ]
        left = np.zeros(3)
        for category, color in zip(CATEGORIES, COLORS):
            values = np.array([r["proportions"][category] for r in cells])
            ax.barh(range(3), values, left=left, color=color, label=category.replace("_", " "), height=0.65)
            for index, (value, row) in enumerate(zip(values, cells)):
                if value >= 0.09:
                    ax.text(
                        left[index] + value / 2,
                        index,
                        str(row["counts"][category]),
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
                figure_rows.append(
                    dict(
                        panel="choices",
                        family=family,
                        endpoint=category,
                        comparison=row["delivery"],
                        estimate=float(value),
                        low=None,
                        high=None,
                        denominator=row["assigned_choices"],
                    )
                )
            left += values
        if not np.allclose(left, 1):
            raise ValueError("Choice bars must retain all assigned outcomes")
        ax.set_yticks(range(3), DELIVERY_LABELS)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 5), ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Share of all assigned choices")
        ax.set_title(f"{family.title()} · 12 worlds, 24 choices per delivery", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.09), frameon=False, fontsize=9
    )
    fig.suptitle(
        "Optional services chosen after externally assigned evidence", x=0.02, ha="left", fontsize=14
    )
    fig.text(
        0.02,
        0.025,
        f"Readiness: {status}. Bar labels are counts; both menus included.\nDevelopment sample; object tables have prior partial exposure. No independent-holdout or causal-regulator claim.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.19, 1, 0.92), w_pad=2)
    save(fig, out, "optional-choices")

    contrasts = ("decisive_minus_redundant", "decisive_minus_novelty")
    titles = ("Decisive − redundant (primary)", "Decisive − independent novelty (secondary)")
    axes_order = [row["name"] for row in data["axes"]]
    fig, panels = plt.subplots(1, 2, figsize=(14, 7.4), sharey=True, sharex=True)
    family_colors = ("#176B86", "#824D99", "#26313C")
    for ax, contrast, title in zip(panels, contrasts, titles):
        for family_index, (family, color) in enumerate(zip(FAMILIES, family_colors)):
            for axis_index, axis in enumerate(axes_order):
                row = next(
                    r
                    for r in data["projection_contrasts"]
                    if r["contrast"] == contrast
                    and r["family"] == family
                    and r["axis"] == axis
                    and r["position"] == "record"
                    and r["scale"] == "scaled"
                )
                estimate, interval = row["estimate"], row["ci95"]
                if estimate is None or interval is None:
                    raise ValueError("Unavailable primary projection; explicit missing-data figure needed")
                y = axis_index + (family_index - 1) * 0.21
                ax.plot(interval, [y, y], color=color, linewidth=1.2)
                ax.scatter(
                    [estimate],
                    [y],
                    s=22,
                    color=color,
                    marker="D" if family == "pooled" else "o",
                    label=family.title() if axis_index == 0 else None,
                    zorder=3,
                )
                figure_rows.append(
                    dict(
                        panel="projections",
                        family=family,
                        endpoint=axis,
                        comparison=contrast,
                        estimate=estimate,
                        low=interval[0],
                        high=interval[1],
                        denominator=row["assigned_worlds"],
                    )
                )
        ax.axvline(0, color="#89949E", linewidth=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Projection difference / fixed corpus-A contrast norm")
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        ax.legend(loc="best", frameon=False, fontsize=9)
    panels[0].set_yticks(
        range(len(axes_order)), [LABELS[axis].replace(" description", " candidate") for axis in axes_order]
    )
    panels[0].invert_yaxis()
    fig.suptitle(
        "Natural resolution: changes on the two candidate and six neighboring axes",
        x=0.02,
        ha="left",
        fontsize=14,
    )
    fig.text(
        0.02,
        0.025,
        f"Readiness: {status}. Locked A layer-15/record directions; pre-choice record captures.\n"
        "95% descriptive intervals: 10,000 whole-world bootstrap draws, stratified by family (12 worlds each).\n"
        "Scale is a description-contrast unit, not a probability or comparable psychological effect size. No felt-state classification.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.93), w_pad=2)
    save(fig, out, "natural-projections")
    with (out / "plotted-values.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(figure_rows[0]))
        writer.writeheader()
        writer.writerows(figure_rows)
    receipt = dict(
        source=str(source.resolve()),
        source_byte_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        plotter_byte_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=data["manifest_sha256"],
        readiness_passed=data["readiness_passed"],
        plotted_rows=len(figure_rows),
        visual_review="pending",
        scope="Every choice category in both families and all eight fixed axes at primary record position; both planned contrasts. Prefill sensitivity and receipts remain in analysis exports.",
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())},
    )
    (out / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis, args.out)
