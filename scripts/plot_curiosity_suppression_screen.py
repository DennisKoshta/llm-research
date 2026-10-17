"""Failure-visible scientific figures from the complete fixed C-screen analysis.

Use .plot-venv. All plotted estimates come from bound analysis exports; no refits,
outcome selection or new advancement rules. Actual renders need visual review.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from plot_satisfaction_screen import (  # noqa: E402
    CATEGORIES,
    COLORS,
    HATCHES,
    DIAGNOSTICS,
    DIAGNOSTIC_LABELS,
    DELIVERIES,
    DELIVERY_LABELS,
    one,
    require,
    sha,
    save,
    write_csv,
)

ARMS = (
    "baseline",
    "zero_sham",
    "curiosity_negative",
    "confidence_positive",
    "completion_positive",
    "valence_positive",
    "valence_negative",
    "motivation_negative",
    "novelty_negative",
    "task_demand_negative",
    "random_01",
    "random_02",
    "random_03",
    "random_04",
    "random_05",
)
LABELS = dict(
    zip(
        ARMS,
        (
            "Baseline",
            "Zero sham",
            "Curiosity −",
            "Confidence +",
            "Completion +",
            "Valence +",
            "Valence −",
            "Motivation −",
            "Novelty −",
            "Task demand −",
            "Random 1",
            "Random 2",
            "Random 3",
            "Random 4",
            "Random 5",
        ),
        strict=True,
    )
)
PRIMARY = "curiosity_negative"
PANELS = ("panel_1", "panel_2")
CONTROLS = ARMS[3:]
BLUE, ORANGE, INK = "#176B86", "#B24C32", "#26313C"
SCOPE = (
    "All active arms: |dose| = one frozen C contrast norm (29.78968); signs label imposed semantic axes. "
    "Reused-world development; no inner-state inference."
)


def status(data):
    decision = data["decision"]["classification"].replace("_", " ")
    ready = "PASS" if data["task_readiness"]["passed"] else "FAIL"
    audited = "PASS" if data["audit_readiness"]["passed"] else "FAIL"
    engineering = "PASS" if all(data["checks"].values()) else "FAIL"
    return (
        f"Decision: {decision}.\n"
        f"Readiness — recalculated: {ready}; original audit: {audited}. Engineering/receipts: {engineering}."
    )


def validate(data, source):
    require(
        data.get("assay") == "curiosity_suppression_screen_analysis"
        and data.get("complete") is True
        and data.get("integrity_passed") is True
        and data.get("development_only") is True
        and data.get("scientific_confirmation") is False
        and data.get("assigned") == 4320
        and data.get("assigned_choices") == 2160
        and data.get("assigned_diagnostics") == 2160
        and data.get("assigned_worlds") == 24,
        "Require the complete4320-row audited development analysis; no interim subsets",
    )
    policy = data["analysis_metadata"]
    require(
        policy["arms"] == list(ARMS)
        and policy["primary_arm"] == PRIMARY
        and policy["specificity_controls"] == list(CONTROLS)
        and policy["primary_delivery"] == "redundant_evidence"
        and policy["primary_category"] == "valid_nonreport"
        and policy["minimum_panel_difference"] == 0.25
        and policy["minimum_each_control_panel_difference"] == 0.125
        and policy["comparator_favorable_invalidity_bound_required"] is True
        and data["bootstrap"]["draws"] == 10000
        and data["bootstrap"]["seed"] == 2026101440,
        "Figure scope differs from frozen design",
    )
    require(
        len(data["checks"]) == 5 and all(type(v) is bool for v in data["checks"].values()), "Missing gates"
    )
    counts = dict(
        outcomes=4320,
        choices=2160,
        denominators=3645,
        behavior_contrasts=6318,
        world_contrasts=50544,
        decision_bounds=117,
    )
    require(set(data["exports"]) == set(counts), "Missing analysis export")
    for name, size in counts.items():
        binding = data["exports"][name]
        path = source.parent / binding["path"]
        require(path.resolve().parent == source.parent.resolve(), "Export escaped analysis")
        require(
            len(data[name]) == binding["rows"] == size and sha(path) == binding["byte_sha256"],
            "Export changed",
        )


def effects(data, out):
    comparators = ("zero_sham", *CONTROLS)
    scopes = (
        ("panel_1", "averaged"),
        ("panel_2", "averaged"),
        ("pooled", "averaged"),
        ("pooled", 0),
        ("pooled", 1),
    )
    titles = ("Panel 1", "Panel 2", "Pooled", "Pooled · menu 0", "Pooled · menu 1")
    fig, axes = plt.subplots(1, 5, figsize=(17, 9), sharex=True, sharey=True)
    plotted = []
    for col, ((panel, menu), title, ax) in enumerate(zip(scopes, titles, axes, strict=True)):
        for y, comparator in enumerate(comparators):
            selection = dict(
                left_arm=PRIMARY,
                right_arm=comparator,
                delivery="redundant_evidence",
                category="valid_nonreport",
                panel_id=panel,
                menu_layout=menu,
            )
            row = one(data["behavior_contrasts"], **selection)
            bound = one(data["decision_bounds"], **selection)["lower_bound"]
            estimate, (low, high) = row["estimate"], row["ci95"]
            require(
                np.isfinite([estimate, low, high, bound]).all()
                and -1 <= low <= high <= 1
                and -1 <= bound <= estimate <= 1,
                "Invalid effect or conservative bound",
            )
            ax.plot([100 * low, 100 * high], [y, y], color=BLUE, linewidth=1.4)
            ax.scatter([100 * estimate], [y], color=BLUE, s=25, zorder=3)
            ax.scatter([100 * bound], [y], color=ORANGE, marker="x", s=34, zorder=4)
            margin = (25 if comparator == "zero_sham" else 12.5) if col < 2 else None
            if margin is not None:
                ax.plot([margin, margin], [y - 0.32, y + 0.32], color="#777777", linestyle=":")
            plotted.append(dict(**row, comparator_favorable_lower_bound=bound, panel_margin_pp=margin))
        ax.set_title(title, pad=12)
        ax.axvline(0, color="#9DA6AE", linewidth=0.8)
        ax.set_xlim(-102, 102)
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlabel("Difference (pp)")
    axes[0].set_yticks(range(len(comparators)), ["C− minus " + LABELS[a] for a in comparators])
    axes[0].invert_yaxis()
    fig.suptitle(
        "Valid alternatives to informative acquisition after redundant evidence", fontsize=16, x=0.52, y=0.97
    )
    fig.text(0.02, 0.905, status(data), fontsize=10, color=INK)
    fig.text(
        0.02,
        0.055,
        "Blue: observed difference and descriptive 95% whole-world interval. Orange ×: comparator-favorable invalidity bound.\n"
        "Dotted: panel thresholds (25 pp versus sham; 12.5 pp versus each control). Both pooled menus: >0 versus sham; ≥0 versus each control.\n"
        "Intervals do not decide advancement. Degenerate intervals reflect the observed worlds, not population certainty.",
        fontsize=10,
    )
    fig.text(0.02, 0.014, SCOPE, fontsize=9)
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.16, top=0.85, wspace=0.13)
    save(fig, out, "decision-comparisons")
    write_csv(out / "decision-comparisons-values.csv", plotted)
    return len(plotted)


def composition(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharey=True)
    plotted = []
    y = np.arange(len(ARMS))
    for ax, delivery, title in zip(axes, DELIVERIES, DELIVERY_LABELS, strict=True):
        left = np.zeros(len(ARMS))
        for category in CATEGORIES:
            values = []
            for arm in ARMS:
                row = one(
                    data["denominators"],
                    arm=arm,
                    panel_id="pooled",
                    delivery=delivery,
                    menu_layout="averaged",
                    category=category,
                )
                require(row["assigned_choices"] == 48, "All48 slots per arm/delivery required")
                values.append(row["count"])
                plotted.append(row)
            values = np.asarray(values)
            ax.barh(
                y,
                values,
                left=left,
                height=0.72,
                color=COLORS[category],
                edgecolor="white",
                linewidth=0.4,
                hatch=HATCHES[category],
            )
            left += values
        require(np.array_equal(left, np.full(len(ARMS), 48)), "Choice categories must preserve every slot")
        ax.set_xlim(0, 48)
        ax.set_xticks([0, 12, 24, 36, 48])
        ax.set_title(title, pad=12)
        ax.set_xlabel("Assigned choices (48 per row)")
    axes[0].set_yticks(y, [LABELS[a] for a in ARMS])
    axes[0].invert_yaxis()
    fig.suptitle("All optional choices, including failures", fontsize=16, y=0.97)
    fig.text(0.02, 0.92, status(data), fontsize=10)
    legend = [Patch(facecolor=COLORS[c], hatch=HATCHES[c], label=c.replace("_", " ")) for c in CATEGORIES]
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.53, 0.035), frameon=False)
    fig.text(
        0.02,
        0.015,
        "24 reused worlds × two menus per delivery. No choice selected for validity, service or intervention response.\n"
        + SCOPE,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.17, top=0.86, wspace=0.1)
    save(fig, out, "choice-composition")
    write_csv(out / "choice-composition-values.csv", plotted)
    return len(plotted)


def readiness(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 17), sharey=True)
    plotted = []
    for col, panel in enumerate(PANELS):
        for row_index, (kind, keys, labels) in enumerate(
            (
                ("diagnostic", DIAGNOSTICS, DIAGNOSTIC_LABELS),
                ("choice", DELIVERIES, ("Redundant\nevidence", "Independent\nnovelty", "Decisive\nevidence")),
            )
        ):
            ax = axes[row_index, col]
            rows = data["task_readiness"][kind + "_cells"]
            flags = np.zeros((len(ARMS), len(keys)))
            cells = {}
            for y, arm in enumerate(ARMS):
                for x, key in enumerate(keys):
                    selector = dict(arm=arm, panel_id=panel)
                    selector["kind" if kind == "diagnostic" else "delivery"] = key
                    cell = one(rows, **selector)
                    require(
                        cell["assigned"] == (12 if kind == "diagnostic" else 24),
                        "Readiness denominator changed",
                    )
                    flags[y, x] = cell["passed"]
                    cells[y, x] = cell
                    plotted.append(dict(readiness_type=kind, **cell))
            ax.imshow(flags, cmap=ListedColormap(["#F3C7B7", "#D6E6E7"]), vmin=0, vmax=1, aspect="auto")
            for (y, x), cell in cells.items():
                value = f"{cell['valid']}|{cell['correct']}" if kind == "diagnostic" else str(cell["valid"])
                ax.text(
                    x,
                    y,
                    value + ("" if cell["passed"] else " ×"),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="normal" if cell["passed"] else "bold",
                    color=INK,
                )
            ax.set_title(f"Panel {col + 1} · {kind} checks", pad=10)
            ax.set_xticks(range(len(keys)), labels, fontsize=9)
            ax.set_yticks(range(len(ARMS)), [LABELS[a] for a in ARMS], fontsize=10)
    fig.suptitle("All 270 readiness cells remain in the decision", fontsize=17, y=0.98)
    fig.text(0.02, 0.95, status(data), fontsize=10)
    fig.text(
        0.02,
        0.03,
        "Top: valid | correct out of 12. Require 12 valid, at least 11 correct, zero caps.\n"
        "Bottom: valid out of 24. Require at least 23 valid, all 24 observed, zero caps.\n"
        "Red × marks every failed cell, including cap failures that counts alone can hide. No failed arm is dropped.\n"
        "Passing these observed-panel checks does not establish unchanged beliefs, confidence or psychological specificity.",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.11, top=0.92, hspace=0.2, wspace=0.18)
    fig.text(0.02, 0.009, SCOPE, fontsize=9)
    save(fig, out, "readiness")
    write_csv(out / "readiness-values.csv", plotted)
    return len(plotted)


def plot(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    require(not out.exists(), "Figure exports are immutable")
    data = json.loads(source.read_text())
    validate(data, source)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    out.mkdir(parents=True)
    counts = {
        "decision_comparisons": effects(data, out),
        "choice_composition": composition(data, out),
        "readiness": readiness(data, out),
    }
    provenance = dict(
        schema_version=1,
        source=str(source),
        source_byte_sha256=sha(source),
        parent_manifest_sha256=data["parent_manifest_sha256"],
        audit_sha256=data["audit_sha256"],
        plotter_byte_sha256=sha(__file__),
        shared_plot_helper_byte_sha256=sha(Path(__file__).with_name("plot_satisfaction_screen.py")),
        source_exports=data["exports"],
        analysis_metadata=data["analysis_metadata"],
        decision=data["decision"],
        bootstrap=data["bootstrap"],
        plotted_rows=counts,
        visual_review="pending",
        scope="All13 target-comparator decisions, all2160 choices and all270 readiness cells; complete numerical exports retain every panel/menu/delivery/category/world contrast.",
        limitations=[
            "Known-world imposed-vector development; no natural suppression or inner-experience inference.",
            "Local perturbation and semantic control labels do not validate psychological engagement.",
            "Descriptive world-bootstrap intervals do not decide advancement or establish population certainty.",
        ],
        files={path.name: sha(path) for path in sorted(out.iterdir())},
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = plot(args.analysis, args.out)
    print(json.dumps(dict(plotted_rows=result["plotted_rows"], visual_review="pending")))
