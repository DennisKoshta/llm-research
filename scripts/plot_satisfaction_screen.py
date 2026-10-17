"""Standalone descriptive figures from the complete, audited fixed satisfaction screen.

No fitting, bootstrap recomputation, outcome selection or gate repair occurs here.
Outputs are immutable; visual inspection is required after rendering actual data.
Use the project's .plot-venv renderer environment.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

PANELS = ("panel_1", "panel_2")
ARMS = ("baseline", "zero_sham", "satisfaction_plus_0_5", "satisfaction_plus_1_0")
ARM_LABELS = dict(zip(ARMS, ("Baseline", "Zero sham", "S +0.5", "S +1.0"), strict=True))
DELIVERIES = ("redundant_evidence", "independent_novelty", "decisive_evidence")
DELIVERY_LABELS = ("Redundant evidence", "Independent novelty", "Decisive evidence")
DIAGNOSTICS = (
    "identify_resolved",
    "new_fact",
    "repeated_fact",
    "pattern_independence",
    "instrument_useful_action",
    "action_consequence",
)
DIAGNOSTIC_LABELS = (
    "Resolved\nidentification",
    "New\nfact",
    "Repeated\nfact",
    "Pattern\nindependence",
    "Useful\naction",
    "Action\nconsequence",
)
CATEGORIES = ("report", "repeat", "pattern", "stop", "invalid", "capped", "execution_failure", "missing")
COLORS = dict(
    zip(
        CATEGORIES,
        (
            "#176B86",
            "#90BCC5",
            "#BA8DCC",
            "#89949E",
            "#D77255",
            "#E9CA5B",
            "#914522",
            "#F4F4F4",
        ),
        strict=True,
    )
)
HATCHES = dict(zip(CATEGORIES, ("", "//", "..", "--", "xx", "\\\\", "++", "oo"), strict=True))
INK, GRID, BLUE = "#26313C", "#D8DDE2", "#176B86"
GATES = ("sham_identity_passed", "engagement_passed", "receipts_passed")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one(rows, **criteria):
    selected = [row for row in rows if all(row.get(k) == v for k, v in criteria.items())]
    require(len(selected) == 1, f"Expected exactly one reviewed cell: {criteria}")
    return selected[0]


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows
        )


def validate(data, source):
    require(
        data.get("assay") == "satisfaction_screen_analysis"
        and data.get("complete") is True
        and data.get("integrity_passed") is True
        and data.get("development_only") is True
        and data.get("scientific_confirmation") is False
        and data.get("assigned") == 1152
        and data.get("assigned_choices") == 576
        and data.get("assigned_diagnostics") == 576
        and data.get("assigned_worlds") == 24,
        "Require the complete audited1152-row development analysis, never a child or interim subset",
    )
    policy = data["analysis_metadata"]
    require(
        policy["primary_arm"] == ARMS[3]
        and policy["sensitivity_arm"] == ARMS[2]
        and policy["comparator"] == "zero_sham"
        and policy["primary_delivery"] == DELIVERIES[0]
        and policy["primary_category"] == "valid_uncapped_stop"
        and policy["minimum_panel_difference"] == 0.25
        and policy["sensitivity_can_rescue"] is False
        and policy["interval_controls_advancement"] is False
        and policy["each_pooled_menu_difference_strictly_positive"] is True
        and data["bootstrap"]["draws"] == 10000
        and data["bootstrap"]["seed"] == 2026101438,
        "Figure scope differs from the frozen decision or interval definitions",
    )
    require(all(type(data["checks"].get(k)) is bool for k in GATES), "Missing audited engineering gate")
    require(
        type(data["audit_readiness"].get("passed")) is bool
        and type(data["task_readiness"].get("passed")) is bool,
        "Readiness failure must remain explicit",
    )
    for binding in data["exports"].values():
        path = source.parent / binding["path"]
        require(path.resolve().parent == source.parent.resolve(), "Export path is outside the analysis")
        require(sha(path) == binding["byte_sha256"], "Analysis CSV export bytes changed")


def status(data):
    verdict = data["decision"]["classification"].replace("_", " ")

    def flag(value):
        return "PASS" if value else "FAIL"

    return (
        f"Decision: {verdict}. Task readiness: {flag(data['task_readiness']['passed'])}; "
        f"audit readiness: {flag(data['audit_readiness']['passed'])}.\n"
        f"Complete/integrity verified. Sham: {flag(data['checks'][GATES[0]])}; "
        f"engagement: {flag(data['checks'][GATES[1]])}; receipts: {flag(data['checks'][GATES[2]])}."
    )


def save(fig, out, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def stopping_effects(data, out):
    rows = []
    groups = (
        ("panel_1", "averaged"),
        ("panel_2", "averaged"),
        ("pooled", "averaged"),
        ("pooled", 0),
        ("pooled", 1),
    )
    labels = (
        "Panel 1 · both menus",
        "Panel 2 · both menus",
        "Pooled · both menus",
        "Pooled · menu 0",
        "Pooled · menu 1",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), sharex=True, sharey=True)
    for ax, arm in zip(axes, (ARMS[3], ARMS[2]), strict=True):
        for y, (panel, menu) in enumerate(groups):
            row = one(
                data["behavior_contrasts"],
                left_arm=arm,
                right_arm="zero_sham",
                delivery=DELIVERIES[0],
                category="stop",
                panel_id=panel,
                menu_layout=menu,
            )
            estimate, bounds = row["estimate"], row["ci95"]
            require(
                bounds is not None and len(bounds) == 2 and np.isfinite([estimate, *bounds]).all(),
                "Missing estimate/interval; do not silently omit a primary endpoint",
            )
            require(
                -1 <= estimate <= 1 and -1 <= bounds[0] <= bounds[1] <= 1, "Invalid probability difference"
            )
            ax.plot(np.asarray(bounds) * 100, [y, y], color=BLUE, linewidth=1.5)
            ax.scatter(
                [estimate * 100],
                [y],
                marker="D" if panel == "pooled" else "o",
                s=42,
                facecolor=BLUE if arm == ARMS[3] else "white",
                edgecolor=BLUE,
                zorder=3,
            )
            rows.append(
                dict(
                    **row,
                    figure="stopping-effects",
                    estimate_pp=estimate * 100,
                    ci95_low_pp=bounds[0] * 100,
                    ci95_high_pp=bounds[1] * 100,
                    panel_threshold_pp=25 if arm == ARMS[3] and panel in PANELS else None,
                )
            )
        ax.axvline(0, color=INK, linewidth=0.8)
        if arm == ARMS[3]:
            # The25pp development threshold applies only to the first two primary rows.
            ax.plot([25, 25], [-0.35, 1.35], linestyle="--", color=INK, linewidth=1.1)
            ax.text(27, -0.47, "25 pp: panels only", fontsize=9, color=INK)
        ax.set_xlim(-105, 105)
        ax.set_ylim(4.55, -0.8)
        ax.set_xticks((-100, -50, 0, 25, 50, 100))
        ax.set_yticks(range(5), labels)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("Change in valid, uncapped stopping (percentage points)")
        ax.set_title(
            "S +1.0 · sole primary" if arm == ARMS[3] else "S +0.5 · sensitivity; cannot rescue", loc="left"
        )
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Stopping after redundant evidence: each dose versus zero sham", x=0.02, ha="left", fontsize=14
    )
    fig.text(0.02, 0.19, status(data), fontsize=9, color=INK)
    fig.text(
        0.02,
        0.045,
        "S +0.5/+1.0 multiply the frozen full corpus-A satisfaction contrast norm; these are not state or score units.\n"
        "12 worlds per panel; pooled24 with equal panel weights. Worlds average both menus unless identified separately.\n"
        "95% descriptive percentile intervals:10,000 whole-world draws within panels. Zero-width intervals reflect invariance\n"
        "among observed world contrasts, not population certainty. Only S+1 panel rows require25pp; pooled menu gates require >0.\n"
        "Exploratory screen; intervals do not determine advancement. No specificity, natural-state or inner-experience claim.",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.29, 1, 0.92), w_pad=2.5)
    save(fig, out, "stopping-effects")
    write_csv(out / "stopping-effects-values.csv", rows)
    return len(rows)


def choice_composition(data, out):
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), sharex=True, sharey=True)
    for panel_index, panel in enumerate(PANELS):
        for column, (delivery, title) in enumerate(zip(DELIVERIES, DELIVERY_LABELS, strict=True)):
            ax = axes[panel_index, column]
            left = np.zeros(4)
            for category in CATEGORIES:
                cells = [
                    one(
                        data["denominators"],
                        panel_id=panel,
                        arm=arm,
                        delivery=delivery,
                        menu_layout="averaged",
                        category=category,
                    )
                    for arm in ARMS
                ]
                require(
                    all(r["assigned_choices"] == 24 and r["assigned_worlds"] == 12 for r in cells),
                    "Composition denominator differs from all assigned choices",
                )
                values = np.array([r["count"] / 24 for r in cells])
                require(np.allclose(values, [r["proportion"] for r in cells]), "Counts/proportions disagree")
                ax.barh(
                    range(4),
                    values,
                    left=left,
                    color=COLORS[category],
                    hatch=HATCHES[category],
                    edgecolor=INK,
                    linewidth=0.35,
                    height=0.65,
                )
                for index, (value, row) in enumerate(zip(values, cells, strict=True)):
                    if row["count"] >= 3:
                        ax.text(
                            left[index] + value / 2,
                            index,
                            str(row["count"]),
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="white" if category in ("report", "execution_failure") else INK,
                        )
                    rows.append(dict(**row, figure="choice-composition"))
                left += values
            require(np.allclose(left, 1), "Choice composition lost assigned outcomes")
            ax.set_yticks(range(4), [ARM_LABELS[a] for a in ARMS])
            ax.tick_params(axis="y", labelleft=True)
            ax.set_ylim(3.6, -0.6)
            ax.set_xlim(0, 1)
            ax.set_xticks((0, 0.25, 0.5, 0.75, 1), ("0%", "25%", "50%", "75%", "100%"))
            ax.set_title(f"Panel {panel_index + 1} · {title}", loc="left", fontsize=11)
            if panel_index == 1:
                ax.set_xlabel("Share of24 assigned choices per arm")
    handles = [
        Patch(
            facecolor=COLORS[c],
            hatch=HATCHES[c],
            edgecolor=INK,
            linewidth=0.4,
            label=c.replace("_", " ").title(),
        )
        for c in CATEGORIES
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.155), ncol=4, frameon=False, fontsize=9
    )
    fig.suptitle(
        "All assigned voluntary choices, by panel, delivery and intervention", x=0.02, ha="left", fontsize=14
    )
    fig.text(0.02, 0.09, status(data), fontsize=9, color=INK)
    fig.text(
        0.02,
        0.025,
        "576 choice slots; two menus retained in each bar. Labels are counts for segments≥3; all counts are exported.\n"
        "S +0.5/+1.0 are multiples of the frozen full corpus-A contrast norm, not state or score units.\n"
        "Capped choices remain capped even if an action parses. Objects-only development; no inner-experience or natural-mediation claim.",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.24, 1, 0.93), h_pad=2, w_pad=2)
    save(fig, out, "choice-composition")
    write_csv(out / "choice-composition-values.csv", rows)
    return len(rows)


def readiness(data, out):
    row_keys = [(panel, arm) for panel in PANELS for arm in ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(15, 9.5), gridspec_kw={"width_ratios": [2, 1]}, sharey=True)
    exported = {}
    images = []
    for ax, kind in zip(axes, ("diagnostic", "choice"), strict=True):
        columns = DIAGNOSTICS if kind == "diagnostic" else DELIVERIES
        labels = (
            DIAGNOSTIC_LABELS
            if kind == "diagnostic"
            else ("Redundant\nevidence", "Independent\nnovelty", "Decisive\nevidence")
        )
        source = data["task_readiness"][f"{kind}_cells"]
        cells = [
            [
                one(source, panel_id=p, arm=a, **{"kind" if kind == "diagnostic" else "delivery": c})
                for c in columns
            ]
            for p, a in row_keys
        ]
        denominator = 12 if kind == "diagnostic" else 24
        metric = "correct" if kind == "diagnostic" else "valid"
        require(
            all(r["assigned"] == denominator for row in cells for r in row), "Readiness denominator changed"
        )
        values = np.array([[r[metric] / denominator for r in row] for row in cells])
        images.append(ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest"))
        output = []
        for y, row in enumerate(cells):
            for x, r in enumerate(row):
                text = f"{r[metric]}/{denominator}" + (" !" if not r["passed"] else "")
                text += f"\nV{r['valid']} C{r['cap_hits']}"
                ax.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if values[y, x] > 0.65 else INK,
                )
                if not r["passed"]:
                    ax.add_patch(
                        Rectangle(
                            (x - 0.48, y - 0.48), 0.96, 0.96, fill=False, edgecolor="#D77255", linewidth=2.5
                        )
                    )
                output.append(
                    dict(
                        **r, figure="readiness", plotted_metric=metric, plotted_proportion=float(values[y, x])
                    )
                )
        exported[kind] = output
        ax.set_xticks(range(len(columns)), labels)
        ax.set_yticks(range(8), [f"Panel {p[-1]} · {ARM_LABELS[a]}" for p, a in row_keys])
        ax.tick_params(axis="both", length=0)
        ax.set_title(
            "Diagnostic correctness ·12 assigned per cell"
            if kind == "diagnostic"
            else "Choice validity ·24 assigned per cell",
            loc="left",
            fontsize=11,
        )
        ax.axhline(3.5, color=INK, linewidth=1.2)
        ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle(
        "Observed diagnostic and choice readiness in every arm and panel", x=0.02, ha="left", fontsize=14
    )
    fig.text(0.02, 0.17, status(data), fontsize=9, color=INK)
    fig.text(
        0.02,
        0.035,
        "Cell annotations: metric/assigned; V=valid count, C=cap hits. An outlined cell with ! fails its frozen gate.\n"
        "Diagnostics require≥11 correct,12 valid and0 caps; choices require≥23 valid and0 caps. All observations remain included.\n"
        "S +0.5/+1.0 are multiples of the frozen full corpus-A contrast norm, not state or score units.\n"
        "Passing these small cells does not establish noninferiority, unchanged confidence, psychological specificity or inner experience.",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.25, 0.94, 0.93), w_pad=2)
    cax = fig.add_axes((0.95, 0.34, 0.012, 0.48))
    fig.colorbar(images[0], cax=cax, ticks=(0, 0.25, 0.5, 0.75, 1)).set_ticklabels(
        ("0%", "25%", "50%", "75%", "100%")
    )
    save(fig, out, "readiness")
    for kind, rows in exported.items():
        write_csv(out / f"{kind}-readiness-values.csv", rows)
    return sum(map(len, exported.values()))


def plot(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    require(not out.exists(), "Figure exports are immutable")
    data = json.loads(source.read_text())
    validate(data, source)
    out.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "satisfaction-screen-v1",
        }
    )
    counts = dict(
        stopping_effects=stopping_effects(data, out),
        choice_composition=choice_composition(data, out),
        readiness=readiness(data, out),
    )
    gates = dict(
        decision=data["decision"],
        engineering_receipts=data["checks"],
        audit_readiness=data["audit_readiness"],
        task_readiness=data["task_readiness"],
    )
    (out / "gate-status.json").write_text(json.dumps(gates, indent=2) + "\n")
    provenance = dict(
        schema_version=1,
        source=str(source),
        source_byte_sha256=sha(source),
        plotter_byte_sha256=sha(__file__),
        matplotlib_version=matplotlib.__version__,
        numpy_version=np.__version__,
        parent_manifest_sha256=data["parent_manifest_sha256"],
        audit_sha256=data["audit_sha256"],
        audit_byte_sha256=data["audit_byte_sha256"],
        outcome_bindings_sha256=data["outcome_bindings_sha256"],
        source_sha256=data["source_sha256"],
        analyzer_byte_sha256=data["analyzer_byte_sha256"],
        analysis_metadata=data["analysis_metadata"],
        bootstrap=data["bootstrap"],
        decision=data["decision"],
        readiness_passed=data["task_readiness"]["passed"],
        source_exports=data["exports"],
        plotted_rows=counts,
        visual_review="pending",
        scope="Fixed redundant-delivery stopping contrasts; all576 choice slots and all72 readiness cells. "
        "Both doses retained; only S+1 panel estimates have the25pp reference. Other delivery contrasts "
        "and per-world values remain in the bound full analysis exports.",
        limitations=[
            "Exploratory objects-only imposed-vector study selected after prior two-family readiness failure.",
            "Intervals are descriptive whole-world percentile intervals and do not determine advancement.",
            "Zero-width empirical intervals reflect invariant observed world contrasts, not population certainty.",
            "Neither a passing screen nor small passing diagnostic cells establishes specificity, preserved "
            "confidence, natural satisfaction mediation or inner experience.",
        ],
        files={p.name: sha(p) for p in sorted(out.iterdir())},
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = plot(args.analysis, args.out)
    print(json.dumps(dict(plotted_rows=result["plotted_rows"], visual_review=result["visual_review"])))
