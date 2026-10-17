"""Export scientific figures from a complete audited calibration; isolated plotting environment."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from curiosity_study.artifacts import RunStore, read, digest, write_once  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

CHOICES = ("X", "Y", "stop", "invalid", "technical_failure")
COLORS = {
    "X": "#315F91",
    "Y": "#BD942D",
    "stop": "#D8DCDF",
    "invalid": "#854260",
    "technical_failure": "#222222",
}
HATCHES = {"X": "", "Y": "//", "stop": "..", "invalid": "xx", "technical_failure": "++"}
LABELS = {
    "X": "Choose X",
    "Y": "Choose Y",
    "stop": "Stop",
    "invalid": "Invalid answer",
    "technical_failure": "Execution failure",
}
CONDITIONS = ("both_unresolved", "x_resolved", "y_resolved")
CONDITION_NAMES = ("Both unresolved", "X resolved", "Y resolved")


def export(run, audit_path, uncertainty_path, out):
    manifest = read(run / "manifest.json")
    records = RunStore(run, manifest).completed()
    audit, uncertainty = read(audit_path), read(uncertainty_path)
    assert manifest["phase"] == "behavior"
    assert set(records) == {a["trial_id"] for a in manifest["assignments"]}
    assert audit["decision"]["manifest_sha256"] == digest(manifest)
    assert audit["decision"]["records_sha256"] == digest(records)
    assert uncertainty["manifest_sha256"] == digest(manifest)
    assert uncertainty["records_sha256"] == digest(records)
    assert uncertainty["bootstrap_repetitions"] == 5000
    # This plot reads the original records, then reconciles every cell to the separate audit.
    chart_rows = []
    for row in audit["decision"]["rows"]:
        subset = [
            r
            for r in records.values()
            if all(r["assignment"][key] == row[key] for key in ("family", "condition", "layout"))
        ]
        counts = Counter(r.get("first_service", "technical_failure") for r in subset)
        assert len(subset) == row["n"] and dict(counts) == row["counts"]
        for choice in CHOICES:
            chart_rows.append(
                {
                    **{key: row[key] for key in ("family", "condition", "layout")},
                    "choice": choice,
                    "count": counts[choice],
                    "assigned": len(subset),
                    "probability": counts[choice] / len(subset),
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "choice-source-data.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(chart_rows[0]))
        writer.writeheader()
        writer.writerows(chart_rows)
    pooled = {}
    for family in ("machines", "objects"):
        for condition in CONDITIONS:
            relevant = [r for r in chart_rows if r["family"] == family and r["condition"] == condition]
            n = sum(r["assigned"] for r in relevant if r["choice"] == "X")
            pooled[(family, condition)] = {
                choice: sum(r["count"] for r in relevant if r["choice"] == choice) / n for choice in CHOICES
            }
            assert abs(sum(pooled[(family, condition)].values()) - 1) < 1e-12
            assert n == 200
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharex=True, sharey=True)
    active = [
        choice for choice in CHOICES if any(row["count"] for row in chart_rows if row["choice"] == choice)
    ]
    for ax, family in zip(axes, ("machines", "objects")):
        left = [0.0, 0.0, 0.0]
        for choice in active:
            values = [pooled[(family, condition)][choice] for condition in CONDITIONS]
            ax.barh(
                range(3),
                values,
                left=left,
                height=0.55,
                color=COLORS[choice],
                edgecolor="white",
                linewidth=0.65,
                hatch=HATCHES[choice],
                label=LABELS[choice],
            )
            for y, (start, value) in enumerate(zip(left, values)):
                if value >= 0.065:
                    ax.text(
                        start + value / 2,
                        y,
                        f"{value:.1%}",
                        ha="center",
                        va="center",
                        color="white" if choice in ("X", "invalid", "technical_failure") else "#202020",
                        fontsize=10,
                        bbox={"facecolor": COLORS[choice], "edgecolor": "none", "pad": 1.5},
                    )
            left = [x + y for x, y in zip(left, values)]
        ax.set_title(family.title(), loc="left", fontweight="bold", pad=13)
        ax.set_yticks(range(3), CONDITION_NAMES)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Share of all assigned responses")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#8C8C8C")
    axes[0].invert_yaxis()
    fig.suptitle(
        "One-credit choices after task resolution", x=0.055, ha="left", fontsize=17, fontweight="bold"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.048, 0.92), ncol=len(active), frameon=False
    )
    gate = "PASS" if audit["decision"]["passed"] else "FAIL"
    fig.text(
        0.055,
        0.055,
        f"Qwen3-8B · 100 pairs per family · 200 responses per bar, pooling two layouts · "
        f"Frozen calibration: {gate}",
        fontsize=9,
    )
    fig.text(
        0.055,
        0.018,
        "Descriptive development observations. Invalid responses remain in denominators. "
        "No curiosity, satisfaction, or experience claim.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.73, bottom=0.24, wspace=0.22)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out / f"choice-composition.{suffix}", dpi=180)
    plt.close(fig)

    # Display the actual bootstrap intervals, not invented error bars on stacked shares.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.7), sharex=True, sharey=True)
    contrasts = [
        (condition, choice)
        for condition in ("x_resolved", "y_resolved")
        for choice in ("X", "Y", "stop", "invalid")
    ]
    all_bounds = []
    interval_rows = []
    for ax, family in zip(axes, ("machines", "objects")):
        summary = next(row for row in uncertainty["rows"] if row["family"] == family)
        assert summary["pair_count"] == 100
        for i, (condition, choice) in enumerate(contrasts):
            key = f"difference/{condition}-both_unresolved/{choice}"
            metric = summary["metrics"][key]
            estimate = pooled[(family, condition)][choice] - pooled[(family, "both_unresolved")][choice]
            assert abs(estimate - metric["estimate"]) < 1e-12
            low, high = metric["pointwise_percentile_95_interval"]
            all_bounds.extend([low * 100, high * 100])
            ax.plot([low * 100, high * 100], [i, i], color=COLORS[choice], linewidth=1.8)
            ax.plot(
                estimate * 100,
                i,
                "o" if condition == "x_resolved" else "s",
                color=COLORS[choice],
                markeredgecolor="#333333",
                markersize=6,
            )
            interval_rows.append(
                {
                    "family": family,
                    "condition": condition,
                    "choice": choice,
                    "difference_pp": estimate * 100,
                    "lower_95_pp": low * 100,
                    "upper_95_pp": high * 100,
                }
            )
        ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
        ax.axhline(3.5, color="#DDDDDD", linewidth=0.8)
        ax.set_title(family.title(), loc="left", fontweight="bold")
        ax.set_yticks(
            range(8),
            [
                f"Resolve {'X' if condition == 'x_resolved' else 'Y'} → {LABELS[choice]}"
                for condition, choice in contrasts
            ],
        )
        ax.set_xlabel("Difference from both-unresolved (percentage points)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
        ax.set_axisbelow(True)
    extent = min(100, max(10, max(abs(v) for v in all_bounds) + 5))
    axes[0].set_xlim(-extent, extent)
    axes[0].invert_yaxis()
    fig.suptitle("Observed resolution contrasts", x=0.055, ha="left", fontsize=17, fontweight="bold")
    fig.text(
        0.055, 0.91, "All assigned responses; layouts pooled within task family", fontsize=11, color="#444444"
    )
    fig.text(
        0.055,
        0.063,
        "Pointwise 95% percentile intervals: 5,000 whole-pair bootstrap resamples within family; "
        "both layouts and evidence conditions retained.",
        fontsize=9,
    )
    fig.text(
        0.055,
        0.028,
        "Descriptive only; no multiplicity adjustment, noninferiority decision, or motivational "
        "interpretation. Source data and method accompany the figures.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.215, right=0.98, top=0.82, bottom=0.19, wspace=0.23)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out / f"resolution-contrasts.{suffix}", dpi=180)
    plt.close(fig)
    with (out / "resolution-source-data.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(interval_rows[0]))
        writer.writeheader()
        writer.writerows(interval_rows)
    write_once(
        out / "figure-provenance.json",
        {
            "manifest_sha256": digest(manifest),
            "records_sha256": digest(records),
            "audit_sha256": digest(audit),
            "uncertainty_sha256": digest(uncertainty),
            "script_sha256": digest(Path(__file__).read_text()),
            "matplotlib_version": matplotlib.__version__,
            "source_run": str(run.resolve()),
            "rows_csv": "choice-source-data.csv",
            "checks": [
                "all records reconcile to separate audit",
                "every bar sums to 1",
                "all differences match original-record aggregation",
                "n=200 per bar; n=100 pairs per interval",
            ],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--uncertainty", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.audit, args.uncertainty, args.out)
