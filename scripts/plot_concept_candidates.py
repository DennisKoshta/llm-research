"""Plot every locked B contrast at the A-selected site, with source bindings."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {
    "curiosity_vs_indifference": "Curiosity description",
    "satisfaction_vs_neutral": "Satisfaction description",
    "confidence_high_vs_low": "Confidence",
    "positive_valence_vs_neutral": "Positive mood",
    "completion_vs_ongoing": "Completion",
    "generic_motivation_vs_indifference": "General motivation",
    "novelty_interest_vs_indifference": "Novelty interest",
    "task_demand_vs_no_demand": "External demand",
}


def plot(source, out):
    source, out = Path(source), Path(out)
    if out.exists():
        raise ValueError("Figure exports are immutable")
    analysis = json.loads(source.read_text())
    out.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharex=True)
    exported = []
    for ax, candidate, color in zip(axes, analysis["candidates"], ("#176B86", "#824D99")):
        rows = [{"contrast": candidate["contrast"], **candidate["b_target"]}, *candidate["b_controls"]]
        means = np.array([row["mean_gap"] for row in rows])
        bounds = np.array([row["descriptive_intervals"]["mean_gap_ci95"] for row in rows])
        y = np.arange(len(rows))
        ax.barh(y, means, color=[color] + ["#ADB6C1"] * 6, height=0.65)
        ax.errorbar(
            means,
            y,
            xerr=np.stack([means - bounds[:, 0], bounds[:, 1] - means]),
            fmt="none",
            ecolor="#26313C",
            capsize=3,
            linewidth=1,
        )
        ax.set_yticks(y, [LABELS[row["contrast"]] for row in rows])
        ax.invert_yaxis()
        ax.axvline(0, color="#596573", linewidth=0.8)
        ax.set_xlabel("Mean paired projection difference (activation units)")
        title = LABELS[candidate["contrast"]].replace(" description", " candidate")
        ax.set_title(
            f"{title}\nLayer {candidate['selected_layer']}, record; 12 new scenario pairs", loc="left"
        )
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        for row, bound in zip(rows, bounds):
            exported.append(
                dict(
                    candidate=candidate["contrast"],
                    evaluated_contrast=row["contrast"],
                    layer=candidate["selected_layer"],
                    position=candidate["position"],
                    scenarios=row["worlds"],
                    mean_gap=row["mean_gap"],
                    ci95_low=float(bound[0]),
                    ci95_high=float(bound[1]),
                    concordance=row["concordance"],
                    auc=row["auc"],
                    fixed_a_threshold_accuracy=row["accuracy"],
                )
            )
    fig.suptitle(
        "Description candidates transfer across authors; curiosity overlaps with novelty and motivation",
        x=0.02,
        ha="left",
        fontsize=14,
    )
    fig.text(
        0.02,
        0.025,
        "Locked corpus-A raw axes; all six nuisance contrasts shown. Bars are signed gaps, not effect sizes across axes.\n"
        "Intervals: 10,000 scenario bootstrap samples, descriptive and conditional on fitted axes. No behavior or felt-state claim.",
        fontsize=9,
        color="#435060",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.92), w_pad=2.0)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"cross-author-transfer.{suffix}", dpi=180)
    plt.close(fig)
    with (out / "plotted-values.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    receipt = dict(
        source=str(source.resolve()),
        source_byte_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        manifest_sha256=analysis["manifest_sha256"],
        plotted_rows=len(exported),
        population="All six nuisance contrasts and the target, at each locked A-selected site; corpus B only",
        visual_review="pending",
        files={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir())},
    )
    (out / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis, args.out)
