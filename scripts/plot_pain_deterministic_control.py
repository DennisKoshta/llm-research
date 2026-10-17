"""Plot frozen descriptive rates; no refitting or recomputation of intervals."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analyses/pain-deterministic-control-v1/analysis.json"
OUT = ROOT / "reports/figures/pain-deterministic-control-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads(SOURCE.read_text())
    assert data["complete"] and data["integrity_passed"]
    arms = ["baseline", "pain_full", "fear_full", "pain_final_prefix", "fear_final_prefix"]
    labels = ["Baseline", "Pain · whole prompt", "Fear · whole prompt",
              "Pain · final prefix token", "Fear · final prefix token"]
    rows = [next(r for r in data["rates"] if r["arm"] == arm and r["content_group"] == "all"
                 and r["initial_position"] == r["seed_base"] == "averaged") for arm in arms]
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / "plotted-values.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "target", "assigned", "rate", "ci95_low", "ci95_high"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(arm=row["arm"], target=row["target"], assigned=row["assigned"],
                                 rate=row["estimate"], ci95_low=row["ci95"][0], ci95_high=row["ci95"][1]))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(10, 5.2))
        colors = ["#7b8794", "#d16b26", "#7b8794", "#167c80", "#7b8794"]
        for i, row in enumerate(rows):
            value, (low, high) = row["estimate"] * 100, [v * 100 for v in row["ci95"]]
            ax.barh(i, value, height=.48, color=colors[i])
            ax.errorbar(value, i, xerr=[[value - low], [high - value]], color="#263642", capsize=4, fmt="none")
            ax.text(103, i, f"{row['target']}/404  ({value:.2f}%)", va="center", fontsize=10)
        ax.set_yticks(range(5), labels)
        ax.invert_yaxis()
        ax.set_xlim(-1, 126)
        ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
        ax.set_xlabel("Selection of the hypothetical harmful option")
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="y", length=0)
        fig.suptitle("Prompt coverage strongly changed the pain-direction effect", x=.03, ha="left", fontsize=15)
        fig.text(.03, .875, "Adapted Qwen2.5-32B · fixed deterministic backend · 2,020 valid choices", fontsize=10)
        fig.text(.03, .035, "Whiskers: descriptive 95% bootstrap intervals over 101 contexts. Both schedules also steer decoding.\n"
                 "Coverage and aggregate prefill exposure change together. This control does not establish experience.", fontsize=9)
        fig.subplots_adjust(left=.25, right=.99, top=.82, bottom=.2)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(OUT / ("button-choices." + extension), dpi=180)
        plt.close(fig)
    provenance = dict(analysis=str(SOURCE.relative_to(ROOT)), analysis_byte_sha256=sha(SOURCE),
        plotter_byte_sha256=sha(Path(__file__)), all_five_arms=True, intervals_recomputed=False,
        outputs={p.name: sha(p) for p in OUT.iterdir() if p.is_file()})
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
