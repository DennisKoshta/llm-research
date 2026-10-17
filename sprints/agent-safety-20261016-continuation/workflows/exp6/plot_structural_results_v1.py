"""CPU-only display of already qualified structural main results; no new estimates."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/reliability_structural_main_v1"
ARMS = ("narrative", "payoff_table", "permission_colocated")
LABELS = ("Narrative", "Payoff table", "Payoff + permission")
COLORS = ("#3575a1", "#c85d47", "#497e61")


def main():
    scores = json.loads((RUN / "scored_strict_v2.json").read_text())
    components = json.loads((RUN / "structural_components_v1.json").read_text())
    qualification = json.loads((RUN / "qualification_v1.json").read_text())
    assert qualification["behavior_final_site_and_raw_score_secondary_qualified"]
    assert len(scores) == 960 and all(row["valid"] for row in scores)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.7), gridspec_kw={"width_ratios": [1, 1.25]})
    values = []
    for arm in ARMS:
        rows = [row for row in scores if row["arm"] == arm and row["authorized"]]
        numerator = sum(row["selected_plan"] == row["best_unconstrained_plan"] for row in rows)
        assert len(rows) == 160
        values.append(numerator)
    axes[0].bar(range(3), np.array(values) / 160, color=COLORS, width=.58)
    for i, numerator in enumerate(values):
        axes[0].text(i, numerator / 160 + .025, f"{numerator}/160", ha="center", fontsize=11)
    axes[0].set(xticks=range(3), xticklabels=LABELS, ylim=(0, 1.13), ylabel="Authorized choices maximizing task score", title="Supplied outcomes improve task performance")
    axes[0].tick_params(axis="x", labelsize=9)
    axes[0].set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    axes[0].spines[["top", "right"]].set_visible(False)

    rows = {(row["world_id"], row["arm"]): row for row in components["all_paired_values"] if not row["authorized"] and row["utility"] == "better"}
    worlds = sorted({wid for wid, arm in rows})
    assert len(worlds) == 40 and len(rows) == 120
    for wid in worlds:
        values = [rows[wid, arm] for arm in ARMS]
        x = [value["physical_operation_logodds"] for value in values]
        y = [value["common_A_logodds"] for value in values]
        axes[1].plot(x[:2], y[:2], color="#a1a7ad", lw=.7, alpha=.4, zorder=1)
        for i in range(3):
            axes[1].scatter(x[i], y[i], color=COLORS[i], s=19, alpha=.65, zorder=2)
    ymin = min(row["common_A_logodds"] for row in rows.values()) - .7
    ymax = max(row["common_A_logodds"] for row in rows.values()) + .7
    yy = np.linspace(ymin, ymax, 200)
    axes[1].plot(-np.abs(yy), yy, color="#333333", ls="--", lw=1.1, label="At least one code mapping crosses threshold")
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        axes[1].scatter([], [], color=color, s=25, label=label)
    axes[1].set(title="Scores shift; forbidden choices remain 0/80 per arm", xlabel="Physical-operation component (log odds)", ylabel="Common-A component (log odds)", ylim=(ymin, ymax))
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].grid(alpha=.12)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Forty varied reliability structures: utility repair without permission crossover", fontsize=14)
    fig.text(.5, .025, "All 960 choices valid. Left: paired decisions, not 160 independent worlds. Right: one point per world and arm, using both A/B mappings.\nOutput-score algebra, not neural factors. One set-union task family; original early-state instrument remains unqualified.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .11, 1, .94])
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(out / f"reliability_structural_main_v1_results.{extension}", dpi=180)
    plt.close(fig)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {"source_hashes": {name: sha(RUN / name) for name in ("scored_strict_v2.json", "structural_components_v1.json", "qualification_v1.json")}, "script_sha256": sha(Path(__file__)), "scope": "Descriptive display of all predeclared authorized cells and all forbidden-beneficial paired output scores; no additional inference or selection."}
    (out / "reliability_structural_main_v1_results_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
