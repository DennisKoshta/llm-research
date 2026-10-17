"""Export descriptive validation figures from a completed, manifest-bound audit.

Run in .plot-venv after the full audit is available. Counts come from the audited
all-assignment summaries; no response-level independence or interval is assumed.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from curiosity_study.artifacts import digest, read, write_once  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402

FAMILIES = ("objects", "machines")
KINDS = ("new", "repeat", "instrument", "memory", "count")
KIND_LABELS = {
    "new": "New information",
    "repeat": "Already known",
    "instrument": "Goal-directed choice",
    "memory": "Observed-value recall",
    "count": "Candidate count",
}
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
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
BLUE, GOLD, INK = "#315F91", "#BD942D", "#222222"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chart_rows(manifest, audit):
    """Reconcile audit counts with the frozen factorial allocation, without fitting."""
    require(audit.get("complete") is True, "The full validation audit must be complete")
    require(audit["manifest_sha256"] == digest(manifest), "Audit and parent manifest differ")
    require(
        manifest["phase"] == audit["phase"] == "validation", "This exporter is for the original validation"
    )
    require(manifest["assay"] == "thinking_information", "Unexpected assay")
    require(manifest["model"]["model_id"] == "Qwen/Qwen3-32B", "Unexpected model")
    assignments = manifest["diagnostics"]
    require(digest(assignments) == manifest["assignment_sha256"], "Parent assignment hash mismatch")
    require(len(assignments) == audit["assigned"] == 4000, "Expected all 4,000 validation assignments")
    require(len({a["trial_id"] for a in assignments}) == 4000, "Duplicate parent trial IDs")
    criteria = manifest["validation_criteria"]
    require(
        criteria["minimum_distinct_items_per_cell"] == 100
        and criteria["minimum_accuracy"] == 0.9
        and criteria["minimum_valid_format"] == 0.95,
        "Validation gates differ from the planned 100-item, 90%/95% criteria",
    )
    allocation = defaultdict(list)
    for a in assignments:
        require(
            a["task_revision"] == "equal-uncertainty-v1" and len(a["known"]) == 1, "Unexpected factorial task"
        )
        allocation[(a["family"], a["kind"], a["known"][0], a["query"])].append(a)
    required = {(f, k, o, q) for f in FAMILIES for k in KINDS for o, q in CELLS}
    require(set(allocation) == required, "Missing or extra factorial cells")
    source_cells = audit["gates"]["cells"]
    keyed = {(c["family"], c["kind"], c["known"], c["query"]): c for c in source_cells}
    require(len(source_cells) == len(keyed) == 40 and set(keyed) == required, "Audit cell coverage differs")
    rows = []
    for family in FAMILIES:
        world_ids = None
        for kind in KINDS:
            for known, query in CELLS:
                key = family, kind, known, query
                source = keyed[key]
                assigned = allocation[key]
                worlds = {(a["seed"], a["index"]) for a in assigned}
                require(
                    len(assigned)
                    == len(worlds)
                    == source["n"]
                    == source["items"]
                    == source["completed"]
                    == 100,
                    "Each cell must contain its 100 assigned world indices",
                )
                if world_ids is None:
                    world_ids = worlds
                require(worlds == world_ids, "All conditions and tasks must retain the same world indices")
                correct, valid = source["correct"], source["valid"]
                require(
                    type(correct) is int and type(valid) is int and 0 <= correct <= valid <= 100,
                    "Invalid audited cell counts",
                )
                passed = correct >= 90 and valid >= 95
                require(source["passed"] is passed, "Cell gate does not reconcile to its exact counts")
                rows.append(
                    dict(
                        family=family,
                        kind=kind,
                        known=known,
                        query=query,
                        information_bits=int(known != query),
                        assigned=100,
                        distinct_world_indices=100,
                        completed=100,
                        correct=correct,
                        valid=valid,
                        accuracy=correct / 100,
                        validity=valid / 100,
                        minimum_accuracy=0.9,
                        minimum_validity=0.95,
                        accuracy_passed=correct >= 90,
                        validity_passed=valid >= 95,
                        cell_passed=passed,
                    )
                )
    require(sum(r["correct"] for r in rows) == audit["correct"], "Correct totals do not reconcile")
    require(sum(r["valid"] for r in rows) == audit["valid"], "Validity totals do not reconcile")
    require(audit["gates"]["passed"] is all(r["cell_passed"] for r in rows), "Overall cell gate mismatch")
    reports = audit["partitions"]
    require(sum(r["assigned"] for r in reports) == 4000, "Partition assigned totals do not reconcile")
    for field in ("correct", "valid", "generated_tokens", "cap_hits"):
        require(sum(r[field] for r in reports) == audit[field], f"Partition {field} totals do not reconcile")
    require(0 <= audit["cap_hits"] <= 4000, "Invalid output-cap count")
    layers = manifest["capture_plan"]["layers"]
    require(manifest["capture_plan"]["steps"] == [32, 128, 512], "Unexpected capture schedule")
    availability = defaultdict(Counter)
    for report in reports:
        require(report["complete"] is True, "A partition audit is incomplete")
        for r in report["capture_availability"]:
            key = r["family"], r["kind"], r["information_bits"], r["position"]
            require(
                type(r["assigned"]) is int
                and type(r["available"]) is int
                and 0 <= r["available"] <= r["assigned"],
                "Invalid capture-availability counts",
            )
            availability[key].update(assigned=r["assigned"], available=r["available"])
    expected_availability = {(f, k, g, p) for f in FAMILIES for k in KINDS for g in (0, 1) for p in POSITIONS}
    require(set(availability) == expected_availability, "Missing or extra capture-availability cells")
    detailed, positions = [], []
    for family in FAMILIES:
        for kind in KINDS:
            for position in POSITIONS:
                counts = []
                for gain in (0, 1):
                    value = availability[(family, kind, gain, position)]
                    require(value["assigned"] == 200, "Each gain stratum must retain two assigned cells")
                    counts.append(value)
                    detailed.append(
                        dict(
                            family=family,
                            kind=kind,
                            information_bits=gain,
                            layers="|".join(map(str, layers)),
                            position=position,
                            assigned=value["assigned"],
                            available=value["available"],
                            unavailable=value["assigned"] - value["available"],
                        )
                    )
                assigned, available = (
                    sum(v["assigned"] for v in counts),
                    sum(v["available"] for v in counts),
                )
                positions.append(
                    dict(
                        family=family,
                        kind=kind,
                        position=position,
                        layers="|".join(map(str, layers)),
                        assigned=assigned,
                        available=available,
                        unavailable=assigned - available,
                        unavailable_fraction=(assigned - available) / assigned,
                    )
                )
    return rows, positions, detailed


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save(fig, out, name):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def plot_cells(rows, out, metric):
    is_accuracy = metric == "accuracy"
    count_field, gate = ("correct", 0.9) if is_accuracy else ("valid", 0.95)
    color = BLUE if is_accuracy else GOLD
    fig, axes = plt.subplots(2, 5, figsize=(16.5, 7.8), sharex=True, sharey=True)
    lookup = {(r["family"], r["kind"], r["known"], r["query"]): r for r in rows}
    for family, axis_row in zip(FAMILIES, axes):
        for kind, ax in zip(KINDS, axis_row):
            cell_rows = [lookup[(family, kind, k, q)] for k, q in CELLS]
            values = [r[metric] for r in cell_rows]
            ax.barh(
                range(4),
                values,
                height=0.58,
                color=color,
                edgecolor=INK,
                linewidth=0.45,
                hatch=None if is_accuracy else "//",
                zorder=3,
            )
            ax.axvline(gate, color=INK, linewidth=1.1, linestyle="--", zorder=4)
            for y, r in enumerate(cell_rows):
                value = r[metric]
                ax.text(
                    value - 0.025 if value >= 0.35 else value + 0.025,
                    y,
                    f"{r[count_field]}/100" + ("*" if value < gate else ""),
                    ha="right" if value >= 0.35 else "left",
                    va="center",
                    fontsize=10,
                    fontweight="medium",
                    color="white" if is_accuracy and value >= 0.35 else INK,
                    bbox=dict(facecolor=color if value >= 0.35 else "white", edgecolor="none", pad=1),
                    zorder=5,
                )
            ax.set_title(f"{family.title()}\n{KIND_LABELS[kind]}", loc="left", fontweight="bold", pad=11)
            ax.set_yticks(range(4), [f"K{k} / Q{q}" for k, q in CELLS])
            ax.tick_params(axis="y", labelleft=True, length=0, pad=6)
            ax.set_xlim(0, 1.04)
            ax.set_xticks([0, 0.5, 1])
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.tick_params(axis="x", labelbottom=True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#999999")
            ax.grid(axis="x", color="#E6E6E6", linewidth=0.65, zorder=0)
            ax.set_axisbelow(True)
    axes[0, 0].invert_yaxis()
    title = (
        "Diagnostic accuracy by factorial condition"
        if is_accuracy
        else "Final-answer validity by factorial condition"
    )
    fig.suptitle(title, x=0.045, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.045,
        0.925,
        f"Qwen3-32B · 100 assigned items per cell · dashed line: frozen {gate:.0%} gate · cells meeting this gate: {sum(r[metric] >= gate for r in rows)}/40",
        fontsize=12,
        color="#444444",
    )
    fig.text(
        0.045,
        0.075,
        "K = observed property; Q = queried property. Labels give exact counts out of 100 assigned responses. * Below the displayed gate.",
        fontsize=10,
    )
    fig.text(
        0.045,
        0.043,
        "The same 100 world indices per family recur across all four cells and five tasks; these responses are repeated measures. No confidence intervals are shown.",
        fontsize=10,
        color="#444444",
    )
    status = "met" if all(r["cell_passed"] for r in rows) else "not met"
    definition = (
        "Every cell must satisfy both accuracy and validity thresholds."
        if is_accuracy
        else "Validity requires a closed thinking block and a final answer in the requested format."
    )
    fig.text(
        0.045,
        0.012,
        f"Overall frozen validation gate: {status}. {definition}",
        fontsize=10,
        color="#444444",
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.16, hspace=0.62, wspace=0.60)
    save(fig, out, f"cell-{metric}")


def plot_availability(rows, audit, out):
    fig, axes = plt.subplots(1, 2, figsize=(15.3, 7.1), sharey=True)
    cmap = LinearSegmentedColormap.from_list("missing_blue", ["#F8FAFC", BLUE])
    lookup = {(r["family"], r["kind"], r["position"]): r for r in rows}
    for family, ax in zip(FAMILIES, axes):
        matrix = np.array(
            [[lookup[(family, kind, pos)]["unavailable_fraction"] for pos in POSITIONS] for kind in KINDS]
        )
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for y, kind in enumerate(KINDS):
            for x, position in enumerate(POSITIONS):
                row = lookup[(family, kind, position)]
                rgb = np.array(cmap(matrix[y, x])[:3])
                linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
                text_color = "white" if 1.05 / (luminance + 0.05) >= 4.5 else "black"
                ax.text(
                    x,
                    y,
                    f"{row['unavailable']}/400\n{row['unavailable_fraction']:.1%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )
        ax.set_xticks(range(len(POSITIONS)), POSITION_LABELS)
        ax.set_yticks(range(len(KINDS)), [KIND_LABELS[k] for k in KINDS])
        ax.tick_params(axis="both", length=0, pad=8)
        ax.set_title(family.title(), loc="left", fontweight="bold", pad=14)
        ax.set_xticks(np.arange(-0.5, len(POSITIONS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(KINDS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.3)
        ax.tick_params(which="minor", length=0)
        ax.spines[:].set_visible(False)
    fig.suptitle(
        "Trajectory positions unavailable for capture",
        x=0.04,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.925,
        f"All assigned responses · identical availability across the four captured layers · output cap reached: {audit['cap_hits']:,}/{audit['assigned']:,}",
        fontsize=11,
        color="#444444",
    )
    color_axis = fig.add_axes([0.29, 0.19, 0.42, 0.023])
    colorbar = fig.colorbar(im, cax=color_axis, orientation="horizontal", ticks=[0, 0.25, 0.5, 0.75, 1])
    colorbar.ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    colorbar.set_label("Fraction unavailable", labelpad=5)
    colorbar.outline.set_visible(False)
    fig.text(
        0.04,
        0.075,
        "Each entry gives unavailable responses / 400 (four cells from 100 world indices); the audit verifies common availability across layers.",
        fontsize=10,
    )
    fig.text(
        0.04,
        0.043,
        "A fixed token stage is absent after an earlier completion. Think-end and answer-boundary stages require the corresponding detectable tokens.",
        fontsize=10,
        color="#444444",
    )
    fig.text(
        0.04,
        0.012,
        "Position availability changes the usable sample; this matrix does not show representational strength or a temporal effect.",
        fontsize=10,
        color="#444444",
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.82, bottom=0.34, wspace=0.14)
    save(fig, out, "trajectory-unavailability")


def csv_file(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(run, audit_path, out):
    manifest, audit = read(run / "manifest.json"), read(audit_path)
    rows, positions, detailed = chart_rows(manifest, audit)
    freeze = read(run / "freeze.json")
    require(freeze["manifest_sha256"] == digest(manifest), "Frozen parent binding differs")
    require(not out.exists(), "Choose a new figure directory; existing exports are not overwritten")
    out.mkdir(parents=True)
    style()
    plot_cells(rows, out, "accuracy")
    plot_cells(rows, out, "validity")
    plot_availability(positions, audit, out)
    csv_file(out / "cell-performance.csv", rows)
    csv_file(out / "trajectory-availability.csv", positions)
    csv_file(out / "trajectory-availability-by-gain.csv", detailed)
    csv_file(
        out / "collection-summary.csv",
        [
            dict(
                assigned=audit["assigned"],
                correct=audit["correct"],
                valid=audit["valid"],
                cap_hits=audit["cap_hits"],
                generated_tokens=audit["generated_tokens"],
                multiple_closure_trials=len(audit["multiple_closure_trials"]),
                gate_passed=audit["gates"]["passed"],
            )
        ],
    )
    outputs = {p.name: file_sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_once(
        out / "figure-provenance.json",
        dict(
            schema_version=1,
            source_run=str(run.resolve()),
            audit_path=str(audit_path.resolve()),
            manifest_sha256=digest(manifest),
            manifest_file_sha256=file_sha(run / "manifest.json"),
            audit_sha256=digest(audit),
            audit_file_sha256=file_sha(audit_path),
            frozen_source_sha256=freeze["source_sha256"],
            frozen_archive_sha256=freeze["archive_sha256"],
            script_sha256=file_sha(__file__),
            matplotlib_version=matplotlib.__version__,
            numpy_version=np.__version__,
            output_sha256=outputs,
            inference_scope="descriptive validation/development",
            numerator_source="completed audit gates.cells and partition capture_availability; denominators reconciled to the frozen parent assignments",
            interval_policy="No confidence intervals; repeated responses are not treated as independent worlds",
            availability_policy="All gains and four cells retained; audit capture_availability already verifies common availability across layers. Layers are neither duplicated nor summed.",
            checks=[
                "all 4,000 assignments and all 40 cells retained",
                "100 assigned world indices per cell, shared across cells and tasks within family",
                "correct and valid counts reconcile to audit totals and frozen 90%/95% gates",
                "partition totals reconcile to the parent audit",
                "trajectory capture denominators retained at every planned gain and position, using the audit's common-layer availability",
            ],
            visual_review="pending rendered inspection; recorded separately after export",
        ),
    )
    print(dict(figures=3, formats=["png", "pdf", "svg"], out=str(out.resolve())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.audit, args.out)
