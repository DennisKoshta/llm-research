"""Standalone fixed-grid figures from the held, final coverage-bridge analysis.

Use .plot-venv. No fitting, resampling, outcome selection, rescoring, or admission.
Inputs are the analysis, its exact scored-input JSON and composed audit receipt.
Synthetic previews require --simulated-test and explicitly marked input files.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

ANALYZER_SHA = "7839834735a0c7c3ca43c82eb2ad95e3759253eee42363be2fa12dd8cac41cb4"
PANELS = ("panel_1", "panel_2", "pooled")
PANEL_LABELS = ("Panel 1", "Panel 2", "Equal-panel pooled")
TARGET = "curiosity_negative"
CONTROLS = ("confidence_positive", "completion_positive", "valence_positive", "valence_negative",
            "motivation_negative", "novelty_negative", "task_demand_negative",
            *(f"random_{i:02d}" for i in range(1, 6)))
DIRECTIONS = (TARGET, *CONTROLS)
ARMS = ("baseline", *(f"{s}__{d}" for d in ("zero_sham", *DIRECTIONS) for s in ("record", "full")))
CATEGORIES = ("report", "repeat", "pattern", "stop", "invalid", "capped", "execution_failure", "missing")
FAILURES = ("invalid", "capped", "execution_failure", "missing")
ENDPOINT = "valid_nonreport"
DELIVERY = "redundant_evidence"
LABELS = dict(zip(("baseline", "zero_sham", *DIRECTIONS),
    ("Untouched baseline", "Zero-dose sham", "Curiosity − (target)", "Confidence +", "Completion +",
     "Valence +", "Valence −", "Motivation −", "Novelty −", "Task demand −",
     "Random 01", "Random 02", "Random 03", "Random 04", "Random 05"), strict=True))
INK, GRID, RECORD, FULL, GRAY = "#27313C", "#E4E8EB", "#146B8A", "#B76124", "#727B84"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed(rows, keys):
    result = {tuple(row[k] for k in keys): row for row in rows}
    require(len(result) == len(rows), "Duplicate plotted cell: " + ",".join(keys))
    return result


def response_category(row):
    if row["status"] == "missing":
        return "missing"
    if row["status"] in ("execution_error", "execution_failure"):
        return "execution_failure"
    require(row["status"] == "completed", "Unrecognized response disposition")
    if row["cap_hit"]:
        return "capped"
    return row["selected_service"] if row["valid"] and row["thinking_closed"] else "invalid"


def validate(data, payload, audit, input_bytes, *, simulated=False):
    require(data["assay"] == "curiosity_coverage_bridge_analysis" and data["schema_version"] == 1
        and data["development_only"] is True and data["scientific_admission"] is False
        and data["causal_admission"] is False and data["assigned"] == 8352
        and data["assigned_choices"] == data["assigned_diagnostics"] == 4176
        and data["assigned_worlds"] == 24, "Require the complete assigned parent analysis, never a child")
    require(data["input_sha256"] == digest(payload)
        and data["input_byte_sha256"] == hashlib.sha256(input_bytes).hexdigest()
        and data["assignment_sha256"] == payload["assignment_sha256"] == digest(payload["assignments"])
        and data["outcomes_sha256"] == digest(payload["outcomes"])
        and data["analyzer_byte_sha256"] == ANALYZER_SHA, "Held analyzer/input byte binding differs")
    require(payload["final_audit_sha256"] == digest(audit)
        and payload["source_assignment_sha256"] == audit["assignment_sha256"]
        and audit["outcome_sha256"] == digest(payload["outcomes"])
        and audit["parent_manifest_sha256"] == audit["manifest_sha256"] == payload["manifest_sha256"],
        "Final composed audit/outcome binding differs")
    for key in ("context_audit_byte_sha256", "lifecycle_audit_byte_sha256"):
        require(payload[key] == audit[key] and isinstance(audit[key], str)
            and len(audit[key]) == 64 and set(audit[key]) <= set("0123456789abcdef"), "Missing final audit chain")
    for key in ("complete", "integrity_passed", "checks"):
        require(data[key] == payload[key] == audit[key], "Failed admission flag was changed: " + key)
    require(data["audit_readiness"] == payload["readiness"] == audit["readiness"]
        and type(audit["lifecycle_shutdown_verified"]) is bool
        and (audit["lifecycle_shutdown_verified"] or not audit["integrity_passed"]), "Uncomposed integrity or repaired readiness")
    require(all(v.get("simulated_test") is True for v in (data, payload, audit)) if simulated
            else not any(v.get("simulated_test") for v in (data, payload, audit)), "Simulation designation differs")
    metadata = data["analysis_metadata"]
    require(metadata == payload["analysis_metadata"] and metadata["arms"] == list(ARMS)
        and metadata["controls"] == list(CONTROLS) and metadata["primary_direction"] == TARGET
        and metadata["primary_delivery"] == DELIVERY and metadata["primary_category"] == ENDPOINT
        and metadata["coefficient_aggregation_before_invalidity_bound"] is True
        and metadata["intervals_control_advancement"] is False
        and data["bootstrap"]["draws"] == 10000 and data["bootstrap"]["seed"] == 2026101552
        and data["bootstrap"]["generator"] == "PCG64", "Fixed estimand/bootstrap scope differs")
    require(len(data["outcomes"]) == len(payload["outcomes"]) == 8352
        and indexed(data["outcomes"], ("trial_id",)) == indexed(payload["outcomes"], ("trial_id",)),
        "Analysis did not retain exact audited outcome rows")
    choices = data["choices"]
    require(len(choices) == len({r["trial_id"] for r in choices}) == 4176
        and all(r["category"] == response_category(r) in CATEGORIES for r in choices), "Choice/error categories changed")
    outcome_map = indexed(payload["outcomes"], ("trial_id",))
    require(all({k: v for k, v in r.items() if k != "category"} == outcome_map[(r["trial_id"],)]
        for r in choices), "Plotted choices differ from the bound audited rows")
    rates = indexed(data["denominators"], ("arm", "delivery", "menu_layout", "category", "panel_id"))
    effects = indexed(data["behavior_contrasts"], ("contrast", "delivery", "menu_layout", "category", "panel_id"))
    bounds = indexed(data["decision_bounds"], ("contrast", "delivery", "menu_layout", "category", "panel_id"))
    require(len(rates) == 7047 and len(effects) == 18711 and len(bounds) == 2079
        and len(data["contrast_specs"]) == 77, "The fixed all-direction export inventory differs")
    # Reconcile the displayed assigned-denominator counts without rescoring text.
    for panel in PANELS:
        for menu in (0, 1, "averaged"):
            expected_n = (24 if panel == "pooled" else 12) * (2 if menu == "averaged" else 1)
            for arm in ARMS:
                selected = [r for r in choices if r["arm"] == arm and r["delivery"] == DELIVERY
                    and (panel == "pooled" or r["panel_id"] == panel)
                    and (menu == "averaged" or r["menu_layout"] == menu)]
                require(len(selected) == expected_n, "A primary cell lost an assigned response")
                counts = Counter(r["category"] for r in selected)
                counts[ENDPOINT] = sum(counts[c] for c in ("repeat", "pattern", "stop"))
                for category in (*CATEGORIES, ENDPOINT):
                    row = rates[arm, DELIVERY, menu, category, panel]
                    require(row["assigned_choices"] == expected_n and row["count"] == counts[category]
                        and math.isclose(row["proportion"], counts[category] / expected_n, abs_tol=1e-12),
                        "Plotted rate/count differs from assigned response inventory")
            for spec in data["contrast_specs"]:
                coefficients = spec["coefficients"]
                estimate = sum(c * rates[a, DELIVERY, menu, ENDPOINT, panel]["proportion"] for a, c in coefficients.items())
                penalty = sum(c * sum(rates[a, DELIVERY, menu, k, panel]["proportion"] for k in FAILURES)
                    for a, c in coefficients.items() if c < 0)
                row = effects[spec["contrast"], DELIVERY, menu, ENDPOINT, panel]
                lower = bounds[spec["contrast"], DELIVERY, menu, ENDPOINT, panel]
                require(math.isclose(row["estimate"], estimate, abs_tol=1e-12)
                    and math.isclose(lower["lower_bound"], estimate + penalty, abs_tol=1e-12),
                    "Primary observed contrast or aggregate-coefficient bound differs")
                require(len(row["ci95"]) == 2 and np.isfinite([row["estimate"], *row["ci95"]]).all()
                    and row["ci95"][0] <= row["ci95"][1], "Unavailable/invalid descriptive interval")
    return rates, effects, bounds


def flag(value):
    return "PASS" if value else "FAIL"


def status(data):
    gates = "pass" if all(data["checks"].values()) else "FAIL: " + ", ".join(k.removesuffix("_passed").replace("_", " ")
        for k, v in data["checks"].items() if not v)
    return ("Classification: " + data["decision"]["classification"].replace("_", " ") + ".\n"
        + f"Readiness: analysis {flag(data['task_readiness']['passed'])} / audit {flag(data['audit_readiness']['passed'])}; "
        + f"complete {flag(data['complete'])}; integrity {flag(data['integrity_passed'])}; engineering {gates}.")


def decorate(fig, data, title, subtitle, *, simulated, footer):
    fig.text(.025, .98, "SIMULATED TEST — NOT STUDY RESULTS" if simulated else "CURIOSITY COVERAGE BRIDGE · EXPOSED DEVELOPMENT",
             ha="left", va="top", fontsize=10, color="#A1262F" if simulated else GRAY, weight="bold")
    fig.text(.025, .945, title, ha="left", va="top", fontsize=19, weight="bold", color=INK)
    fig.text(.025, .906, subtitle, ha="left", va="top", fontsize=11, color=INK)
    fig.text(.025, .872, "\n".join(textwrap.fill(line, 160) for line in status(data).splitlines()),
             ha="left", va="top", fontsize=9.5, color=INK, linespacing=1.4)
    fig.text(.025, .025, footer, ha="left", va="bottom", fontsize=9.5, color=INK, linespacing=1.5)


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.tick_params(colors=INK)


def save(fig, out, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=220, facecolor="white",
                    metadata={"Creator": "Fixed-grid coverage bridge renderer"} if suffix != "png" else None)
    plt.close(fig)


def plot_rates(data, rates, out, simulated):
    labels = ("baseline", "zero_sham", *DIRECTIONS)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 9.2), sharex=True, sharey=True)
    for ax, panel, title in zip(axes, PANELS[:2], PANEL_LABELS[:2], strict=True):
        style_axis(ax)
        for i, label in enumerate(labels):
            if label == "baseline":
                row = rates["baseline", DELIVERY, "averaged", ENDPOINT, panel]
                ax.plot(row["proportion"] * 100, i, "D", color=GRAY, markersize=6)
            else:
                for schedule, offset, color, marker in (("record", -.13, RECORD, "o"), ("full", .13, FULL, "s")):
                    row = rates[f"{schedule}__{label}", DELIVERY, "averaged", ENDPOINT, panel]
                    ax.plot(row["proportion"] * 100, i + offset, marker, color=color, markersize=6)
        ax.set_title(title + " · 12 worlds / 24 choices per arm", fontsize=11, pad=12)
        ax.set_xlim(-3, 103)
        ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_yticks(range(len(labels)), [LABELS[k] for k in labels])
        ax.set_ylim(len(labels) - .4, -.6)
        ax.set_xlabel("Observed valid non-report choices", labelpad=10)
    fig.legend(handles=[Line2D([], [], color=RECORD, marker="o", linestyle="", label="Record-prefill + decode"),
        Line2D([], [], color=FULL, marker="s", linestyle="", label="All-attended prefill + decode"),
        Line2D([], [], color=GRAY, marker="D", linestyle="", label="Untouched baseline")],
        loc="upper center", bbox_to_anchor=(.58, .795), frameon=False, ncol=3, fontsize=9)
    fig.subplots_adjust(left=.23, right=.975, top=.72, bottom=.16, wspace=.17)
    decorate(fig, data, "Observed optional-choice rates", "Primary redundant-evidence condition · fixed direction order · both menus retained",
        simulated=simulated, footer="Valid non-report = repeat, pattern or stop, with closed, uncapped, valid output. All assigned choices stay in the denominator.\n"
        "Invalid, capped, missing and execution-failure mass remains in the CSV; it is never counted as non-report. Points are observed rates, not contrasts.\n"
        "The 24 reused worlds are development material. These figures establish neither belief preservation nor an inner-experience claim.")
    save(fig, out, "primary_observed_rates")


def plot_interactions(data, effects, out, simulated):
    families = ((DIRECTIONS, "interaction:", "Coverage change D(direction)"),
                (CONTROLS, "interaction_selectivity:", "Target-minus-control coverage change"))
    selected = [effects[prefix + d, DELIVERY, "averaged", ENDPOINT, panel]
                for directions, prefix, _ in families for d in directions for panel in PANELS]
    maximum = max(25., math.ceil(max(abs(v) * 100 for r in selected for v in (r["estimate"], *r["ci95"])) / 25) * 25)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 12.8), sharex=True)
    for ri, (directions, prefix, heading) in enumerate(families):
        for ci, (panel, title) in enumerate(zip(PANELS, PANEL_LABELS, strict=True)):
            ax = axes[ri, ci]
            style_axis(ax)
            ax.axvline(0, color=GRAY, linewidth=.9)
            for i, d in enumerate(directions):
                row = effects[prefix + d, DELIVERY, "averaged", ENDPOINT, panel]
                color = RECORD if d == TARGET else GRAY
                low, high = [x * 100 for x in row["ci95"]]
                ax.plot([low, high], [i, i], color=color, linewidth=1.3)
                ax.plot(row["estimate"] * 100, i, "D" if d == TARGET else "o", color=color, markersize=5)
            ax.set_title(title + (" · 24 worlds" if panel == "pooled" else " · 12 worlds"), fontsize=11, pad=10)
            ax.set_yticks(range(len(directions)), [LABELS[d] for d in directions] if ci == 0 else [""] * len(directions))
            ax.set_ylim(len(directions) - .3, -.7)
            ax.set_xlim(-maximum * 1.06, maximum * 1.06)
            ax.set_xticks(np.linspace(-maximum, maximum, 5))
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_xlabel("Difference in percentage points", labelpad=8)
            if ci == 0:
                ax.text(-.66, 1.12, heading, transform=ax.transAxes, fontsize=12, weight="bold", color=INK)
    fig.subplots_adjust(left=.18, right=.985, top=.735, bottom=.16, hspace=.40, wspace=.16)
    decorate(fig, data, "Coverage interactions across every fixed control",
        "Primary redundant evidence · both menus averaged within world · fixed order, no effect-based selection",
        simulated=simulated, footer="D(direction) = (full − full-zero sham) − (record − record-zero sham). Bottom row = D(curiosity −) − D(control).\n"
        "Bars are the held 95% percentile bootstrap intervals: 10,000 whole-world draws within panel; pooled draws equally weight the two panels.\n"
        "Intervals are descriptive only, including degenerate intervals. Advancement uses the frozen panel/menu effect rules and conservative invalidity bounds,\n"
        "not interval exclusion of zero. All bounds, both menu checks and the 12 individual control comparisons remain in the accompanying CSV files.")
    save(fig, out, "primary_coverage_interactions")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def render(analysis_path, input_path, audit_path, out, *, simulated=False):
    paths = dict(analysis=Path(analysis_path).resolve(), scored_input=Path(input_path).resolve(), final_audit=Path(audit_path).resolve())
    raw = {k: p.read_bytes() for k, p in paths.items()}
    data, payload, final_audit = (json.loads(raw[k]) for k in ("analysis", "scored_input", "final_audit"))
    rates, effects, _ = validate(data, payload, final_audit, raw["scored_input"], simulated=simulated)
    out = Path(out).resolve()
    require(not out.exists(), "Figure output directory is immutable; choose a new destination")
    require(not simulated or "reports" not in out.parts, "Keep simulated previews outside actual report directories")
    out.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelcolor": INK,
        "text.color": INK, "svg.fonttype": "none", "pdf.fonttype": 42})
    plot_rates(data, rates, out, simulated)
    plot_interactions(data, effects, out, simulated)
    exports = {
        "primary_rates.csv": [r for r in data["denominators"] if r["delivery"] == DELIVERY],
        "primary_contrasts.csv": [r for r in data["behavior_contrasts"] if r["delivery"] == DELIVERY],
        "primary_invalidity_bounds.csv": [r for r in data["decision_bounds"] if r["delivery"] == DELIVERY],
        "readiness.csv": [dict(scope=k, **r) for k in ("diagnostic_cells", "choice_cells", "arm_diagnostics")
                          for r in data["task_readiness"][k]],
    }
    for name, rows in exports.items():
        write_csv(out / name, rows)
    require(all(p.read_bytes() == raw[k] for k, p in paths.items()), "Source bytes changed during rendering")
    provenance = dict(schema_version=1, simulated_test=simulated, renderer_byte_sha256=sha(__file__),
        held_analyzer_byte_sha256=ANALYZER_SHA,
        inputs={k: dict(path=str(paths[k]), byte_sha256=hashlib.sha256(v).hexdigest()) for k, v in raw.items()},
        source_manifest_sha256=final_audit["manifest_sha256"], classification=data["decision"]["classification"],
        completeness=data["complete"], integrity_passed=data["integrity_passed"],
        task_readiness=data["task_readiness"]["passed"], audit_readiness=data["audit_readiness"]["passed"],
        checks=data["checks"], fixed_display_order=list(LABELS),
        exported_rows={k: len(v) for k, v in exports.items()},
        validation=dict(input_and_audit_chain=True, exact_8352_outcomes=True,
            primary_assigned_counts_reconciled=True, all77_primary_contrasts_reconciled=True,
            aggregate_coefficient_bounds_reconciled=True, bootstrap_recomputed=False,
            raw_execution_reaudited=False, visual_inspection_required_after_actual_render=True),
        outputs={p.name: dict(byte_sha256=sha(p), byte_count=p.stat().st_size) for p in sorted(out.iterdir())},
        scope="Fixed redundant-evidence figures only; all other deliveries remain in the source analysis. No selection, fitting, recalibration or admission.")
    with (out / "provenance.json").open("x") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("analysis", "input", "audit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--simulated-test", action="store_true")
    args = parser.parse_args()
    result = render(args.analysis, args.input, args.audit, args.out, simulated=args.simulated_test)
    print(json.dumps({k: result[k] for k in ("simulated_test", "classification", "exported_rows")}))
