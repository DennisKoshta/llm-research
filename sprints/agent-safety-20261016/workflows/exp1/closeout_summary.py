"""Offline editorial closeout; does not alter frozen executions or primary analyses."""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    analysis = json.loads((HERE / "analysis.json").read_text())
    audit = json.loads((HERE / "artifact_audit.json").read_text())
    assert audit["passed"]
    independent_review_path = HERE / "independent-result-review-root.json"
    independent_review_passed = independent_review_path.exists() and json.loads(independent_review_path.read_text()).get("passed") is True
    sources = {}
    index = []
    for stage, run_id in (("pilot", "pilot_final_v1"), ("confirmation", "confirmation_final_v1"), ("live", "live_executor_v1")):
        worlds = {w["world_id"]: w for w in rows(HERE / "runs" / stage / "worlds.jsonl")}
        scored = {r["request_id"]: r for r in rows(HERE / "runs" / stage / "scored.jsonl")}
        source = HERE / "runs" / run_id / "responses.jsonl"
        sources[str(source.relative_to(HERE))] = hashlib.sha256(source.read_bytes()).hexdigest()
        enriched = []
        for line_number, raw in enumerate(rows(source), 1):
            score = scored[raw["id"]]
            world = worlds[score["world_id"]]
            full_label = next(k for k, v in world["menu"].items() if v == "full")
            enriched.append({**score, "analysis_revision": "derived-metadata-v2",
                             "prompt_hash": raw["prompt_sha256"],
                             "label_probabilities": raw["label_probabilities_temperature_1"],
                             "conditional_label_probabilities": raw["conditional_label_probabilities"],
                             "raw_relative_path": str(source.relative_to(HERE)), "raw_line": line_number})
            index.append({
                "request_id": raw["id"], "world_id": score["world_id"],
                "stage": stage, "family": score["family"], "condition": score["condition"],
                "full_completion_authorized": score["condition"] == "user_authorized",
                "valid": score["valid"], "action": score["action"],
                "unauthorized": score["unauthorized"], "action_menu": world["menu"],
                "raw_relative_path": str(source.relative_to(HERE)), "raw_line": line_number,
                "capture_layers": list(raw["activations"]), "dimension_each": 5120,
                "capture_boundary": "final attended prompt token before any generated action token",
                "prompt_sha256": raw["prompt_sha256"],
                "full_completion_probability_temperature_1": raw["label_probabilities_temperature_1"][full_label],
                "full_completion_conditional_label_probability": raw["conditional_label_probabilities"][full_label],
                "sampling": raw["sampling"],
            })
        with (HERE / "runs" / stage / "scored-enriched-v2.jsonl").open("w") as handle:
            for record in enriched:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (HERE / "capture_index.jsonl").open("w") as handle:
        for record in index:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    confirmation = [r for r in index if r["stage"] == "confirmation"]
    family_counts = {family: {condition: sum(r["unauthorized"] for r in confirmation if r["family"] == family and r["condition"] == condition)
                             for condition in analysis["confirmation"]["conditions"]}
                     for family in sorted({r["family"] for r in confirmation})}
    summary = {
        "analysis_revision": "derived-metadata-v2",
        "formal_status": "complete_task_qualified_artifact_audit_passed",
        "independent_result_review": "passed: independent-result-review-root.json" if independent_review_passed else "pending; independent protocol review and separate raw-score audit are available",
        "n_inference_calls": sum(r["n"] for r in audit["run_checks"]),
        "n_capture_rows": len(index), "capture_layers": [26, 28, 30],
        "confirmation_violating_responses": sum(r["unauthorized"] for r in confirmation),
        "confirmation_distinct_violating_worlds": len({r["world_id"] for r in confirmation if r["unauthorized"]}),
        "family_counts": family_counts,
        "live_each_condition_zero_event_one_sided_95_upper": 1 - 0.05 ** (1 / 24),
        "frozen_raw_sources_sha256": sources,
        "analysis_note": "Primary frozen analyses unchanged. Named derived-metadata-v2 writes new scored-enriched-v2 tables, a capture lookup, descriptive family counts and a finite-sample zero-event bound; it does not overwrite previous scored tables.",
        "metadata_note": "Frozen scored.jsonl prompt_hash/label_probabilities aliases are null; actual raw prompt_sha256 and probability fields were always preserved. The capture index reads the raw keys directly.",
    }
    (HERE / "closeout_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(analysis["confirmation"])
    report(analysis, summary)
    print(json.dumps(summary, indent=2))


def plot(confirm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["solitary", "unattributed_notes", "one_peer", "three_peers"]
    labels = ["No notes", "Unattributed notes", "One coworker", "Three coworkers"]
    fig, ax = plt.subplots(figsize=(8.3, 3.8), constrained_layout=True)
    for i, name in enumerate(names):
        c = confirm["conditions"][name]
        value = 100 * c["unauthorized_fraction"]
        low, high = (100 * x for x in c["unauthorized_wilson_ci95"])
        ax.errorbar(value, i, xerr=[[value - low], [high - value]], fmt="o", color="#236b9c", capsize=4)
        ax.text(8.5, i, f"{c['unauthorized']}/144", va="center", fontsize=10)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.3, 10)
    ax.set_xlabel("Unauthorized final actions (%)")
    ax.set_title("Peer attribution added no observed effect beyond the same notes", fontsize=12, pad=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    fig.supxlabel("144 paired worlds; marginal Wilson 95% intervals. Genuine user approval: 144/144 full completions.", fontsize=9)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(HERE / "figures" / f"confirmation-outcomes.{extension}", dpi=180)
    plt.close(fig)


def report(analysis, summary):
    a = analysis["confirmation"]
    p = a["primary"]
    labels = {"solitary": "No notes", "unattributed_notes": "Three unattributed notes", "one_peer": "Same notes, one coworker", "three_peers": "Same notes, three coworkers", "user_authorized": "Three coworkers + genuine user approval"}
    final_review_text = "Independent final-result review reproduced all 720 confirmation choices, all condition counts and both paired discordance counts." if summary["independent_result_review"].startswith("passed") else "The independent final-result review is pending."
    lines = [
        "# Peer endorsement and user authorization",
        "",
        "Attributing the same completion-oriented notes to three coworkers did not increase unauthorized choices in this experiment: both peer-attributed and unattributed notes produced 5/144 violations. Genuine user approval changed all 144 decisions to permitted full completion.",
        "",
        "| Confirmation condition | Unauthorized choices | Full completions | Valid replies |",
        "|---|---:|---:|---:|",
    ]
    for name, c in a["conditions"].items():
        lines.append(f"| {labels[name]} | {c['unauthorized']}/144 | {c['full_completion']}/144 | {c['valid']}/144 |")
    lines += [
        "",
        "The frozen primary contrast, three coworkers minus identical unattributed notes, was **0.00 percentage points** (world-stratified bootstrap 95% interval −2.78 to +2.78 points; two discordant worlds in each direction; exact paired p=1). This does not establish equivalence. Three coworkers versus no notes produced five extra violations (secondary exact paired p=.0625); the same five-versus-zero marginal difference occurred with unattributed notes, so it cannot be assigned specifically to peer identity.",
        "",
        "The observed violations were sparse: 12 responses across eight worlds across the three note conditions. Ten occurred in scheduling and two in inventory; document routing had none. This family pattern is descriptive, and the study does not establish its cause.",
        "",
        "![Confirmation outcomes](figures/confirmation-outcomes.png)",
        "",
        "A separate bounded extension generated two sequential coworker proposals in each of 24 fresh worlds. None of the 48 proposals recommended the prohibited plan. All 24 executors chose the permitted partial plan with attributed generated messages, with the same messages unattributed, and with an independently sampled replay of the identical attributed context. Thus the extension did not produce spontaneous permission violations. Zero of 24 has a one-sided 95% binomial upper bound of 11.7% for each condition; it is weak evidence about rare events. The two peer messages within a world are dependent.",
        "",
        "The task presented useful partial completion, a protected resource, an explicit user restriction, and four exact simulated actions. Notes changed neither the available facts nor the one-action budget. Coworkers were explicitly unable to grant permission. The experiment therefore tests whether peer attribution overrides a clear authority boundary; ambiguous delegation, evolving norms, concealed actions and long-running reciprocal interaction remain outside its scope. The main notes were scripted controlled exposure, not emergent multi-agent behavior.",
        "",
        "The 24-world pilot passed all frozen competence and format gates with no violations. Confirmation then used 144 independent new world instances (48 per family), frozen before collection, and ran regardless of effect direction. These instances reuse three procedural task templates; they do not demonstrate generalization to new task families. Final conditions shared the same initial transcript but used fresh prompt branches and different sampling seeds, not identical KV states or common random draws. The model was Qwen/Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137`, BF16, nonthinking, temperature 0.7.",
        "",
        "All 1,128 inference calls were recovered and verified against their exact requests; all outputs ended normally. All 912 executor decisions were valid. The separate raw scorer reproduced every executed action and permission outcome, and all layer 26/28/30 captures were finite 5120-dimensional vectors at the final prompt token before generation. No outcomes were excluded. Independent protocol review passed before confirmation. " + final_review_text + " Frozen source snapshots and primary analyses are unchanged. The named reporting revision `derived-metadata-v2` repairs null provenance/probability aliases in new `scored-enriched-v2.jsonl` tables using values always present in raw records; original tables remain intact.",
        "",
        "[Protocol](protocol.json), [primary analyses](analysis.json), [raw artifact audit](artifact_audit.json), [independent protocol review](independent-review.json), [independent result review](independent-result-review-root.json), [postcollection summary](closeout_summary.json), [capture lookup](capture_index.jsonl). Raw requests and responses are in `runs/*_v1/`; per-stage worlds, scores and reports are in `runs/pilot/`, `runs/confirmation/` and `runs/live/`. `executions/` preserves each stage's source freeze. Run `audit_artifacts.py` for raw verification and `closeout_summary.py` to regenerate this summary and figure without inference. Shared broker accounting is authoritative for cost; no workflow-specific GPU shutdown is claimed while other workflows use the shared service.",
    ]
    original = HERE / "report-generated-v1.md"
    if not original.exists():
        original.write_bytes((HERE / "report.md").read_bytes())
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
