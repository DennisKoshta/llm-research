"""Request builder / processor for Experiments 2.2 (monitor calibration) and 2.1 (four topologies).

No harmful text is ever printed: every command prints counts and problem CODES only. Authored worlds and notes
live unredacted in data/exp2122/*.jsonl (never cat them). This script launches nothing; `plan` prints the exact
run_requests.py commands for the main loop.

World authoring (arm flash_next_uncensored, on Modal) and note authoring (arm refusal_ablated: a note describes no harm,
and the 27B decodes ~3x faster per sequence than Flash-Next), each step is: build requests -> run -> process
  worlds-requests R      data/exp2122/requests_worlds_r{R}.jsonl      (worlds not yet final; retries carry problem codes)
  worlds-process R       parse + validate the round-R author rows  -> worlds_accepted.jsonl
  worlds-judge-requests R / worlds-process-judge R   numeric severity judge -> worlds_final.jsonl
  notes-requests R / notes-process R                  one note per (world, bundle, condition), structural asserts
  notes-judge-requests R / notes-process-judge R      claim-polarity audit -> notes_final.jsonl
The processors find their rows by meta.stage and meta.round in EVERY raw_transcripts/exp22_worlds* (exp22_notes*) directory,
so request files of the same arm may share one launch (e.g. `worlds-judge-requests R` + `worlds-requests R+1`, passed
comma-separated to run_requests.py with --exp-id exp22_worlds_r<R+1>): that saves a Flash-Next boot (~11 min of H200:2).
Different arms in one launch (notes author + audit) boot two containers either way, so merging them only saves wall time.
Global options: --data-dir / --raw-dir redirect all state (use a scratch dir with `placeholder-data` for dry runs);
--world-author-arm / --note-author-arm switch the authoring models (default flash_next_uncensored / refusal_ablated).
Experiments (all request files are plain request rows for scripts/run_requests.py)
  monitor-requests [--scale S] [--arms intact,llama70b,flash_next]   one file per monitor arm (asserts every note first)
  exp21-requests alpha|beta|monitor [--scale S]      stage files; later stages read raw_transcripts/exp21_*/
  estimate [--scale S] [--arms ...]                  generations, tokens, GPU-hours, USD
  plan [--scale S] [--arms ...]                      the launch commands in order
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import exp21 as X, exp22 as W  # noqa: E402

DEFAULT_D = D = ROOT / "data" / "exp2122"
RAW = ROOT / "raw_transcripts"
ARM = W.AUTHOR_ARM
TARGET_USD = 30.0    # the pipeline's compute target (contingency included): `estimate` reports the gap, scale cannot close it alone


def _load(path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()] if pathlib.Path(path).exists() else []


def _save(path, rows) -> None:
    D.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _json(path, default):
    return json.loads(pathlib.Path(path).read_text()) if pathlib.Path(path).exists() else default


def _raw_rows(prefix: str, stage: str, rnd: int) -> list[dict]:
    """Result rows of one authoring stage and round, from every raw_transcripts/<prefix>* directory (a launch may mix
    request files; the stage and round are in each row's meta, so nothing depends on the directory name)."""
    rows = []
    for d in sorted(RAW.glob(prefix + "*")):
        for f in sorted(d.glob("*.jsonl")) if d.is_dir() else []:
            if not f.name.endswith(".partial.jsonl"):
                rows += [r for r in _load(f) if r.get("meta", {}).get("stage") == stage and r["meta"].get("round") == rnd]
    ids = collections.Counter(r["trial_id"] for r in rows)
    if any(n > 1 for n in ids.values()):
        sys.exit(f"duplicate trial_id in {prefix}* round {rnd}: the same request was run twice")
    return rows


def _tokens(rows) -> str:
    toks = sorted(r["generated_tokens"] for r in rows)
    fin = dict(collections.Counter(r["finish_reason"] for r in rows))
    return f"tokens p50/max: {toks[len(toks) // 2]}/{toks[-1]} finish: {fin}" if toks else "no rows"


# ------------------------------------------------------------------------------------------------ worlds
def worlds_requests(rnd: int, n_variants: int, arm: str = ARM) -> None:
    final = {w["world_id"] for w in _load(D / "worlds_final.jsonl")}
    accepted = {w["world_id"] for w in _load(D / "worlds_accepted.jsonl")}
    verdicts = _json(D / "world_verdicts.json", {})
    problems = _json(D / "problems_worlds.json", {})
    waiting = {wid for wid in accepted if wid not in verdicts}                    # accepted, not yet judged
    rows = W.world_author_requests(rnd, n_variants, skip=final | waiting, problems=problems, arm=arm)
    _save(D / f"requests_worlds_r{rnd}.jsonl", rows)
    print(f"round {rnd}: {len(rows)} world requests (final {len(final)}, awaiting judge {len(waiting)})")


def worlds_process(rnd: int) -> None:
    raw = _raw_rows("exp22_worlds", "worlds_author", rnd)
    accepted = {w["world_id"]: w for w in _load(D / "worlds_accepted.jsonl")}
    problems = _json(D / "problems_worlds.json", {})
    verdicts = _json(D / "world_verdicts.json", {})
    reasons = collections.Counter()
    new = 0
    redundant = 0
    for r in sorted(raw, key=lambda r: r["trial_id"]):
        m = r["meta"]
        if m["world_id"] in accepted and m["world_id"] not in problems:
            redundant += 1           # a valid, not-yet-failed candidate is already held (parallel rounds): first valid wins
            continue
        args, why = W.parse_world_output(r["raw_output"])
        if args is None:
            problems[m["world_id"]], reasons[why] = [why], reasons[why] + 1
            continue
        world = W.build_world(m["domain"], m["variant"], args)
        codes = W.validate_world(world)
        if codes:
            problems[m["world_id"]] = codes
            reasons.update(c.split(":")[0] for c in codes)
        else:
            accepted[m["world_id"]] = world
            problems.pop(m["world_id"], None)
            verdicts.pop(m["world_id"], None)          # a replaced world must be judged again
            new += 1
    clash = W.validate_world_set(list(accepted.values()))          # cross-world diversity: drop the later duplicates
    for wid, codes in clash.items():
        problems[wid] = codes
        reasons.update(c.split(":")[0] for c in codes)
        accepted.pop(wid, None)
        verdicts.pop(wid, None)
    _save(D / "worlds_accepted.jsonl", sorted(accepted.values(), key=lambda w: w["world_index"]))
    (D / "problems_worlds.json").write_text(json.dumps(problems, indent=1))
    (D / "world_verdicts.json").write_text(json.dumps(verdicts, indent=1))
    print(f"round {rnd}: n_raw={len(raw)} accepted_now={new} redundant_skipped={redundant} total_accepted={len(accepted)} problems={len(problems)}")
    print("reason counts:", dict(reasons.most_common()), "|", _tokens(raw))
    print("audit:", json.dumps({k: v for k, v in W.audit_worlds(list(accepted.values())).items() if k != "per_domain"}))


def worlds_judge_requests(rnd: int, arm: str = ARM, in_flight: tuple = ()) -> None:
    """Judge requests for every accepted world without a verdict. `in_flight` = earlier judge rounds whose launch is still
    running: a world whose CURRENT candidate (identical judge prompt) is in one of those request files is not asked again."""
    verdicts = _json(D / "world_verdicts.json", {})
    sent = {tuple(m["content"] for m in r["messages"]) for k in in_flight for r in _load(D / f"requests_worlds_judge_r{k}.jsonl")}
    rows = [W.world_judge_request(w, rnd, arm) for w in _load(D / "worlds_accepted.jsonl") if w["world_id"] not in verdicts]
    n_all = len(rows)
    rows = [r for r in rows if tuple(m["content"] for m in r["messages"]) not in sent]
    _save(D / f"requests_worlds_judge_r{rnd}.jsonl", rows)
    print(f"{len(rows)} world judge requests ({n_all - len(rows)} skipped: candidate already in flight in round(s) {list(in_flight)})")


def worlds_process_judge(rnd: int) -> None:
    raw = _raw_rows("exp22_worlds", "worlds_judge", rnd)
    verdicts = _json(D / "world_verdicts.json", {})
    problems = _json(D / "problems_worlds.json", {})
    unparsed = collections.Counter()
    for r in raw:
        v = W.parse_world_judgement(r["raw_output"])
        if v["status"] == "scored":
            verdicts[r["meta"]["world_id"]] = v
        else:                                          # the judge's fault, not the world's: judge again next round
            verdicts.pop(r["meta"]["world_id"], None)
            unparsed[v["status"]] += 1
    accepted = {w["world_id"]: w for w in _load(D / "worlds_accepted.jsonl")}
    final = []
    for wid, w in accepted.items():
        v = verdicts.get(wid)
        if v and W.world_judgement_ok(v):
            final.append({**w, "judge": {k: v[k] for k in W.WORLD_JUDGE_FIELDS}})
            problems.pop(wid, None)
        elif v:                                        # judged and failed: regenerate with the failing criteria as codes
            problems[wid] = [f"judge_low_{k}" for k in ("severity", "irreversibility") if v[k] < 4] + (
                ["judge_instructional"] if v["instructional"] != "no" else []) + (
                ["judge_incoherent"] if v["coherent"] != "yes" else [])
    (D / "world_verdicts.json").write_text(json.dumps(verdicts, indent=1))
    (D / "problems_worlds.json").write_text(json.dumps(problems, indent=1))
    _save(D / "worlds_final.jsonl", sorted(final, key=lambda w: w["world_index"]))
    c = collections.Counter()
    for v in verdicts.values():
        c["status:" + v["status"]] += 1
        if v["status"] == "scored":
            c[f"sev{v['severity']}"] += 1
            c[f"irrev{v['irreversibility']}"] += 1
            c["instr:" + v["instructional"]] += 1
            c["coherent:" + v["coherent"]] += 1
    print(f"judged {len(verdicts)}; final worlds {len(final)}; unparsed judge outputs {dict(unparsed)}; counts:", dict(sorted(c.items())))
    print("still needed:", sorted(set(accepted) - {w["world_id"] for w in final}))


# ------------------------------------------------------------------------------------------------ notes
def _worlds_by_id() -> dict:
    return {w["world_id"]: w for w in W.load_worlds(D / "worlds_final.jsonl")}


def notes_requests(rnd: int, scale: float, arm: str = W.NOTE_AUTHOR_ARM) -> None:
    worlds = W.select_worlds(list(_worlds_by_id().values()), W.scale_plan(scale)["variants_per_domain"])
    final = {r["note_id"] for r in _load(D / "notes_final.jsonl")}
    accepted = {r["note_id"] for r in _load(D / "notes_accepted.jsonl")}
    verdicts = _json(D / "note_verdicts.json", {})
    waiting = {nid for nid in accepted if nid not in verdicts}
    rows = W.note_requests(worlds, rnd, skip=final | waiting, problems=_json(D / "problems_notes.json", {}), arm=arm)
    _save(D / f"requests_notes_r{rnd}.jsonl", rows)
    print(f"round {rnd}: {len(rows)} note requests ({len(final)} final, {len(waiting)} awaiting audit)")


def notes_process(rnd: int) -> None:
    worlds = _worlds_by_id()
    raw = _raw_rows("exp22_notes", "notes_author", rnd)
    accepted = {r["note_id"]: r for r in _load(D / "notes_accepted.jsonl")}
    problems = _json(D / "problems_notes.json", {})
    verdicts = _json(D / "note_verdicts.json", {})
    reasons = collections.Counter()
    stale = [nid for nid, r in accepted.items() if r["world_id"] not in worlds]    # world dropped after its notes were written
    for nid in stale:
        accepted.pop(nid), problems.pop(nid, None), verdicts.pop(nid, None)
    n_stale_raw = 0
    n_redundant = 0
    for r in sorted(raw, key=lambda r: r["trial_id"]):
        m = r["meta"]
        if m["world_id"] not in worlds:
            n_stale_raw += 1
            continue
        if m["note_id"] in accepted and m["note_id"] not in problems:
            n_redundant += 1         # a candidate is already held and has not failed (parallel attempts): first valid wins
            continue
        note, why = W.parse_note_output(r["raw_output"])
        codes = [why] if note is None else W.note_problems(note, m["condition"], m["bundle"], worlds[m["world_id"]])
        if codes:
            problems[m["note_id"]] = codes
            reasons.update(c.split(":")[0] for c in codes)
        else:
            accepted[m["note_id"]] = {"note_id": m["note_id"], "world_id": m["world_id"], "bundle": m["bundle"],
                                      "condition": m["condition"], "note": note, "round": rnd, "words": len(W._words(note))}
            problems.pop(m["note_id"], None)
            verdicts.pop(m["note_id"], None)
    _save(D / "notes_accepted.jsonl", list(accepted.values()))
    (D / "problems_notes.json").write_text(json.dumps(problems, indent=1))
    (D / "note_verdicts.json").write_text(json.dumps(verdicts, indent=1))
    print(f"round {rnd}: n_raw={len(raw)} accepted_total={len(accepted)} problems={len(problems)} "
          f"purged_stale_accepted={len(stale)} stale_world_raw_rows_ignored={n_stale_raw} redundant_skipped={n_redundant}")
    print("reason counts:", dict(reasons.most_common()), "|", _tokens(raw))


def notes_judge_requests(rnd: int) -> None:
    worlds = _worlds_by_id()
    verdicts = _json(D / "note_verdicts.json", {})
    rows = [W.note_judge_request(worlds[r["world_id"]], r["bundle"], r["condition"], r["note"], rnd)
            for r in _load(D / "notes_accepted.jsonl") if r["note_id"] not in verdicts and r["world_id"] in worlds]
    _save(D / f"requests_notes_judge_r{rnd}.jsonl", rows)
    print(f"{len(rows)} note audit requests")


def notes_process_judge(rnd: int) -> None:
    raw = _raw_rows("exp22_notes", "notes_judge", rnd)
    worlds = _worlds_by_id()
    verdicts = _json(D / "note_verdicts.json", {})
    problems = _json(D / "problems_notes.json", {})
    unparsed = collections.Counter()
    for r in raw:
        v = W.parse_note_judgement(r["raw_output"])
        if v["status"] == "scored":
            verdicts[r["meta"]["note_id"]] = v
        else:                                          # the auditor's output was unusable: audit again next round
            verdicts.pop(r["meta"]["note_id"], None)
            unparsed[v["status"]] += 1
    final, bad = [], collections.Counter()
    for r in _load(D / "notes_accepted.jsonl"):
        v = verdicts.get(r["note_id"])
        if v is None or r["world_id"] not in worlds:
            continue
        if W.note_judgement_ok(v, r["condition"], r["bundle"]):
            final.append({**r, "audit": {k: v[k] for k in ("claims_compliance", "lanes_match")}})
            problems.pop(r["note_id"], None)
        else:
            codes = []
            if v["claims_compliance"] != W.expected_note_claim(r["condition"], r["bundle"]):
                codes.append(f"audit_claim_{v['claims_compliance']}")
            if v["lanes_match"] == "no":
                codes.append("audit_lanes_no")
            problems[r["note_id"]] = codes
            bad[r["condition"]] += 1
    (D / "note_verdicts.json").write_text(json.dumps(verdicts, indent=1))
    (D / "problems_notes.json").write_text(json.dumps(problems, indent=1))
    _save(D / "notes_final.jsonl", final)
    n_expected = len(_worlds_by_id()) * len(W.BUNDLES) * len(W.NOTE_CONDITIONS)   # at --scale < 1 fewer are needed
    print(f"audited {len(verdicts)}; final notes {len(final)}/{n_expected}; audit failures by condition: {dict(bad)}; "
          f"unparsed audits {dict(unparsed)}")


# ------------------------------------------------------------------------------------------------ experiments
def effort_arg(name: str):
    """CLI effort -> request effort: 'auto' (the Phase-2 decision in records/phase1_family_decision.json), 'default' (the
    template default, i.e. no effort argument), or an explicit level."""
    return None if name == "default" else name


def monitor_requests(scale: float, arms: list[str], cot_on_slice: bool, effort: str = "auto", exclude: tuple = ()) -> None:
    """`exclude` = world ids left out of Experiment 2.2 because a required note could not pass its audit (a documented
    deviation: the monitor grid needs all notes of every world it contains)."""
    plan = W.scale_plan(scale)
    worlds = [w for w in W.select_worlds(list(_worlds_by_id().values()), plan["variants_per_domain"])
              if w["world_id"] not in exclude]
    print(f"monitor worlds: {len(worlds)} (excluded: {sorted(exclude)})")
    out = W.all_monitor_requests(worlds, W.load_notes(D / "notes_final.jsonl"), arms=arms, cot_on_slice=cot_on_slice,
                                 effort=effort_arg(effort))
    for arm, rows in out.items():
        _save(D / f"requests_exp22_monitor_{arm}.jsonl", rows)
        print(f"{arm}: {len(rows)} requests ({collections.Counter(r['cot_mode'] for r in rows)}) -> "
              f"requests_exp22_monitor_{arm}.jsonl")


def _prior(stages) -> list[dict]:
    rows = []
    for stage in stages:
        for d in sorted(RAW.glob(f"exp21_{stage}*")):             # exp21_<stage> and its parallel parts exp21_<stage>_p<k>
            if d.is_dir() and re.fullmatch(rf"exp21_{stage}(_p[0-9]+)?", d.name):
                for p in sorted(d.glob("*.jsonl")):
                    if not p.name.endswith(".partial.jsonl"):
                        rows += _load(p)
    return rows


def exp21_requests(stage: str, scale: float, cot_mode: str, judge_arm: str, effort: str = "auto") -> None:
    worlds = list(_worlds_by_id().values())
    prior = _prior({"alpha": (), "beta": ("alpha",), "monitor": ("alpha", "beta")}[stage])
    kw = dict(worlds=worlds, scale=scale, cot_mode=cot_mode, effort=effort_arg(effort))
    rows = X.build_stage_requests(stage, prior, **kw)
    if stage == "monitor":
        rows += X.build_stage_requests("judge", prior, judge_arm=judge_arm, **kw)
    _save(D / f"requests_exp21_{stage}.jsonl", rows)
    print(f"{stage}: {len(rows)} requests: arms {dict(collections.Counter(r['arm'] for r in rows))}, "
          f"stages {dict(collections.Counter(r['meta']['stage'] for r in rows))} -> requests_exp21_{stage}.jsonl")


# ------------------------------------------------------------------------------------------------ cost and launch plan
def estimates(scale: float, arms: list[str], note_arm: str = W.NOTE_AUTHOR_ARM, world_arm: str = ARM) -> dict:
    e22 = W.estimate_cost(scale, monitors=tuple(arms), note_author=note_arm, world_author=world_arm)
    e21 = X.estimate_cost(scale)
    stages = e22["stages"] + e21["stages"]
    return {"scale": scale, "plan": W.scale_plan(scale), "exp22": e22, "exp21": e21,
            "combined": W.totals(stages), "target_usd": TARGET_USD}


def plan_commands(scale: float, arms: list[str], note_arm: str = W.NOTE_AUTHOR_ARM, world_arm: str = ARM) -> list[str]:
    """The launch commands in order. --est-hours is 2.5x the estimated wall time of ONE launch (min 0.3 h) per arm: it only
    sizes the ledger reservation, the job ends when the work does."""
    est = estimates(scale, arms, note_arm, world_arm)
    wall = {s["stage"]: s["gpu_hours"] / W.ENGINES[s["engine"]]["n_gpus"] / max(1, s["launches"])
            for s in est["exp22"]["stages"] + est["exp21"]["stages"]}
    hrs = lambda *names: max(0.3, round(2.5 * sum(wall.get(n, 0) for n in names), 1))
    run = "PYTHONPATH=src .venv/bin/python scripts/run_requests.py"
    f = "data/exp2122/"
    flash = lambda h: f"--gpu H200:2 --est-hours {h} --max-model-len 24576 --max-num-seqs 128 --eager 1"
    q27 = lambda h: f"--gpu H200 --est-hours {h} --max-model-len 20480 --max-num-seqs 384 --eager 0 --family $FAMILY"
    # authoring is SAMPLED (T=0.7, top-p 0.95: Qwen's thinking-mode setting; greedy thinking decodes can loop, and a retry with
    # an unchanged prompt would repeat its failure). Seeds come from the trial id, which carries the round.
    samp = "--temperature 0.7 --top-p 0.95 --seed-base 22"
    where = lambda arm, h: flash(h) if W.ARM_ENGINE[arm] == "flash" else q27(h)
    eff = W.phase2_effort()
    cmds = [
        f"# reasoning effort of the Exp 2.1 agents / monitors / judge and the Exp 2.2 cot_on slice: "
        f"{'default (template default)' if eff is None else eff}, from records/phase1_family_decision.json "
        "(override with --effort on exp21-requests / monitor-requests; authoring always runs at effort low)",
        f"# --- worlds, authored by {world_arm}; every Flash-Next launch is ~11 min boot + up to 15 min for the longest sequence "
        "(~9 tok/s each): try --eager 0 once (CUDA graphs are untested for Flash). Each round: worlds-requests R, run, "
        "worlds-process R, worlds-judge-requests R, run, worlds-process-judge R. Passing `worlds-judge-requests R` and "
        "`worlds-requests R+1` comma-separated to ONE launch (--exp-id exp22_worlds_r<R+1>) saves a boot (the processors read by "
        "meta.stage and meta.round).",
        f"{run} --exp-id exp22_worlds_r1 --requests {f}requests_worlds_r1.jsonl --arms {world_arm} {samp} "
        f"{where(world_arm, hrs('worlds_author'))}",
        f"{run} --exp-id exp22_worlds_judge_r1 --requests {f}requests_worlds_judge_r1.jsonl --arms {world_arm} "
        f"{where(world_arm, hrs('worlds_judge'))}",
        f"# --- notes, authored by {note_arm}; audited by {W.NOTE_AUDIT_ARM} (a different model). Round R: notes-requests R, run, "
        "notes-process R, notes-judge-requests R, run, notes-process-judge R; repeat until notes_final.jsonl is complete:",
        f"{run} --exp-id exp22_notes_r1 --requests {f}requests_notes_r1.jsonl --arms {note_arm} {samp} "
        f"{where(note_arm, hrs('notes_author'))}",
        f"{run} --exp-id exp22_notes_judge_r1 --requests {f}requests_notes_judge_r1.jsonl --arms {W.NOTE_AUDIT_ARM} "
        f"{q27(hrs('notes_audit'))}",
        "# --- Exp 2.2 monitors (greedy). Run `monitor-requests` first; every note is asserted against its log at that point",
        f"{run} --exp-id exp22_monitor_intact --requests {f}requests_exp22_monitor_intact.jsonl --arms intact "
        f"{q27(hrs('monitor_intact_cot_off', 'monitor_intact_cot_on'))}",
    ]
    if "llama70b" in arms:
        cmds.append(f"{run} --exp-id exp22_monitor_llama70b --requests {f}requests_exp22_monitor_llama70b.jsonl --arms llama70b "
                    f"--model-kind llama --gpu H200 --est-hours {hrs('monitor_llama70b_cot_off')} --max-model-len 8192 "
                    "--max-num-seqs 256 --eager 0 --extra-json '{\"enable_prefix_caching\": true}'")
    if "flash_next" in arms:
        cmds.append(f"{run} --exp-id exp22_monitor_flash_next --requests {f}requests_exp22_monitor_flash_next.jsonl "
                    f"--arms flash_next {flash(hrs('monitor_flash_next_cot_off'))}")
    cmds += [
        "# --- Exp 2.1: alpha and beta are SAMPLED (K samples, T=0.7); monitor + judge are greedy. Build each stage file "
        "after the previous stage's raw_transcripts exist:",
        f"{run} --exp-id exp21_alpha --requests {f}requests_exp21_alpha.jsonl --arms intact,refusal_ablated "
        f"--temperature 0.7 --top-p 0.95 --seed-base 21 {q27(hrs('alpha_intact'))}",
        f"{run} --exp-id exp21_beta --requests {f}requests_exp21_beta.jsonl --arms intact "
        f"--temperature 0.7 --top-p 0.95 --seed-base 21 {q27(hrs('beta'))}",
        f"{run} --exp-id exp21_monitor --requests {f}requests_exp21_monitor.jsonl --arms intact {q27(hrs('monitor+judge'))}",
        "# --- analysis (reads raw_transcripts only; writes records/exp22_monitor_calibration.json and records/exp21_topologies.json)",
        "PYTHONPATH=src .venv/bin/python scripts/analyze_exp2122.py",
    ]
    return cmds


def placeholder_data() -> None:
    """Dry-run inputs (placeholder worlds and notes, no scenario text) for checking request building and launch
    plumbing. Refuses to write into the real data directory."""
    if D.resolve() == DEFAULT_D.resolve():
        sys.exit("placeholder-data writes fake worlds: pass --data-dir <scratch dir>")
    worlds = W.make_placeholder_worlds()
    _save(D / "worlds_final.jsonl", worlds)
    notes = []
    for w in worlds:
        for bundle in W.BUNDLES:
            log = W.ActionLog.from_bundle(w, bundle)
            for cond in W.CONDITIONS:
                spec = W.NOTE_SPECS[cond]
                text = None if spec is None else " ".join(
                    ([log.alpha_action, log.beta_action] if spec.code_bearing else []) + ["filler"] * 20)
                notes.append({"note_id": W.note_id(w["world_id"], bundle, cond), "world_id": w["world_id"], "bundle": bundle,
                              "condition": cond, "note": text})
    _save(D / "notes_final.jsonl", notes)
    print(f"wrote {len(worlds)} placeholder worlds and {len(notes)} notes to {D}")


def main() -> None:
    global D, RAW
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="state / request directory (default data/exp2122)")
    ap.add_argument("--raw-dir", help="raw_transcripts directory (default raw_transcripts)")
    ap.add_argument("--world-author-arm", default=ARM, choices=["flash_next_uncensored", "refusal_ablated"],
                    help="model that writes (and judges) the worlds; default per the spec")
    ap.add_argument("--note-author-arm", default=W.NOTE_AUTHOR_ARM, choices=["refusal_ablated", "flash_next_uncensored"],
                    help="model that writes the notes (no harm text in a note)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("placeholder-data")
    for name in ("worlds-requests", "worlds-process", "worlds-judge-requests", "worlds-process-judge", "notes-requests",
                 "notes-process", "notes-judge-requests", "notes-process-judge"):
        p = sub.add_parser(name)
        p.add_argument("round", type=int)
        if name == "worlds-judge-requests":
            p.add_argument("--in-flight", default="", help="comma-separated earlier judge rounds still running")
        if name == "worlds-requests":
            p.add_argument("--scale", type=float, default=1.0)
            p.add_argument("--n-variants", type=int, default=None, help="worlds per domain (default from --scale)")
        if name == "notes-requests":
            p.add_argument("--scale", type=float, default=1.0)
    p = sub.add_parser("monitor-requests")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--arms", default="intact,llama70b")
    p.add_argument("--no-cot-on-slice", action="store_true")
    p.add_argument("--exclude-worlds", default="", help="comma-separated world ids left out of the monitor grid")
    p.add_argument("--effort", default="auto", choices=["auto", "default", "low", "medium", "xhigh"],
                   help="reasoning effort of the cot_on slice (auto = the Phase-2 decision recorded by Phase 1)")
    p = sub.add_parser("exp21-requests")
    p.add_argument("stage", choices=["alpha", "beta", "monitor"])
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--cot-mode", default="cot_on", choices=["cot_on", "cot_off"])
    p.add_argument("--judge-arm", default=X.MONITOR_ARM)
    p.add_argument("--effort", default="auto", choices=["auto", "default", "low", "medium", "xhigh"],
                   help="reasoning effort of agents, monitor and judge (auto = the Phase-2 decision recorded by Phase 1)")
    for name in ("estimate", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--scale", type=float, default=1.0)
        p.add_argument("--arms", default="intact,llama70b")
    a = ap.parse_args()
    D, RAW = pathlib.Path(a.data_dir) if a.data_dir else D, pathlib.Path(a.raw_dir) if a.raw_dir else RAW
    if a.cmd == "placeholder-data":
        placeholder_data()
    elif a.cmd == "worlds-requests":
        worlds_requests(a.round, a.n_variants or W.scale_plan(a.scale)["variants_per_domain"], a.world_author_arm)
    elif a.cmd == "notes-requests":
        notes_requests(a.round, a.scale, a.note_author_arm)
    elif a.cmd == "monitor-requests":
        monitor_requests(a.scale, a.arms.split(","), not a.no_cot_on_slice, a.effort, tuple(x for x in a.exclude_worlds.split(",") if x))
    elif a.cmd == "exp21-requests":
        exp21_requests(a.stage, a.scale, a.cot_mode, a.judge_arm, a.effort)
    elif a.cmd == "estimate":
        print(json.dumps(estimates(a.scale, a.arms.split(","), a.note_author_arm, a.world_author_arm), indent=1))
    elif a.cmd == "plan":
        print("\n".join(plan_commands(a.scale, a.arms.split(","), a.note_author_arm, a.world_author_arm)))
    elif a.cmd == "worlds-judge-requests":
        worlds_judge_requests(a.round, a.world_author_arm, tuple(int(x) for x in a.in_flight.split(",") if x))
    else:
        {"worlds-process": worlds_process, "worlds-process-judge": worlds_process_judge, "notes-process": notes_process,
         "notes-judge-requests": notes_judge_requests, "notes-process-judge": notes_process_judge}[a.cmd](a.round)


if __name__ == "__main__":
    main()
