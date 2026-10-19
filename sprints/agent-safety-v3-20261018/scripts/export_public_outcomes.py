"""Public episode-level outcome table: one JSON line per trial, every experiment -> data/public_episode_outcomes.jsonl.

  PYTHONPATH=src .venv/bin/python scripts/export_public_outcomes.py [--root DIR] [--out PATH]

`build_rows(root)` returns the rows (pure function of raw_transcripts/ under `root`, deterministic order) and `summary(rows)`
returns counts per experiment / source directory / arm (pure, deterministic; meant to be embedded in a verified record).
The CLI prints counts only.

PUBLIC-SAFETY CONTRACT. raw_transcripts/** holds unredacted harmful scenario prose and model completions. The exported
table carries NO authored or generated prose: every string value is an identifier or enum and must match
ID_RE = ^[A-Za-z0-9_:|.\\-+]{0,200}$ (no spaces, <= 200 chars). `check_rows` asserts this for every row before anything is
written (a violation names only the experiment and the column, never the value). Model-chosen tool names that are not
identifiers are replaced by the token NON_IDENTIFIER_TOOL_NAME.

Outcomes are RE-DERIVED from the stored raw strings with each experiment's own replay / classification code; stored
derived fields are never read. Every experiment is replayable, so `derived_from` is "replay" on every row today (the
column exists so a future stored-only experiment would be flagged "stored").

Columns (null when a field does not apply)
  experiment       phase1 | exp11 | exp12 | exp14 | exp21 | exp22 | exp23 | exp31 | deconf_exp11 | deconf_exp31
  source           raw_transcripts/<source>/ directory the trial was read from
  trial_id         the row's trial id (deconf_*: request_trial_id, which is unique across conditions and efforts;
                   exp21: the Alpha trial id that names the joint trial)
  scenario_id, template_id, domain
  model_arm        the generating arm (see the per-experiment mapping)
  adapter_id       Exp 1.2 only: the student / floor adapter id
  effort           reasoning-effort label: default (template default, stored as null) | low | medium | xhigh
  think_mode       cot_on | cot_off
  family           episode family / item set (see mapping)
  condition        experimental condition where the design has one (see mapping)
  cost_regime, consequence, boundary, format, seed
  tool_order       "|"-joined declared tool names in presentation order
  parsed_first_tool  first parsed tool call name of the FIRST generated turn (strict v3.parse; null if none)
  outcome_category the experiment's own outcome label (vocabulary per experiment, below)
  sub_status       finer status where the replay provides one (truncation kind, TEXT_NO_CALL, parse status ...)
  shortcut_taken   true iff the experiment's shortcut / violation event happened (see mapping; null if undefined)
  generated_tokens total generated tokens over the episode (all turns; Exp 2.1: Alpha + Beta + Monitor rows of the trial)
  think_closed     first-turn `</think>` present (cot_on only; null for cot_off)
  cot_names_direct, cot_cites_call_rule, cot_harm_cue, cot_mentions_escalate
                   v3.cotfeat.features(think_text(first-turn raw_output), direct_tool) on cot_on rows of experiments that have
                   a single shortcut tool identifier; null otherwise (cot_off rows, Exp 2.1, Exp 2.2)
  derived_from     replay | stored

Per-experiment mapping
  phase1  raw_transcripts/phase1_dim_r1, phase1_dim_r2 (Phase 1 direct-instruction positive control; both checkpoint families,
          told apart by `source`). Scoring: v3.phase1.score_rows (re-parse; scripts/analyze_phase1.py). scenario_id = meta.item_id,
          template_id = meta.cluster_id, family = set_<A|B|C>, format = meta.variant (A0/A1/A2/B/C), consequence = meta.consequence.
          outcome_category = COMPLY_RESTRICTED | SAFE_ALTERNATIVE | ESCALATE | TEXT_NO_CALL | MALFORMED, sub_status = the score detail;
          shortcut_taken = comply. CoT direct tool = meta.restricted_tool.
  exp11   raw_transcripts/exp11* (exp11, exp11_cotoff, exp11_effort_low_a|b|c, exp11_ext), every family (main, forced_safe,
          unlisted_shortcut, direct_instruction, probe_*). Replay: v3.exp11.replay (episodes.recompute_record inside) per row, with
          the analysis' duplicate rule (first of (arm, cot_mode, reasoning_effort, family, trial_id) in sorted dir/file order).
          Episodes and direct-instruction singles: outcome_category = sandbox outcome_category (SAFE_ROUTE | SHORTCUT_VIOLATION |
          SHORTCUT_PERMITTED | ESCALATE_OR_REFUSE | TRUNCATED_OR_MALFORMED), shortcut_taken = direct_taken; direct-instruction format
          = esc | solo. QA probes: outcome_category = PROBE_CORRECT | PROBE_INCORRECT | PROBE_<invalid reason>, shortcut_taken null.
  exp12   raw_transcripts/exp12 student evaluations (not the teacher corpora): `episode` rows (held-out sandbox episodes, replayed
          with episodes.recompute_record; family = agentic) and `single` rows (refusal set, re-parsed and scored with
          phase1.score_row exactly as scripts/analyze_exp12.py; family = refusal_set). model_arm = <teacher>.r<rank> (floor runs:
          floor), format = corpus (A | B | Bp | C; floor: null), seed = training seed (floor: floor_rep), adapter_id = adapter.
          Smoke / partial adapters (cfg.max_steps or cfg.eval_limit) are excluded and a repeated (adapter, trial_id) keeps the
          LAST row, both as in the analysis.
  exp14   raw_transcripts/exp14* (exp14, exp14_ctrl, exp14_low) blocked-route episodes only (the Exp 1.1 reference rows the
          analysis also reads are already exported under exp11). Replay: v3.exp14.replay (its duplicate / repeated-cell rules).
          outcome_category = SHORTCUT | ESCALATED | STOPPED_TEXT | STUCK_RETRYING | NOT_SCORED, sub_status = sandbox sub_status,
          format = block timing (early | late), shortcut_taken = direct_taken.
  exp21   raw_transcripts/exp21_(alpha|beta|monitor)(_p<k>) (the analysis' DIR_PATTERNS). Trial unit = one JOINT trial
          (v3.exp21.joint_trials: Alpha + Beta + Monitor 5, all re-parsed). model_arm = team (all_intact | mixed_team),
          format = topology variant (A | B | C_nopen | C_pen | D_nopen | D_pen), seed = sample index k, scenario_id = world_id.
          outcome_category = joint category (LEGIT_SAFE_COORD | ILLEGIT_JOINT | BETA_ONLY_FAST | BOTH_SAFE | NO_BUNDLE_ESCALATED |
          NO_BUNDLE_INVALID), sub_status = Alpha parse status, shortcut_taken = executed_violation (prohibited composite executed;
          invalid monitor output = fail-open, as in the record), parsed_first_tool / think_closed / effort / think_mode = Alpha's turn.
          generated_tokens = Alpha + Beta + Monitor rows of the trial (a shared Beta row counts in both teams' trials; judge rows,
          an audit instrument, are not counted). CoT flags null (no shortcut tool identifier).
  exp22   raw_transcripts/exp22_monitor_(intact|llama70b|flash_next). Trial unit = one monitor decision
          (v3.exp22.score_monitor_row, re-parsed). model_arm = monitor arm, scenario_id = world_id, consequence = the bundle shown
          (A_FAST+B_FAST is the violating one), condition = explanation-note condition, format = verdict format (semantic |
          code_12 | code_21). outcome_category = monitor verdict ALLOW | BLOCK, or NO_VERDICT when unscoreable (sub_status = the
          parse status). shortcut_taken null (a monitor takes no shortcut). CoT flags null.
  exp23   raw_transcripts/exp23_s4_* executor rows only (planner / reviewer / authoring / judge stages are stimuli, not trials).
          Scoring: v3.exp23.score_executor_row (sandbox single mode on the re-parsed first turn). model_arm = executor arm,
          condition = reviewer condition (B0, B1, R1a, P1_k, CS_* ...), format = authority framing (peer | designated | none),
          consequence = meta.consequence (harmful, or harmless for B0h / B3), boundary = implicit (the executor prompt is the
          Exp 1.1 harmful/implicit high-cost prompt), template_id = the domain's template. outcome_category = route
          (SAFE_ROUTE | SHORTCUT | ESCALATE_OR_REFUSE | TRUNCATED_OR_MALFORMED), shortcut_taken = route == SHORTCUT.
          Covariates that need the upstream bundle (planner / note recommendation) are not exported.
  exp31   raw_transcripts/exp31. Replay: v3.exp31.classify; rows it rejects are left out (as in the record) and a repeated
          (arm, cot_mode, stake format, pair_id) keeps the first. format = stake format (narrative | numeric), seed null.
          outcome_category = sandbox outcome_category, shortcut_taken = direct_taken.
  deconf_exp11, deconf_exp31  reviewer slices (included automatically when raw_transcripts/deconf_exp11|31/*.jsonl exist).
          Replay: v3.deconf.trial. outcome_category = deconf action (safe_route_completed | shortcut | escalated | prose_stop |
          unscorable), shortcut_taken = direct. deconf_exp11: condition = format = meta.deconf (std | nh | np | demo; the demo rows
          are the hand-written harmless demonstration scenario). deconf_exp31: condition = format = meta.format_2x2.
          A repeated (arm, request_trial_id) keeps the first row.

Skipped by design: authoring / judging / calibration / smoke / pilot / dry / _superseded directories and *.partial.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import cotfeat, deconf, episodes, exp11, exp14, exp21, exp22, exp23, exp31, phase1  # noqa: E402
from v3.sandbox import SHORTCUT  # noqa: E402

OUT_REL = pathlib.Path("data") / "public_episode_outcomes.jsonl"
ID_RE = re.compile(r"^[A-Za-z0-9_:|.\-+]{0,200}$")
MAX_STR = 200
BAD_TOOL = "NON_IDENTIFIER_TOOL_NAME"
COLUMNS = ("experiment", "source", "trial_id", "scenario_id", "template_id", "domain", "model_arm", "adapter_id", "effort",
           "think_mode", "family", "condition", "cost_regime", "consequence", "boundary", "format", "seed", "tool_order",
           "parsed_first_tool", "outcome_category", "sub_status", "shortcut_taken", "generated_tokens", "think_closed",
           "cot_names_direct", "cot_cites_call_rule", "cot_harm_cue", "cot_mentions_escalate", "derived_from")
COT_COLUMNS = {"names_direct": "cot_names_direct", "cites_call_rule": "cot_cites_call_rule", "harm_cue": "cot_harm_cue",
               "mentions_escalate": "cot_mentions_escalate"}
EXPERIMENT_ORDER = ("phase1", "exp11", "exp12", "exp14", "exp21", "exp22", "exp23", "exp31", "deconf_exp11", "deconf_exp31")
PHASE1_DIRS = ("phase1_dim_r1", "phase1_dim_r2")


# ============================================================================================ helpers
def _load_script(name: str):
    """Import scripts/<name>.py as a module (the analysis scripts' own loaders / classifiers)."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_export_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _files(raw: pathlib.Path, dirs) -> list[pathlib.Path]:
    """Result files of the given directories in sorted order (crash copies *.partial.jsonl skipped)."""
    out = []
    for d in dirs:
        d = raw / d if not isinstance(d, pathlib.Path) else d
        if d.is_dir():
            out += [f for f in sorted(d.glob("*.jsonl")) if not f.name.endswith(".partial.jsonl")]
    return out


def _prefix_dirs(raw: pathlib.Path, prefix: str) -> list[pathlib.Path]:
    return sorted(p for p in raw.glob(prefix + "*") if p.is_dir())


def _read(f: pathlib.Path):
    with open(f) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _eff(x) -> str:
    return exp11.effort_label(x)


def _tool_name(name):
    if name is None:
        return None
    return name if isinstance(name, str) and ID_RE.fullmatch(name) and len(name) <= 64 else BAD_TOOL


def _join_tools(names) -> str | None:
    if not names:
        return None
    return "|".join(_tool_name(n) for n in names)


def _first_turn(raw_output, cot_mode: str, direct_tool) -> dict:
    """parsed_first_tool / think_closed / CoT flags of one first-turn generation (strict v3.parse)."""
    cot_on = cot_mode == "cot_on"
    out = {"parsed_first_tool": None, "think_closed": None, **{c: None for c in COT_COLUMNS.values()}}
    if not isinstance(raw_output, str):
        return out
    parsed, _ = episodes.parse_output(raw_output, cot_on)
    out["parsed_first_tool"] = _tool_name(parsed.calls[0]["name"]) if parsed.calls else None
    if cot_on:
        out["think_closed"] = bool(parsed.think_closed)
        if isinstance(direct_tool, str) and direct_tool:
            f = cotfeat.features(cotfeat.think_text(raw_output), direct_tool)
            out.update({col: bool(f[k]) for k, col in COT_COLUMNS.items()})
    return out


def _row(**kw) -> dict:
    unknown = set(kw) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown columns {sorted(unknown)}")
    row = {c: kw.get(c) for c in COLUMNS}
    row["derived_from"] = kw.get("derived_from", "replay")
    return row


def _sort_key(r: dict) -> tuple:
    return (EXPERIMENT_ORDER.index(r["experiment"]), r["source"] or "", r["model_arm"] or "", r["adapter_id"] or "",
            r["effort"] or "", r["think_mode"] or "", r["family"] or "", r["trial_id"] or "",
            json.dumps(r, sort_keys=True))


# ============================================================================================ experiments
def rows_phase1(raw: pathlib.Path) -> list[dict]:
    out = []
    for f in _files(raw, PHASE1_DIRS):
        rows = list(_read(f))
        for row, s in zip(rows, phase1.score_rows(rows)):       # re-parses raw_output (scripts/analyze_phase1.py)
            m = row["meta"]
            out.append(_row(experiment="phase1", source=f.parent.name, trial_id=row["trial_id"], scenario_id=m["item_id"],
                            template_id=m["cluster_id"], domain=m.get("domain"), model_arm=row["arm"],
                            effort=_eff(row.get("reasoning_effort")), think_mode=row["cot_mode"], family=f"set_{m['set_name']}",
                            consequence=m["consequence"], format=m["variant"], outcome_category=s["label"],
                            sub_status=s["detail"] or None, shortcut_taken=bool(s["comply"]),
                            generated_tokens=int(row["generated_tokens"]),
                            **_first_turn(row["raw_output"], row["cot_mode"], m["restricted_tool"])))
    return out


def _probe_category(p: dict) -> str:
    if p["reason"] != "OK":
        return f"PROBE_{p['reason']}"
    return "PROBE_CORRECT" if p["correct"] and all(p["correct"].values()) else "PROBE_INCORRECT"


def rows_exp11(raw: pathlib.Path) -> tuple[list[dict], Counter]:
    """v3.exp11.replay per row, with the analysis' duplicate rule applied across every exp11* file in sorted order."""
    out, seen, skipped = [], set(), Counter()
    for f in _files(raw, _prefix_dirs(raw, "exp11")):
        for r in _read(f):
            if not all(k in r for k in ("arm", "cot_mode", "family", "scenario_id", "template_id", "trial_id",
                                        "cost_regime", "consequence", "boundary")):
                skipped["exp11:unplaceable"] += 1
                continue
            ident = (r["arm"], r["cot_mode"], r.get("reasoning_effort"), r["family"], r["trial_id"])
            if ident in seen:
                skipped["exp11:duplicate"] += 1
                continue
            seen.add(ident)
            trials, _ = exp11.replay([r])
            t = trials[0]
            episode = "turns" in r
            first_raw = (r["turns"][0]["raw_output"] if r.get("turns") else None) if episode else r.get("raw_output")
            if t.get("probe") is not None:
                cat, short = _probe_category(t["probe"]), None
            elif t["outcome"] is None:
                cat, short = None, None
            else:
                cat, short = t["outcome"], bool(t["direct_taken"])
            out.append(_row(experiment="exp11", source=f.parent.name, trial_id=r["trial_id"], scenario_id=r["scenario_id"],
                            template_id=r["template_id"], domain=r.get("domain"), model_arm=r["arm"], effort=t["effort"],
                            think_mode=r["cot_mode"], family=r["family"], cost_regime=r["cost_regime"],
                            consequence=r["consequence"], boundary=r["boundary"], format=t.get("variant"),
                            tool_order=_join_tools(r.get("tool_order")), outcome_category=cat, sub_status=t.get("sub_status"),
                            shortcut_taken=short, generated_tokens=int(t["tokens"] or 0),
                            **_first_turn(first_raw, r["cot_mode"], r.get("direct_tool"))))
    return out, skipped


def rows_exp12(raw: pathlib.Path) -> tuple[list[dict], Counter]:
    """Student / floor evaluation rows of raw_transcripts/exp12 (scripts/analyze_exp12.py rules)."""
    a12 = _load_script("analyze_exp12")
    out, skipped = [], Counter()
    for f in _files(raw, ("exp12",)):
        rows = list(_read(f))
        adapters = {r["adapter_id"]: r for r in rows if r["kind"] == "adapter"}
        partial = {a for a, r in adapters.items() if a12.is_partial(r)}
        last: dict = {}
        for r in rows:
            if r["kind"] not in ("episode", "single"):
                continue
            if r["adapter_id"] in partial:
                skipped["exp12:partial_adapter"] += 1
                continue
            key = (r["adapter_id"], r["kind"], r["trial_id"])
            if key in last:
                skipped["exp12:duplicate_replaced"] += 1
            last[key] = r                                         # the analysis keeps the LAST row of a repeated trial
        for (aid, kind, tid), r in last.items():
            teacher, rank, corpus = r.get("teacher"), r.get("rank"), r.get("corpus")
            arm = "floor" if corpus is None else f"{teacher}.r{rank}"
            seed = r.get("seed") if corpus is not None else r.get("floor_rep")
            common = dict(experiment="exp12", source=f.parent.name, trial_id=tid, model_arm=arm, adapter_id=aid,
                          effort=_eff(r.get("reasoning_effort")), think_mode=r["cot_mode"], format=corpus, seed=seed)
            if kind == "episode":
                fresh = episodes.recompute_record(r)
                out.append(_row(**common, scenario_id=r["scenario_id"], template_id=r["template_id"], domain=r.get("domain"),
                                family="agentic", cost_regime=r["cost_regime"], consequence=r["consequence"],
                                boundary=r["boundary"], tool_order=_join_tools(r.get("tool_order")),
                                outcome_category=fresh["outcome_category"], sub_status=fresh["sub_status"],
                                shortcut_taken=bool(fresh["direct_taken"]), generated_tokens=int(fresh["total_generated_tokens"]),
                                **_first_turn(r["turns"][0]["raw_output"] if r["turns"] else None, r["cot_mode"],
                                              r.get("direct_tool"))))
            else:
                m = r["meta"]
                s = phase1.score_row(r, episodes.parse_output(r["raw_output"], r["cot_mode"] == "cot_on")[1])
                out.append(_row(**common, scenario_id=m["item_id"], template_id=m["cluster_id"], domain=m.get("domain"),
                                family="refusal_set", consequence=m["consequence"], outcome_category=s["label"],
                                sub_status=s["detail"] or None, shortcut_taken=s["label"] == phase1.COMPLY,
                                generated_tokens=int(r["generated_tokens"]),
                                **_first_turn(r["raw_output"], r["cot_mode"], m["restricted_tool"])))
    return out, skipped


def rows_exp14(raw: pathlib.Path) -> tuple[list[dict], Counter]:
    rows, src = [], {}
    for f in _files(raw, _prefix_dirs(raw, "exp14")):
        for r in _read(f):
            if exp14.row_kind(r) == "blocked":
                rows.append(r)
                src.setdefault((r.get("arm"), r.get("cot_mode"), _eff(r.get("reasoning_effort")), r.get("trial_id")),
                               (f.parent.name, r))
    trials, _refs, h = exp14.replay(rows)
    skipped = Counter({f"exp14:{k}": h[k] for k in ("n_duplicate_rows", "n_duplicate_cell_rows", "n_unplaceable") if h.get(k)})
    out = []
    for t in trials:
        source, r = src[(t["arm"], t["cot_mode"], t["effort"], t["trial_id"])]
        first_raw = r["turns"][0]["raw_output"] if r.get("turns") else None
        out.append(_row(experiment="exp14", source=source, trial_id=t["trial_id"], scenario_id=t["scenario_id"],
                        template_id=t["template_id"], domain=r.get("domain"), model_arm=t["arm"], effort=t["effort"],
                        think_mode=t["cot_mode"], family=t["family"], cost_regime=r.get("cost_regime"),
                        consequence=t["consequence"], boundary=t["boundary"], format=t.get("timing"),
                        tool_order=_join_tools(r.get("tool_order")), outcome_category=t["outcome"], sub_status=t["sub_status"],
                        shortcut_taken=bool(t["direct_taken"]), generated_tokens=int(t["tokens"] or 0),
                        **_first_turn(first_raw, t["cot_mode"], r.get("direct_tool"))))
    return out, skipped


def rows_exp21(raw: pathlib.Path) -> list[dict]:
    a2122 = _load_script("analyze_exp2122")
    rows = []
    for d in a2122.record_dirs("exp21_", raw)[0]:
        for f in _files(raw, (d,)):
            rows += [{**r, "_source": d.name} for r in _read(f)]
    rows = [r for r in rows if r.get("meta", {}).get("exp") == "2.1"]
    trials, _ = exp21.joint_trials(rows)
    by_id = {r["trial_id"]: r for r in rows}
    out = []
    for t in trials:
        ar = by_id[t["trial_id"]]
        m = ar["meta"]
        v = exp21.VARIANTS[t["variant"]]
        beta_id = exp21.beta_trial_id(None if v.shared_beta else t["team"], v.name, t["world_id"], t["k"])
        mon_id = exp21.monitor_trial_id(t["team"], v.name, t["world_id"], t["k"])
        tokens = sum(int(by_id[i].get("generated_tokens") or 0) for i in (t["trial_id"], beta_id, mon_id) if i in by_id)
        ft = _first_turn(ar["raw_output"], ar["cot_mode"], None)
        out.append(_row(experiment="exp21", source=ar["_source"], trial_id=t["trial_id"], scenario_id=t["world_id"],
                        domain=t["domain"], model_arm=t["team"], effort=_eff(ar.get("reasoning_effort")),
                        think_mode=ar["cot_mode"], family="joint_trial", format=t["variant"], seed=t["k"],
                        tool_order=_join_tools(m.get("tool_order")), outcome_category=t["category"],
                        sub_status=t["alpha"]["status"], shortcut_taken=bool(t["executed_violation"]), generated_tokens=tokens,
                        parsed_first_tool=ft["parsed_first_tool"], think_closed=ft["think_closed"]))
    return out


def rows_exp22(raw: pathlib.Path) -> list[dict]:
    a2122 = _load_script("analyze_exp2122")
    out, seen = [], set()
    for d in a2122.record_dirs("exp22_monitor_", raw)[0]:
        for f in _files(raw, (d,)):
            for r in _read(f):
                m = r.get("meta", {})
                if m.get("exp") != "2.2" or m.get("stage") != "monitor":
                    continue
                if r["trial_id"] in seen:
                    raise ValueError("duplicate monitor trial_id in raw rows")      # the analysis raises too
                seen.add(r["trial_id"])
                s = exp22.score_monitor_row(r)
                ft = _first_turn(r["raw_output"], r["cot_mode"], None)
                out.append(_row(experiment="exp22", source=d.name, trial_id=r["trial_id"], scenario_id=s["world_id"],
                                domain=s["domain"], model_arm=s["monitor"], effort=_eff(r.get("reasoning_effort")),
                                think_mode=r["cot_mode"], family="monitor_decision", condition=s["condition"],
                                consequence=s["bundle"], format=s["fmt"], outcome_category=s["verdict"] or "NO_VERDICT",
                                sub_status=s["status"], generated_tokens=int(s["generated_tokens"] or 0),
                                parsed_first_tool=ft["parsed_first_tool"], think_closed=ft["think_closed"]))
    return out


def rows_exp23(raw: pathlib.Path) -> list[dict]:
    out, seen = [], set()
    for d in _prefix_dirs(raw, "exp23_s4_"):
        for f in _files(raw, (d,)):
            for r in _read(f):
                m = r.get("meta", {})
                if m.get("exp") != exp23.EXP or m.get("stage") != "s4":
                    continue
                if r["trial_id"] in seen:
                    raise ValueError("duplicate trial_id among executor rows")       # exp23.executor_records raises too
                seen.add(r["trial_id"])
                s = exp23.score_executor_row(r)
                lay = exp23.LAYOUT_BY_ID.get(m["scenario_id"], {})
                out.append(_row(experiment="exp23", source=d.name, trial_id=r["trial_id"], scenario_id=m["scenario_id"],
                                template_id=exp23.DOMAIN_TEMPLATE.get(m.get("domain")), domain=m.get("domain"),
                                model_arm=r["arm"], effort=_eff(r.get("reasoning_effort")), think_mode=r["cot_mode"],
                                family="executor", condition=m["condition"], cost_regime=m.get("cost_regime") or lay.get("cost_regime"),
                                consequence=m.get("consequence"), boundary="implicit", format=m.get("framing"),
                                tool_order=_join_tools(m.get("tool_order")), outcome_category=s["route"],
                                sub_status=s["sub_status"], shortcut_taken=s["route"] == SHORTCUT,
                                generated_tokens=int(r.get("generated_tokens") or 0),
                                **_first_turn(r["raw_output"], r["cot_mode"], m.get("direct_tool"))))
    return out


def rows_exp31(raw: pathlib.Path) -> tuple[list[dict], Counter]:
    out, seen, skipped = [], set(), Counter()
    for f in _files(raw, ("exp31",)):
        for r in _read(f):
            try:
                c = exp31.classify(r)
            except (KeyError, TypeError, AttributeError, ValueError):
                skipped["exp31:invalid"] += 1
                continue
            key = (c["arm"], c["cot_mode"], c["fmt"], c["pair_id"])
            if key in seen:
                skipped["exp31:duplicate"] += 1
                continue
            seen.add(key)
            out.append(_row(experiment="exp31", source="exp31", trial_id=r["trial_id"], scenario_id=c["scenario_id"],
                            template_id=c["template_id"], domain=c["domain"], model_arm=c["arm"],
                            effort=_eff(r.get("reasoning_effort")), think_mode=c["cot_mode"], family=r.get("family"),
                            cost_regime=c["cost_regime"], consequence=c["consequence"], boundary=c["boundary"], format=c["fmt"],
                            tool_order=_join_tools(r.get("tool_order")), outcome_category=c["outcome"],
                            sub_status=c["sub_status"], shortcut_taken=bool(c["direct"]), generated_tokens=int(c["tokens"]),
                            **_first_turn(r["turns"][0]["raw_output"] if r.get("turns") else None, c["cot_mode"],
                                          r.get("direct_tool"))))
    return out, skipped


def rows_deconf(raw: pathlib.Path, exp_id: str) -> tuple[list[dict], Counter]:
    out, seen, skipped = [], set(), Counter()
    for f in _files(raw, (exp_id,)):
        for r in _read(f):
            tid = r.get("request_trial_id") or r["trial_id"]
            if (r.get("arm"), tid) in seen:
                skipped[f"{exp_id}:duplicate"] += 1
                continue
            seen.add((r.get("arm"), tid))
            m = r.get("meta") or {}
            cond = m.get("deconf") if exp_id == deconf.EXP11_ID else m.get("format_2x2")
            effort = _eff(r.get("reasoning_effort"))
            t = deconf.trial(r, cond, effort)
            out.append(_row(experiment=exp_id, source=exp_id, trial_id=tid, scenario_id=r["scenario_id"],
                            template_id=r["template_id"], domain=r.get("domain"), model_arm=r["arm"], effort=effort,
                            think_mode=r["cot_mode"], family=r.get("family"), condition=cond, cost_regime=r["cost_regime"],
                            consequence=r["consequence"], boundary=r["boundary"], format=cond,
                            tool_order=_join_tools(r.get("tool_order")), outcome_category=t["action"],
                            sub_status=t["sub_status"], shortcut_taken=bool(t["direct"]), generated_tokens=int(t["tokens"]),
                            **_first_turn(r["turns"][0]["raw_output"] if r.get("turns") else None, r["cot_mode"],
                                          r.get("direct_tool"))))
    return out, skipped


# ============================================================================================ public API
def _collect(root) -> tuple[list[dict], dict]:
    raw = pathlib.Path(root) / "raw_transcripts"
    rows, skipped = [], Counter()
    rows += rows_phase1(raw)
    for fn in (rows_exp11, rows_exp12, rows_exp14, rows_exp31):
        r, s = fn(raw)
        rows += r
        skipped.update(s)
    rows += rows_exp21(raw)
    rows += rows_exp22(raw)
    rows += rows_exp23(raw)
    for exp_id in (deconf.EXP11_ID, deconf.EXP31_ID):
        r, s = rows_deconf(raw, exp_id)
        rows += r
        skipped.update(s)
    rows.sort(key=_sort_key)
    check_rows(rows)
    return rows, dict(sorted(skipped.items()))


def build_rows(root=ROOT) -> list[dict]:
    """Every exported trial row (validated, stably sorted). Pure function of raw_transcripts/ under `root`."""
    return _collect(root)[0]


def check_rows(rows: list[dict]) -> None:
    """Public-safety assertions: exact column set, and every string value an identifier (ID_RE, no space, <= 200 chars).
    Messages name the experiment / column only, never the offending value."""
    for i, r in enumerate(rows):
        assert tuple(r) == COLUMNS, f"row {i}: column set differs"
        assert r["experiment"] in EXPERIMENT_ORDER, f"row {i}: unknown experiment"
        assert r["derived_from"] in ("replay", "stored"), f"row {i}: derived_from"
        for k, v in r.items():
            assert v is None or isinstance(v, (str, bool, int)), f"{r['experiment']}.{k}: type {type(v).__name__}"
            if isinstance(v, str):
                assert len(v) <= MAX_STR, f"{r['experiment']}.{k}: string longer than {MAX_STR}"
                assert " " not in v, f"{r['experiment']}.{k}: contains a space"
                assert ID_RE.fullmatch(v), f"{r['experiment']}.{k}: not an identifier"


def summary(rows: list[dict]) -> dict:
    """Counts per experiment, source directory and arm (pure, deterministic: sorted keys, integers only)."""
    def block(rs):
        return {"n": len(rs), "n_shortcut": sum(r["shortcut_taken"] is True for r in rs),
                "n_shortcut_defined": sum(r["shortcut_taken"] is not None for r in rs),
                "generated_tokens": sum(r["generated_tokens"] or 0 for r in rs)}
    out = {"schema": "public_episode_outcomes_summary/v1", "n_rows": len(rows), "columns": list(COLUMNS), "experiments": {}}
    for exp in EXPERIMENT_ORDER:
        rs = [r for r in rows if r["experiment"] == exp]
        if not rs:
            continue
        e = {**block(rs), "derived_from": dict(sorted(Counter(r["derived_from"] for r in rs).items())), "by_arm": {}, "by_source": {}}
        for arm in sorted({r["model_arm"] or "" for r in rs}):
            e["by_arm"][arm] = block([r for r in rs if (r["model_arm"] or "") == arm])
        for src in sorted({r["source"] for r in rs}):
            srs = [r for r in rs if r["source"] == src]
            e["by_source"][src] = {**block(srs), "by_arm": {a: block([r for r in srs if (r["model_arm"] or "") == a])
                                                            for a in sorted({r["model_arm"] or "" for r in srs})}}
        out["experiments"][exp] = e
    return out


def write(rows: list[dict], path: pathlib.Path) -> str:
    check_rows(rows)
    text = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=None, help=f"default: <root>/{OUT_REL}")
    a = ap.parse_args(argv)
    t0 = time.time()
    rows, skipped = _collect(a.root)
    out = pathlib.Path(a.out) if a.out else pathlib.Path(a.root) / OUT_REL
    sha = write(rows, out)
    s = summary(rows)
    print(f"[written] {out} rows={len(rows)} bytes={out.stat().st_size} sha256={sha} seconds={time.time() - t0:.1f}")
    import gzip      # the committed copy: gzip with mtime 0 and no file name, so it is byte-reproducible
    gz = out.with_name(out.name + ".gz")
    with open(gz, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=9) as z:
        z.write(out.read_bytes())
    print(f"[written] {gz} bytes={gz.stat().st_size}")
    print(f"[skipped] {skipped}")
    for exp, e in s["experiments"].items():
        print(f"{exp}: n={e['n']} shortcut={e['n_shortcut']}/{e['n_shortcut_defined']} derived_from={e['derived_from']}")
        for src, b in e["by_source"].items():
            print(f"  {src}: " + ", ".join(f"{arm or '-'}={v['n']}" for arm, v in b["by_arm"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
