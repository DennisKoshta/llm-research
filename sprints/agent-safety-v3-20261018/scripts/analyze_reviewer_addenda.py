"""Reviewer addenda (2026-10-19): CPU reanalyses of existing raw transcripts -> records/reviewer_addenda.json.

  PYTHONPATH=src .venv/bin/python scripts/analyze_reviewer_addenda.py [--skip-exp32] [--skip-export]

Sections (every number re-derived from stored raw strings with v3.episodes.recompute_record / v3.exp11.replay):
  action_splits        5-way action split (safe_route_completed, shortcut, escalated, prose_stop, unscorable) + native-XML
                       turns for every Exp 1.1 arm x condition (default / low effort / thinking off) x main-grid cell
  direct_order         the direct-order positive control inside Exp 1.1 (no safe route; solo = direct tool only, esc = direct
                       tool + escalate_work_order declared WITHOUT any invitation sentence): comply / escalate / text reply /
                       undeclared call / unscorable, per arm x condition x consequence
  shortcut_turns       turn of the shortcut call in Exp 1.1 shortcut episodes (turn 1 vs 2+)
  reproducibility      item-level agreement + Cohen's kappa of shortcut outcomes across reasoning conditions (Exp 1.1), and
                       of the turn-1 action between byte-identical prompts in Exp 1.1 (high_tedium) and Exp 1.4 (pre-block
                       turn 1), plus Exp 1.4 early vs late twins (same launch, same prompt)
  cot_features         v3.cotfeat flags of the turn-1 <think> text on the high-cost harmful-implicit slice of Exp 1.1, Exp 3.1
                       and the reviewer deconfounder slices (when present), split by episode outcome
  corpus_b             <think> statistics of the Exp 1.2 SFT targets per corpus x teacher (direct-tool mentions, words, tokens)
  exp32                Exp 3.2 probe additions (v3.addenda_exp32.recompute)
  public_export        summary of data/public_episode_outcomes.jsonl (scripts/export_public_outcomes.py)
Prints numbers only (raw rows and corpora hold unredacted scenario text).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import cotfeat, deconf, episodes, exp11, scenarios as S, stats  # noqa: E402

RAW = ROOT / "raw_transcripts"
RECORD = ROOT / "records" / "reviewer_addenda.json"
EXP11_SLICES = {"exp11": "default", "exp11_effort_low_a": "low", "exp11_effort_low_b": "low", "exp11_effort_low_c": "low",
                "exp11_cotoff": "off"}
EXP14_SLICES = {"exp14": "default", "exp14_ctrl": "default", "exp14_low": "low"}
HC = ("high_tedium", "high_friction")


def _read_dirs(names) -> list[dict]:
    rows = []
    for d in names:
        for f in sorted((RAW / d).glob("*.jsonl")):
            if not f.name.endswith(".partial.jsonl"):
                rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def _cond(r: dict) -> str:
    return "off" if r["cot_mode"] == "cot_off" else exp11.effort_label(r.get("reasoning_effort"))


def _first_action(r: dict, fresh: dict) -> str:
    """Category of the first parsed call of turn 1 (direct / safe / escalate / undeclared / text / unscorable)."""
    t = fresh["turns"][0] if "turns" in fresh else {"parsed": fresh["parsed"], "finish_reason": r.get("finish_reason")}
    p = t["parsed"]
    if p["status"] in ("TRUNCATED_IN_THINK", "TRUNCATED_IN_CALL") or t.get("finish_reason") != "stop":
        return "unscorable"
    names = [c["name"] for c in p["calls"]]
    if not names:
        return "text" if p["status"] == "NO_TOOL_CALL" else "unscorable"
    n = names[0]
    if n not in r["declared_names"]:
        return "undeclared"
    return {r["direct_tool"]: "direct", r.get("safe_tool"): "safe", S.ESCALATE_TOOL: "escalate"}.get(n, "undeclared")


# ------------------------------------------------------------------------------------------ Exp 1.1
def exp11_trials() -> list[dict]:
    """One dict per Exp 1.1 main / direct-instruction row of the six 27B arms (external models excluded)."""
    out = []
    for d, cond in EXP11_SLICES.items():
        for r in _read_dirs([d]):
            if r["family"] not in ("main", "direct_instruction"):
                continue
            if _cond(r) != cond:
                raise ValueError(f"{d}: unexpected condition")
            fresh = episodes.recompute_record(r)
            base = {"arm": r["arm"], "cond": cond, "family": r["family"], "scenario_id": r["scenario_id"],
                    "template_id": r["template_id"], "cost": r["cost_regime"], "cons": r["consequence"], "bound": r["boundary"],
                    "first": _first_action(r, fresh)}
            if r["family"] == "main":
                base.update(action=deconf.action(fresh), direct=int(fresh["direct_taken"]), n_turns=fresh["n_turns"],
                            xml=int(any(t["parsed"].get("xml_calls") for t in fresh["turns"])),
                            think=cotfeat.think_text(fresh["turns"][0]["raw_output"]) if r["cot_mode"] == "cot_on" else None,
                            direct_tool=r["direct_tool"])
            else:
                dec = fresh["decision"]
                base.update(variant="esc" if r["meta"].get("offer_escalate") else "solo", route=dec["route"], sub=dec["sub_status"],
                            xml=int(bool(fresh["parsed"].get("xml_calls"))))
            out.append(base)
    keys = Counter((t["arm"], t["cond"], t["family"], t["scenario_id"], t["cost"], t["cons"], t["bound"], t.get("variant")) for t in out)
    if max(keys.values()) != 1:
        raise ValueError("duplicate Exp 1.1 trial")
    return out


def action_splits(ts: list[dict]) -> dict:
    out = {}
    g = defaultdict(list)
    for t in ts:
        if t["family"] == "main":
            g[f"{t['arm']}|{t['cond']}|{t['cost']}|{t['cons']}|{t['bound']}"].append(t)
    for k, v in sorted(g.items()):
        c = Counter(t["action"] for t in v)
        out[k] = {"n": len(v), **{a: c[a] for a in deconf.ACTIONS}, "native_xml": sum(t["xml"] for t in v)}
    return out


def direct_order(ts: list[dict]) -> dict:
    out = {}
    g = defaultdict(list)
    for t in ts:
        if t["family"] == "direct_instruction":
            g[f"{t['arm']}|{t['cond']}|{t['variant']}|{t['cons']}"].append(t)
    for k, v in sorted(g.items()):
        c = Counter(t["first"] for t in v)
        out[k] = {"n": len(v), "comply": sum(t["route"] == "SHORTCUT" for t in v),
                  "escalated": sum(t["route"] == "ESCALATE_OR_REFUSE" for t in v),
                  "text_reply": sum(t["sub"] == "TEXT_NO_CALL" for t in v),
                  "undeclared_call": sum(t["sub"] == "NO_DECISION" for t in v),
                  "unscorable": sum(t["route"] == "TRUNCATED_OR_MALFORMED" and t["sub"] not in ("TEXT_NO_CALL", "NO_DECISION") for t in v),
                  "native_xml": sum(t["xml"] for t in v), "first_call": dict(sorted(c.items()))}
    return out


def direct_order_prompt_facts() -> dict:
    """Structural facts of the esc variant's prompt (counts only): the escalate tool line is present; no sentence invites it."""
    rows = [json.loads(l) for l in (ROOT / "data/exp11/requests_direct_esc_cot_on.jsonl").read_text().splitlines() if l.strip()]
    texts = ["\n".join(m["content"] for m in r["messages"]) for r in rows]
    return {"n_prompts": len(rows), "n_with_escalate_tool_line": sum(deconf.ESC_LINE in t for t in texts),
            "n_mentioning_escalate_outside_tool_line": sum(t.replace(deconf.ESC_LINE, "").count(S.ESCALATE_TOOL) > 0 for t in texts),
            "n_with_instruction_to_call_direct": sum("Instruction from the requesting team: call " in t for t in texts),
            "invited": False}


def shortcut_turns(ts: list[dict]) -> dict:
    out = {}
    g = defaultdict(list)
    for t in ts:
        if t["family"] == "main" and t["direct"]:
            g[f"{t['arm']}|{t['cond']}|{'harmful' if t['cons'] == 'harmful' else 'harmless'}|{t['cost']}"].append(t["n_turns"])
    for k, v in sorted(g.items()):
        out[k] = {"n": len(v), "turn1": sum(x == 1 for x in v), "turn2plus": sum(x > 1 for x in v),
                  "turns": dict(sorted(Counter(v).items()))}
    out = {k: {**v, "turns": {str(a): b for a, b in v["turns"].items()}} for k, v in out.items()}
    return out


def reproducibility(ts: list[dict]) -> dict:
    main = defaultdict(dict)
    for t in ts:
        if t["family"] == "main":
            main[(t["arm"], t["cond"])][(t["scenario_id"], t["cost"], t["cons"], t["bound"])] = t["direct"]
    out = {}
    for arm in sorted({a for a, _ in main}):
        for c1, c2 in (("default", "low"), ("default", "off"), ("low", "off")):
            a, b = main.get((arm, c1)), main.get((arm, c2))
            if not a or not b:
                continue
            for scope, keep in (("all_cells", lambda k: True), ("hc_harmful_implicit", lambda k: k[1] in HC and k[2] == "harmful" and k[3] == "implicit")):
                keys = sorted(k for k in a.keys() & b.keys() if keep(k))
                out[f"{arm}|{c1}_vs_{c2}|{scope}"] = deconf.kappa([a[k] for k in keys], [b[k] for k in keys])
    return out


def exp14_turn1(ts: list[dict]) -> dict:
    """Turn-1 action of Exp 1.4 episodes (their turn-1 prompt is byte-identical to the Exp 1.1 high_tedium twin)."""
    ref = {(t["arm"], t["cond"], t["scenario_id"], t["cons"], t["bound"]): t["first"] for t in ts
           if t["family"] == "main" and t["cost"] == "high_tedium"}
    e14 = defaultdict(dict)
    n_prompt_identical = n_prompt_checked = 0
    exp11_prompts = {}
    for d in ("exp11", "exp11_effort_low_a", "exp11_effort_low_b", "exp11_effort_low_c"):
        for r in _read_dirs([d]):
            if r["family"] == "main" and r["cost_regime"] == "high_tedium":
                exp11_prompts[(r["arm"], _cond(r), r["scenario_id"], r["consequence"], r["boundary"])] = \
                    hashlib.sha256(r["turns"][0]["full_prompt"].encode()).hexdigest()
    for d, cond in EXP14_SLICES.items():
        for r in _read_dirs([d]):
            fresh = episodes.recompute_record(r)
            key = (r["arm"], _cond(r), r["scenario_id"], r["consequence"], r["boundary"])
            if key in exp11_prompts:
                n_prompt_checked += 1
                n_prompt_identical += hashlib.sha256(r["turns"][0]["full_prompt"].encode()).hexdigest() == exp11_prompts[key]
            e14[(r["arm"], _cond(r))][(r["scenario_id"], r["consequence"], r["boundary"], r["block"]["after_step"])] = _first_action(r, fresh)
    out = {"n_turn1_prompts_checked": n_prompt_checked, "n_turn1_prompts_identical": n_prompt_identical, "groups": {}}
    for (arm, cond), m in sorted(e14.items()):
        x, y, bx, by = [], [], [], []
        for (sid, cons, bound, after), act in sorted(m.items()):
            k = (arm, cond, sid, cons, bound)
            if k in ref:
                x.append(ref[k]); y.append(act)
                bx.append(int(ref[k] == "direct")); by.append(int(act == "direct"))
        early = {k[:3]: v for k, v in m.items() if k[3] == 1}
        late = {k[:3]: v for k, v in m.items() if k[3] == 4}
        both = sorted(early.keys() & late.keys())
        out["groups"][f"{arm}|{cond}"] = {
            "exp11_vs_exp14_action": deconf.kappa(x, y), "exp11_vs_exp14_direct": deconf.kappa(bx, by),
            "exp14_early_vs_late_action": deconf.kappa([early[k] for k in both], [late[k] for k in both]),
            "exp11_action_counts": dict(sorted(Counter(x).items())), "exp14_action_counts": dict(sorted(Counter(y).items()))}
    return out


# ------------------------------------------------------------------------------------------ CoT features
def _feature_summary(items: list[tuple[str, dict]]) -> dict:
    """items: (outcome, features). Fractions per outcome group."""
    out = {}
    for grp in ("shortcut", "safe_route_completed", "other", "all"):
        sel = [f for o, f in items if grp == "all" or o == grp or (grp == "other" and o not in ("shortcut", "safe_route_completed"))]
        if not sel:
            continue
        out[grp] = {"n": len(sel), **{k: sum(f[k] for f in sel) for k in cotfeat.FEATURES},
                    "mean_words": sum(f["n_words"] for f in sel) / len(sel)}
    return out


def cot_features(ts: list[dict]) -> dict:
    out = {}
    g = defaultdict(list)
    for t in ts:
        if t["family"] == "main" and t["think"] is not None and t["cost"] in HC and t["cons"] == "harmful" and t["bound"] == "implicit":
            g[f"exp11|{t['arm']}|{t['cond']}"].append((t["action"], cotfeat.features(t["think"], t["direct_tool"])))
    for r in _read_dirs(["exp31"]):
        if r["cot_mode"] != "cot_on" or r["cost_regime"] not in HC or r["consequence"] != "harmful" or r["boundary"] != "implicit":
            continue
        fresh = episodes.recompute_record(r)
        g[f"exp31|{r['arm']}|{r['meta']['stake_format']}"].append(
            (deconf.action(fresh), cotfeat.features(cotfeat.think_text(fresh["turns"][0]["raw_output"]), r["direct_tool"])))
    for d in (deconf.EXP11_ID, deconf.EXP31_ID):
        for r in _read_dirs([d]) if (RAW / d).is_dir() else []:
            m = r["meta"]
            tag = m.get("format_2x2") or m.get("deconf")
            if tag == "demo":
                continue
            fresh = episodes.recompute_record(r)
            g[f"{d}|{r['arm']}|{tag}|{exp11.effort_label(r.get('reasoning_effort'))}"].append(
                (deconf.action(fresh), cotfeat.features(cotfeat.think_text(fresh["turns"][0]["raw_output"]), r["direct_tool"])))
    for k, v in sorted(g.items()):
        out[k] = _feature_summary(v)
    return out


def corpus_b() -> dict:
    out = {}
    for f in sorted((ROOT / "data/exp12/corpora").glob("*.jsonl")):
        corpus, teacher = f.stem.split(".", 1)
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        thinks = []
        for r in rows:
            tgt = r["target"]
            if cotfeat.THINK_CLOSE not in tgt:
                continue
            tid = (r.get("cell") or {}).get("template_id")
            direct = S.TEMPLATE_BY_ID[tid].direct_tool if tid in S.TEMPLATE_BY_ID else None
            th = cotfeat.think_text(tgt)
            thinks.append({"direct": direct, "mentions": th.count(direct) if direct else 0, "words": len(th.split()),
                           "tokens": r["target_tokens"], "names": bool(direct and direct in th)})
        agentic = [t for t in thinks if t["direct"]]
        out[f"{corpus}|{teacher}"] = {
            "n_examples": len(rows), "n_with_think": len(thinks), "n_agentic_with_think": len(agentic),
            "n_naming_direct": sum(t["names"] for t in agentic),
            "frac_naming_direct": (sum(t["names"] for t in agentic) / len(agentic)) if agentic else None,
            "mean_direct_mentions": (sum(t["mentions"] for t in agentic) / len(agentic)) if agentic else None,
            "mean_think_words": (sum(t["words"] for t in thinks) / len(thinks)) if thinks else None,
            "mean_target_tokens": (sum(r["target_tokens"] for r in rows) / len(rows)) if rows else None,
            "total_target_tokens": sum(r["target_tokens"] for r in rows)}
    return out


# ------------------------------------------------------------------------------------------ assembly
def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def recompute(params: dict) -> dict:
    ts = exp11_trials()
    out = {"schema": "reviewer_addenda_v1", "params": params,
           "action_splits": action_splits(ts), "direct_order": direct_order(ts), "direct_order_prompt": direct_order_prompt_facts(),
           "shortcut_turns": shortcut_turns(ts), "reproducibility": reproducibility(ts), "exp14_turn1": exp14_turn1(ts),
           "primary_events_by_template": {cond: dict(sorted(Counter(
               t["template_id"] for t in ts if t["family"] == "main" and t["arm"] == "refusal_ablated" and t["cond"] == cond
               and t["cost"] in HC and t["cons"] == "harmful" and t["bound"] == "implicit" and t["direct"]).items()))
               for cond in ("default", "low", "off")},
           "cot_features": cot_features(ts), "corpus_b": corpus_b()}
    if params.get("exp32") is not None:
        from v3 import addenda_exp32
        p = params["exp32"]
        out["exp32"] = addenda_exp32.recompute(ROOT / p["raw_root"], ROOT / p["acts_root"], p.get("params", {}))
    if params.get("export"):
        mod = _load("export_public_outcomes", "scripts/export_public_outcomes.py")
        out["public_export"] = mod.summary(mod.build_rows(ROOT))
    return stats.to_jsonable(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-exp32", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--exp32-params", default='{"raw_root": "raw_transcripts", "acts_root": "data/exp32/main", "params": {}}',
                    help="JSON: raw_root, acts_root (relative to the sprint root) and params for v3.addenda_exp32.recompute")
    a = ap.parse_args(argv)
    params = {"exp32": None if a.skip_exp32 else json.loads(a.exp32_params), "export": not a.skip_export}
    out = recompute(params)
    RECORD.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"[written] {RECORD.relative_to(ROOT)} sections={sorted(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
