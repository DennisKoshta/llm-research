"""Model-agnostic lock-step episode runner and deterministic record replay.

`run_episodes` advances every still-active episode one turn per round with ONE batched
`generate_fn` call (never one call per episode). Each output is parsed with `v3.parse.parse_turn`
(Invariant I2) and stepped through a `v3.sandbox.Sandbox`. History only ever carries the post-think
text of an assistant turn (never the reasoning) plus `"Sandbox observation: " + json.dumps(obs)`.

Episode end rules (PLAN.md):
  * `TRUNCATED_IN_THINK` / `TRUNCATED_IN_CALL` / `finish_reason == "length"` -> TRUNCATED_OR_MALFORMED,
    nothing in that turn is executed (so a capped output can never count as a shortcut). Any other
    non-"stop" finish_reason (a backend's "abort", None...) is treated the same way (FINISH_UNEXPECTED);
  * closed think but zero calls -> TRUNCATED_OR_MALFORMED / `TEXT_NO_CALL` (no refusal-substring detection);
  * only malformed blocks, no valid call -> TRUNCATED_OR_MALFORMED / `MALFORMED_CALL`;
  * valid calls run in order, stopping at the first terminal event; a malformed block next to valid
    calls does not stop them (`parsed.n_malformed` is recorded);
  * still running after `max_turns` -> TRUNCATED_OR_MALFORMED / `MAX_TURNS`.

`recompute_record` / `check_record` re-derive every derived field from the stored raw strings and the
stored instance alone (Invariant I3).
"""
from __future__ import annotations

import copy
import json
import re

from v3.parse import MALFORMED_CALL, NO_TOOL_CALL, THINK_CLOSE, TRUNCATED_IN_CALL, TRUNCATED_IN_THINK, ParsedTurn, parse_turn
from v3.sandbox import IGNORED_AFTER_TERMINAL, Sandbox

MIN_NEW_TOKENS = {"cot_on": 2048, "cot_off": 512}   # Invariant I1
EFFORTS = (None, "low", "medium", "xhigh")
BANNED_PROMPT_LITERALS = ("</think>", "DECLARED_TOOL_NAME")   # PLAN hard rule 5
DERIVED_EPISODE_FIELDS = ("turns", "route", "outcome_category", "sub_status", "direct_taken",
                          "undeclared_direct_attempt", "n_turns", "total_generated_tokens")
DERIVED_SINGLE_FIELDS = ("think_closed", "parsed", "decision")
DECISION_FAMILY = "direct_instruction"    # the only single-mode family scored by the Sandbox
OBSERVATION_PREFIX = "Sandbox observation: "
_GEN_KEYS = ("raw_output", "generated_tokens", "finish_reason", "full_prompt")
_TRAILING_EOS = re.compile(r"(\s|<\|im_end\|>|<\|endoftext\|>)+$")


# ---- validation ------------------------------------------------------------------------------
def check_decoding(cot_mode: str, reasoning_effort, max_new_tokens: int, allow_short: bool = False) -> None:
    """Invariant I1: >=2048 new tokens on cot_on, >=512 on cot_off (`allow_short` is for unit tests only)."""
    if cot_mode not in MIN_NEW_TOKENS:
        raise ValueError(f"cot_mode must be 'cot_on' or 'cot_off', got {cot_mode!r}")
    if reasoning_effort not in EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {EFFORTS}, got {reasoning_effort!r}")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be a positive int, got {max_new_tokens!r}")
    if max_new_tokens < MIN_NEW_TOKENS[cot_mode] and not allow_short:
        raise ValueError(f"I1 violated: max_new_tokens={max_new_tokens} < {MIN_NEW_TOKENS[cot_mode]} for {cot_mode}")


def _check_item(item: dict, expected: dict) -> None:
    """Prompt hygiene + no forced prefix (history must end on a user turn) + group consistency."""
    msgs = item.get("messages")
    if not msgs or msgs[-1].get("role") != "user":
        raise ValueError(f"{item.get('trial_id')!r}: messages must end with a user turn (no forced prefix)")
    for m in msgs:
        for lit in BANNED_PROMPT_LITERALS:
            if lit in m.get("content", ""):
                raise ValueError(f"{item.get('trial_id')!r}: prompt contains banned literal {lit!r}")
    for k, v in expected.items():
        if k in item and item[k] != v:
            raise ValueError(f"{item.get('trial_id')!r}: item {k}={item[k]!r} conflicts with run {k}={v!r}")


def _generate(generate_fn, reqs: list) -> list:
    outs = generate_fn(reqs)
    if len(outs) != len(reqs):
        raise RuntimeError(f"generate_fn returned {len(outs)} outputs for {len(reqs)} requests")
    for o in outs:
        missing = [k for k in _GEN_KEYS if k not in o]
        if missing:
            raise RuntimeError(f"generation output lacks {missing}")
    return outs


# ---- shared deterministic pieces (used by both the live run and the replay) -------------------
def assistant_content(post_think_text: str) -> str:
    """History content for an assistant turn: post-think text only, stripped (trailing EOS markers
    are dropped too so the chat template does not double them)."""
    return _TRAILING_EOS.sub("", post_think_text.strip())


def observation_message(observations: list) -> str:
    """User message carrying the sandbox result: a dict for one call, a list for several.
    '<' is JSON-escaped so a model-chosen tool name echoed back (ERR_UNKNOWN_TOOL) can never put a
    literal '</think>' or tool-call tag into a later prompt (PLAN hard rule 5); the JSON stays valid."""
    body = json.dumps(observations[0] if len(observations) == 1 else observations)
    return OBSERVATION_PREFIX + body.replace("<", "\\u003c")


def parse_output(raw_output: str, cot_on: bool) -> tuple:
    """`(ParsedTurn, its dict)` via v3.parse. A tool-call body nested thousands deep makes the json module
    (or dataclasses.asdict) raise RecursionError and would kill a whole paid run; it is scored like any
    other undecodable block (MALFORMED_CALL, nothing executes). Live run and replay share this path."""
    try:
        parsed = parse_turn(raw_output, cot_on)
        return parsed, parsed.to_dict()
    except RecursionError:
        parsed = ParsedTurn(MALFORMED_CALL, True, [], 1, raw_output.count(THINK_CLOSE), "")
        return parsed, parsed.to_dict()


def apply_turn(box: Sandbox, parsed: ParsedTurn, finish_reason: str) -> list:
    """Step the sandbox with one parsed turn; return its observations ([] when the turn ended the
    episode without executing anything)."""
    if parsed.status in (TRUNCATED_IN_THINK, TRUNCATED_IN_CALL):
        box.truncate(parsed.status)
        return []
    if finish_reason != "stop":     # "length", or anything a backend invents ("abort", None): never executed
        box.truncate("FINISH_LENGTH" if finish_reason == "length" else "FINISH_UNEXPECTED")
        return []
    if not parsed.calls:
        box.truncate("TEXT_NO_CALL" if parsed.status == NO_TOOL_CALL else parsed.status)
        return []
    return box.run_turn(parsed.calls)


def derive_turn(box: Sandbox, raw: dict, cot_on: bool, max_turns: int) -> dict:
    """Parse one stored generation and step `box`; returns the derived turn fields."""
    parsed, parsed_dict = parse_output(raw["raw_output"], cot_on)
    observations = apply_turn(box, parsed, raw["finish_reason"])
    if not box.done and raw["turn"] >= max_turns:
        box.truncate("MAX_TURNS")
    return {"think_closed": parsed.think_closed, "parsed": parsed_dict, "observations": observations,
            "ignored_after_terminal": sum(o["status"] == IGNORED_AFTER_TERMINAL for o in observations)}


def rebuild_messages(record: dict, turn: int) -> list:
    """Messages the model saw when generating `turn` (1-indexed), rebuilt from a stored episode record."""
    messages = [dict(m) for m in record["messages"]]
    for t in record["turns"][: turn - 1]:
        messages.append({"role": "assistant", "content": assistant_content(t["parsed"]["post_think_text"])})
        messages.append({"role": "user", "content": observation_message(t["observations"])})
    return messages


# ---- episodes -----------------------------------------------------------------------------------
def run_episodes(generate_fn, instances: list, *, arm: str, exp_id: str, cot_mode: str, reasoning_effort,
                 max_new_tokens: int, max_turns: int = 10, allow_short: bool = False) -> list:
    """Run multi-turn sandbox episodes in lock-step. One `generate_fn(reqs)` call per round over the
    still-active episodes only. Returns one episode record per instance, in input order."""
    check_decoding(cot_mode, reasoning_effort, max_new_tokens, allow_short)
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")
    ids = [i["trial_id"] for i in instances]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate trial_id in instances")
    expected = {"arm": arm, "exp_id": exp_id, "cot_mode": cot_mode,
                "reasoning_effort": reasoning_effort, "max_new_tokens": max_new_tokens}
    for inst in instances:
        _check_item(inst, expected)
    cot_on = cot_mode == "cot_on"
    boxes = [Sandbox(inst) for inst in instances]
    messages = [[dict(m) for m in inst["messages"]] for inst in instances]
    turns: list = [[] for _ in instances]
    for turn in range(1, max_turns + 1):
        active = [i for i, b in enumerate(boxes) if not b.done]
        if not active:
            break
        reqs = [{"messages": [dict(m) for m in messages[i]], "cot_mode": cot_mode,
                 "reasoning_effort": reasoning_effort, "max_new_tokens": max_new_tokens,
                 "key": f"{ids[i]}:turn{turn}"} for i in active]
        for i, out in zip(active, _generate(generate_fn, reqs)):
            raw = {"turn": turn, "full_prompt": out["full_prompt"], "raw_output": out["raw_output"],
                   "generated_tokens": int(out["generated_tokens"]), "finish_reason": out["finish_reason"]}
            rec = {**raw, **derive_turn(boxes[i], raw, cot_on, max_turns)}
            turns[i].append(rec)
            if not boxes[i].done:
                messages[i].append({"role": "assistant", "content": assistant_content(rec["parsed"]["post_think_text"])})
                messages[i].append({"role": "user", "content": observation_message(rec["observations"])})
    return [{**copy.deepcopy(inst), **expected, "max_turns": max_turns, "turns": turns[i], **boxes[i].result(),
             "n_turns": len(turns[i]), "total_generated_tokens": sum(t["generated_tokens"] for t in turns[i])}
            for i, inst in enumerate(instances)]


# ---- single-decision items (direct-instruction positive control, QA probes) ---------------------
def derive_single(item: dict, raw_output: str, finish_reason: str, cot_on: bool) -> dict:
    """Parse one single-shot generation. `decision` is the Sandbox(mode='single') classification for
    `family == "direct_instruction"` items, else None: QA probes also carry direct_tool/declared_names
    (scenarios.py), but a probe answer is not a sandbox decision, so scorers interpret those."""
    parsed, parsed_dict = parse_output(raw_output, cot_on)
    decision = None
    if item.get("family") == DECISION_FAMILY:
        box = Sandbox(item, mode="single")
        observations = apply_turn(box, parsed, finish_reason)
        if not box.done:
            box.truncate("NO_DECISION")
        decision = {"observations": observations, **box.result()}
    return {"think_closed": parsed.think_closed, "parsed": parsed_dict, "decision": decision}


def run_single(generate_fn, items: list, *, arm: str, exp_id: str, cot_mode: str, reasoning_effort,
               max_new_tokens: int, allow_short: bool = False) -> list:
    """One batched generation over `kind == "single"` items; returns one record per item."""
    check_decoding(cot_mode, reasoning_effort, max_new_tokens, allow_short)
    ids = [i["trial_id"] for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate trial_id in items")
    expected = {"arm": arm, "exp_id": exp_id, "cot_mode": cot_mode,
                "reasoning_effort": reasoning_effort, "max_new_tokens": max_new_tokens}
    for it in items:
        _check_item(it, expected)
    if not items:
        return []
    reqs = [{"messages": [dict(m) for m in it["messages"]], "cot_mode": cot_mode,
             "reasoning_effort": reasoning_effort, "max_new_tokens": max_new_tokens, "key": it["trial_id"]}
            for it in items]
    records = []
    for it, out in zip(items, _generate(generate_fn, reqs)):
        raw = {"full_prompt": out["full_prompt"], "raw_output": out["raw_output"],
               "generated_tokens": int(out["generated_tokens"]), "finish_reason": out["finish_reason"]}
        records.append({**copy.deepcopy(it), **expected, **raw,
                        **derive_single(it, raw["raw_output"], raw["finish_reason"], cot_mode == "cot_on")})
    return records


# ---- replay (Invariant I3) ----------------------------------------------------------------------
def recompute_record(record: dict) -> dict:
    """Re-derive every derived field of an episode (or single) record from its stored raw strings
    (`raw_output`, `finish_reason`, `generated_tokens`, `full_prompt`) and its stored instance fields:
    re-parse with v3.parse, re-step a fresh Sandbox. Returns a dict keyed by `DERIVED_EPISODE_FIELDS`
    (or `DERIVED_SINGLE_FIELDS` for single records). Raises ValueError on a structurally impossible
    record (turn after a terminal event, or no terminal event)."""
    cot_on = record["cot_mode"] == "cot_on"
    if "turns" not in record:
        return derive_single(record, record["raw_output"], record["finish_reason"], cot_on)
    max_turns = record.get("max_turns", 10)
    box = Sandbox(record)
    turns = []
    for n, stored in enumerate(record["turns"], 1):
        if box.done:
            raise ValueError(f"{record['trial_id']}: turn {n} follows a terminal event")
        if stored["turn"] != n:
            raise ValueError(f"{record['trial_id']}: turn numbering broken at {n}")
        raw = {k: stored[k] for k in ("turn", "full_prompt", "raw_output", "generated_tokens", "finish_reason")}
        turns.append({**raw, **derive_turn(box, raw, cot_on, max_turns)})
    if not box.done:
        raise ValueError(f"{record['trial_id']}: record has no terminal event")
    return {"turns": turns, **box.result(), "n_turns": len(turns),
            "total_generated_tokens": sum(t["generated_tokens"] for t in turns)}


def _same(a, b) -> bool:
    """Equality on canonical JSON, so 1 / 1.0 / True (equal in Python, different on disk) are not conflated."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def check_record(record: dict) -> list:
    """Names of derived fields whose stored value differs from the recomputed one ([] means I3 holds).
    A record that cannot be replayed (missing keys, wrong types, turn after a terminal event, no terminal
    event) yields ["INVALID_RECORD: ..."] instead of raising, so a verifier can count it."""
    try:
        fresh = recompute_record(record)
        return [k for k, v in fresh.items() if not _same(record.get(k), v)]
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        return [f"INVALID_RECORD: {type(e).__name__}: {e}"]
