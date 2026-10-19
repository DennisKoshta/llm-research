"""Strict post-</think> tool-call parser (Invariant I2).

No substring matching on tool names or prose. A turn is only ever scored from
`<tool_call>{json}</tool_call>` blocks that appear AFTER the closing `</think>`
(cot_on) or anywhere in the output (cot_off, where the template pre-closes an
empty think block). Multiple tool calls in one turn are all parsed.

Two documented leniencies (both flagged on the result, neither involves substring matching):
  * `unclosed_tag`: `<tool_call>{complete valid JSON}` followed only by whitespace / EOS tokens and
    no `</tool_call>`. Run 2 found 3/133 ablated vs 0/133 intact outputs ending this way, so
    rejecting them would bias arm comparisons.
  * `xml_calls`: Qwen-native `<function=NAME>` blocks (Flash-Next-Uncensored emits these). They are
    recorded for sensitivity analysis but NEVER counted in `calls` / `status`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

THINK_CLOSE = "</think>"
CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"
_EOS_TAIL = re.compile(r"^(\s|<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|<\|end_of_text\|>)*$")
_XML_FN = re.compile(r"<function=([A-Za-z0-9_\-\.]+)>")

# turn_status values
OK = "OK"                                  # >=1 well-formed call, no malformed blocks
NO_TOOL_CALL = "NO_TOOL_CALL"              # closed think (or cot_off) but zero call blocks
MALFORMED_CALL = "MALFORMED_CALL"          # a block exists but its JSON / schema is invalid
TRUNCATED_IN_THINK = "TRUNCATED_IN_THINK"  # cot_on and no </think>: never scored
TRUNCATED_IN_CALL = "TRUNCATED_IN_CALL"    # opened <tool_call> but output ended before a complete JSON object


@dataclass
class ParsedTurn:
    status: str
    think_closed: bool
    calls: list = field(default_factory=list)       # [{"name": str, "arguments": dict}]
    n_malformed: int = 0
    n_think_close: int = 0
    post_think_text: str = ""
    unclosed_tag: bool = False
    xml_calls: list = field(default_factory=list)   # names only; never scored

    @property
    def tool_names(self) -> list:
        return [c["name"] for c in self.calls]

    @property
    def first_tool_name(self):
        return self.calls[0]["name"] if self.calls else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tool_names"] = self.tool_names
        return d


def _call_from_obj(obj):
    """Return {"name","arguments"} or None if the schema is invalid."""
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):  # some templates stringify arguments
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError, RecursionError):
            return None
    if not isinstance(args, dict):
        return None
    return {"name": obj["name"], "arguments": args}


def _extract_calls(text: str):
    """Return (calls, n_malformed, truncated_open, unclosed_tag)."""
    calls, malformed = [], 0
    pos, truncated, unclosed = 0, False, False
    dec = json.JSONDecoder()
    while True:
        i = text.find(CALL_OPEN, pos)
        if i < 0:
            break
        start = i + len(CALL_OPEN)
        j = text.find(CALL_CLOSE, start)
        if j < 0:
            # No closer. Accept only a complete JSON object followed by EOS-ish whitespace.
            rest = text[start:].lstrip()
            try:
                obj, end = dec.raw_decode(rest)
            except (json.JSONDecodeError, ValueError, RecursionError):
                truncated = True
                break
            call = _call_from_obj(obj) if _EOS_TAIL.match(rest[end:]) else None
            if call is None:
                truncated = True
            else:
                calls.append(call)
                unclosed = True
            break
        body = text[start:j].strip()
        pos = j + len(CALL_CLOSE)
        try:
            obj = json.loads(body)
        except (json.JSONDecodeError, ValueError, RecursionError):
            malformed += 1
            continue
        call = _call_from_obj(obj)
        if call is None:
            malformed += 1
            continue
        calls.append(call)
    return calls, malformed, truncated, unclosed


def parse_turn(raw_output: str, cot_on: bool) -> ParsedTurn:
    """Parse one generation. `raw_output` excludes the prompt (which already ends in
    `<think>\\n` for cot_on, or `<think>\\n\\n</think>\\n\\n` for cot_off).

    The think/answer split is at the LAST `</think>`: a model that quotes the closing tag while
    reasoning (Run 2's exp23_wetlab_07 had 8 of them) must not leak reasoning-time tool calls
    into the scored answer."""
    n_close = raw_output.count(THINK_CLOSE)
    if cot_on:
        if n_close == 0:
            return ParsedTurn(TRUNCATED_IN_THINK, False, [], 0, 0, "")
        post = raw_output.rpartition(THINK_CLOSE)[2]
    else:
        post = raw_output.rpartition(THINK_CLOSE)[2] if n_close else raw_output
    calls, malformed, truncated, unclosed = _extract_calls(post)
    if truncated and not calls and not malformed:
        status = TRUNCATED_IN_CALL
    elif malformed:
        status = MALFORMED_CALL
    elif truncated:
        status = TRUNCATED_IN_CALL
    elif not calls:
        status = NO_TOOL_CALL
    else:
        status = OK
    xml = _XML_FN.findall(post) if not calls else []
    return ParsedTurn(status, True, calls, malformed, n_close, post, unclosed, xml)


# ---- authoring / judging stages ONLY ----------------------------------------------------------------
_XML_BLOCK = re.compile(r"<function=([A-Za-z0-9_\-\.]+)>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=([A-Za-z0-9_\-\.]+)>\s*(.*?)\s*</parameter>", re.S)
_BARE_INT = re.compile(r"^-?\d+$")


def parse_turn_authoring(raw_output: str, cot_on: bool, int_keys=frozenset()) -> ParsedTurn:
    """parse_turn plus a fallback for the uncensored authoring model's native XML call syntax
    (`<function=NAME><parameter=KEY>VALUE</parameter>...</function>`), which carries the same call content.

    NEVER use this to score model behaviour (shortcut / violation / monitor verdict / agent action): those paths use the
    strict `parse_turn`, which never executes an XML call. This wrapper is for stages that merely *produce or rate text*
    (scenario authoring, severity judging, plan/review text, paraphrases), where an XML-only reply would otherwise discard
    an otherwise valid draft. It only fires when strict parsing found no JSON call at all, the post-think text holds exactly
    one well-formed XML call, and `int_keys` lists the parameters that must be integers (bare-integer strings are converted)."""
    parsed = parse_turn(raw_output, cot_on)
    if parsed.calls or not parsed.think_closed or not parsed.xml_calls:
        return parsed
    blocks = _XML_BLOCK.findall(parsed.post_think_text)
    if len(blocks) != 1:
        return parsed
    name, body = blocks[0]
    pairs = _XML_PARAM.findall(body)
    keys = [k for k, _ in pairs]
    if not pairs or len(set(keys)) != len(keys):
        return parsed
    if set(keys) in ({"arguments"}, {"name", "arguments"}):   # the whole JSON arguments object in one parameter
        if "name" in keys:
            name = dict(pairs)["name"].strip() or name
        try:
            obj = json.loads(dict(pairs)["arguments"])
        except (json.JSONDecodeError, ValueError, RecursionError):
            return parsed
        if not isinstance(obj, dict):
            return parsed
        args = obj
    else:
        args = {k: (int(v) if k in int_keys and _BARE_INT.match(v) else v) for k, v in pairs}
    return ParsedTurn(OK, True, [{"name": name, "arguments": args}], 0, parsed.n_think_close, parsed.post_think_text,
                      False, [])
