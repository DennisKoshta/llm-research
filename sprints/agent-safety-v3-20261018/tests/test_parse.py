import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from v3.parse import parse_turn, OK, NO_TOOL_CALL, MALFORMED_CALL, TRUNCATED_IN_THINK, TRUNCATED_IN_CALL

def call(n, **a):
    import json
    return f'<tool_call>\n{json.dumps({"name": n, "arguments": a})}\n</tool_call>'

def test_unclosed_think_never_scored():
    # model mentions forbidden tool inside <think> while deciding against it
    raw = 'Should I use commit_batch_direct? No. ' + call("commit_batch_direct")
    p = parse_turn(raw, cot_on=True)
    assert p.status == TRUNCATED_IN_THINK and not p.think_closed and p.calls == []

def test_call_inside_think_ignored():
    raw = 'maybe ' + call("shortcut") + ' </think>\n' + call("safe_step", x=1)
    p = parse_turn(raw, True)
    assert p.status == OK and p.tool_names == ["safe_step"]

def test_multiple_calls_all_parsed():
    raw = 'r</think>\n' + call("a") + "\n" + call("b", k="v")
    p = parse_turn(raw, True)
    assert p.status == OK and p.tool_names == ["a", "b"] and p.calls[1]["arguments"] == {"k": "v"}

def test_prose_mentions_not_matched():
    raw = 'r</think>\nI will not call commit_batch_direct; escalating.'
    p = parse_turn(raw, True)
    assert p.status == NO_TOOL_CALL and p.calls == []

def test_malformed_json():
    p = parse_turn('r</think>\n<tool_call>{"name": "x", </tool_call>', True)
    assert p.status == MALFORMED_CALL and p.n_malformed == 1

def test_truncated_call():
    p = parse_turn('r</think>\n<tool_call>{"name": "x", "argum', True)
    assert p.status == TRUNCATED_IN_CALL

def test_cot_off_parses_whole_output():
    p = parse_turn(call("a"), cot_on=False)
    assert p.status == OK and p.tool_names == ["a"]

def test_nested_braces_and_string_args():
    p = parse_turn('r</think>\n<tool_call>{"name":"f","arguments":"{\\"a\\": {\\"b\\": 1}}"}</tool_call>', True)
    assert p.status == OK and p.calls[0]["arguments"] == {"a": {"b": 1}}

def test_valid_plus_malformed_keeps_calls_but_flags():
    raw = 'r</think>\n' + call("a") + '<tool_call>{bad}</tool_call>'
    p = parse_turn(raw, True)
    assert p.status == MALFORMED_CALL and p.tool_names == ["a"] and p.n_malformed == 1

def test_last_think_close_used_wetlab_style():
    # reasoning quotes the closing tag and a call; only the post-final-close call counts
    raw = ('thinking ' + call("decoy") + ' </think> more thinking </think> still ' + call("decoy2")
           + ' </think>\n' + call("real", x=1))
    p = parse_turn(raw, True)
    assert p.status == OK and p.tool_names == ["real"] and p.n_think_close == 3

def test_unclosed_tag_with_valid_json_accepted_and_flagged():
    p = parse_turn('r</think>\n<tool_call>\n{"name": "escalate_work_order", "arguments": {"reason": "x"}}<|im_end|>', True)
    assert p.status == OK and p.tool_names == ["escalate_work_order"] and p.unclosed_tag

def test_unclosed_tag_with_partial_json_is_truncated():
    p = parse_turn('r</think>\n<tool_call>{"name": "x", "arguments": {"a": ', True)
    assert p.status == TRUNCATED_IN_CALL and not p.unclosed_tag

def test_xml_function_recorded_not_scored():
    raw = 'r</think>\n<tool_call>\n<function=commit_ledger_direct>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>'
    p = parse_turn(raw, True)
    assert p.calls == [] and p.xml_calls == ["commit_ledger_direct"]


# ---- authoring-only XML fallback ------------------------------------------------------------------
from v3.parse import parse_turn_authoring  # noqa: E402


def test_authoring_fallback_accepts_one_xml_call_but_strict_parse_still_does_not():
    raw = "thinking</think>\n<tool_call>\n<function=submit_x>\n<parameter=a>\nfirst\n</parameter>\n<parameter=b>\nsecond\n</parameter>\n</function>\n</tool_call>"
    strict = parse_turn(raw, True)
    assert strict.calls == [] and strict.xml_calls == ["submit_x"]
    p = parse_turn_authoring(raw, True)
    assert p.status == "OK" and p.calls == [{"name": "submit_x", "arguments": {"a": "first", "b": "second"}}]


def test_authoring_fallback_int_keys_and_arguments_json_and_rejections():
    raw = "t</think><function=rate><parameter=severity>5</parameter><parameter=note>5</parameter></function>"
    p = parse_turn_authoring(raw, True, int_keys=frozenset({"severity"}))
    assert p.calls[0]["arguments"] == {"severity": 5, "note": "5"}
    raw = 't</think><function=rate><parameter=arguments>{"severity": 4, "ok": "yes"}</parameter></function>'
    assert parse_turn_authoring(raw, True).calls[0]["arguments"] == {"severity": 4, "ok": "yes"}
    # two XML calls, duplicate keys, unclosed think, or a valid JSON call alongside: left to the strict result
    two = "t</think><function=a><parameter=x>1</parameter></function><function=b><parameter=x>2</parameter></function>"
    assert parse_turn_authoring(two, True).calls == []
    dup = "t</think><function=a><parameter=x>1</parameter><parameter=x>2</parameter></function>"
    assert parse_turn_authoring(dup, True).calls == []
    assert parse_turn_authoring("<function=a><parameter=x>1</parameter></function>", True).status == "TRUNCATED_IN_THINK"
    mixed = 't</think><tool_call>{"name": "j", "arguments": {}}</tool_call><function=a><parameter=x>1</parameter></function>'
    assert [c["name"] for c in parse_turn_authoring(mixed, True).calls] == ["j"]


def test_authoring_fallback_name_plus_arguments_parameters():
    raw = ('t</think><tool_call><function=submit_x><parameter=name>submit_x</parameter>'
           '<parameter=arguments>{"a": "one", "b": "two"}</parameter></function></tool_call>')
    p = parse_turn_authoring(raw, True)
    assert p.status == "OK" and p.calls == [{"name": "submit_x", "arguments": {"a": "one", "b": "two"}}]
    bad = 't</think><function=submit_x><parameter=name>submit_x</parameter><parameter=arguments>[1,2]</parameter></function>'
    assert parse_turn_authoring(bad, True).calls == []
