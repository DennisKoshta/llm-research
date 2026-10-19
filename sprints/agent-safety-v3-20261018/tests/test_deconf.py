import json

import pytest

from v3 import deconf, episodes, exp31, scenarios as S
from v3.sandbox import Sandbox


@pytest.fixture(scope="module")
def bases():
    return S.make_placeholder_bases(40)


def test_exp11_rows_counts_and_conditions(bases):
    rows = deconf.build_exp11_rows(bases)
    by = {}
    for r in rows:
        by.setdefault((r["meta"]["condition"], r["meta"]["effort"]), []).append(r)
    assert {k: len(v) for k, v in by.items()} == {("std", "default"): 80, ("nh", "default"): 80, ("np", "default"): 80,
                                                  ("std", "low"): 80, ("nh", "low"): 80, ("demo", "default"): 3}
    assert all(r["arm"] == "intact" for r in by[("demo", "default")])
    grid = {i["trial_id"]: i for i in S.build_exp11_main_grid(bases)}
    for r in by[("std", "default")]:
        inst = r["instance"]
        assert inst["messages"] == grid[inst["meta"]["base_trial_id"]]["messages"]
    for r in by[("nh", "low")]:
        inst = r["instance"]
        assert S.ESCALATE_TOOL not in json.dumps(inst["messages"]) and S.ESCALATE_TOOL not in inst["declared_names"]
        assert r["max_new_tokens"] == 4096 and r["reasoning_effort"] == "low"
    for r in by[("np", "default")]:
        assert S.PRESSURE not in r["instance"]["messages"][0]["content"]
        assert S.NORM in r["instance"]["messages"][0]["content"]


def test_nh_sandbox_rejects_escalate(bases):
    inst = deconf.apply_condition(deconf.exp11_instance(bases[0], "high_tedium"), "nh")
    box = Sandbox(inst)
    assert box.call(S.ESCALATE_TOOL, {"reason": "x"})["status"] == "ERR_UNKNOWN_TOOL"
    assert not box.done


def test_exp31_2x2_formats():
    bases = exp31.placeholder_bases(40)
    verdicts = exp31.placeholder_verdicts(bases)
    ids = [b["scenario_id"] for b in exp31.select_bases(bases, verdicts, per_domain=8)]
    insts = deconf.exp31_instances(bases, verdicts, ids)
    assert len(insts) == 32 * 2 * 2 * 4
    by = {}
    for i in insts:
        by.setdefault(i["meta"]["pair_id"], {})[i["meta"]["format_2x2"]] = i
    pair = next(iter(by.values()))
    user = {f: pair[f]["messages"][1]["content"] for f in deconf.FORMATS_2X2}
    assert "Stakes table:" in user["concrete_table"] and "Stakes table:" in user["abstract_table"]
    assert "Stakes table:" not in user["abstract_sentence"] and "collateral damage to the workspace." in user["abstract_sentence"]
    tele = next(b for b in bases if b["scenario_id"] == pair["concrete_sentence"]["scenario_id"])["telemetry_harmful"]
    assert tele in user["concrete_table"] and tele in user["concrete_sentence"]
    assert tele not in user["abstract_table"] and tele not in user["abstract_sentence"]
    assert all(p["concrete_sentence"]["messages"][0] == p["abstract_table"]["messages"][0] for p in by.values())


def test_action_and_kappa():
    assert deconf.action({"route": "SAFE_ROUTE", "sub_status": "BATCH_COMPLETED_STAGED"}) == "safe_route_completed"
    assert deconf.action({"route": "TRUNCATED_OR_MALFORMED", "sub_status": "TEXT_NO_CALL"}) == "prose_stop"
    assert deconf.action({"route": "TRUNCATED_OR_MALFORMED", "sub_status": "MAX_TURNS"}) == "unscorable"
    k = deconf.kappa([1, 0, 1, 0], [1, 0, 1, 0])
    assert k["kappa"] == 1.0 and k["agreement"] == 1.0
    assert deconf.kappa([1, 1, 0, 0], [0, 0, 1, 1])["kappa"] == -1.0
    m = deconf.cluster_mean([1, 0, 1, 1], ["a", "a", "b", "b"], n_boot=200)
    assert m["estimate"] == 0.75 and m["n_clusters"] == 2


def test_demo_is_not_authored_text():
    for inst in deconf.demo_instances():
        assert inst["consequence"] == "harmless"
        assert deconf.DEMO_BASE["telemetry_harmful"] not in json.dumps(inst["messages"])
        episodes.check_decoding("cot_on", None, deconf.CAPS["default"])
