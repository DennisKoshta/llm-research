import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_report_claims", ROOT / "scripts" / "check_report_claims.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

R = {"rec": {"a": {"rate": 0.0999, "k": 8, "xs": [1, 2, 3]}}}


def test_claim_matches_after_rounding_and_scale():
    ok, _ = mod.check_claim({"id": "x", "value": 10.0, "expr": "R['rec']['a']['rate']", "scale": 100, "decimals": 1}, R)
    assert ok
    ok, msg = mod.check_claim({"id": "x", "value": 10.1, "expr": "R['rec']['a']['rate']", "scale": 100, "decimals": 1}, R)
    assert not ok and "10.0" in msg


def test_counts_and_comprehensions_work():
    assert mod.check_claim({"id": "c", "value": 6, "expr": "sum(R['rec']['a']['xs'])", "decimals": 0}, R)[0]
    assert mod.check_claim({"id": "c", "value": 2, "expr": "len([x for x in R['rec']['a']['xs'] if x > 1])", "decimals": 0}, R)[0]


def test_unsafe_and_broken_expressions_are_rejected_not_executed():
    for bad in ("__import__('os').system('true')", "R['rec'].__class__", "open('/etc/passwd')", "R['missing']['x']"):
        ok, msg = mod.check_claim({"id": "b", "value": 1, "expr": bad, "decimals": 0}, R)
        assert not ok and "cannot evaluate" in msg
