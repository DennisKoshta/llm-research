"""Mechanical check that every number quoted in the final report is the value stored in a verified records/*.json file.

  PYTHONPATH=src .venv/bin/python scripts/check_report_claims.py [reports/sections/NAME.claims.json ...]

A claims file is a JSON list of {"id", "text", "value", "expr", "decimals"[, "scale"]}:
  * `expr` is a Python expression over `R`, a dict {record name (file stem of records/*.json): parsed JSON}, e.g.
        "R['exp11_main']['contrasts']['P1.1']['rate_a']"
    Only a small set of builtins is available (len, sum, min, max, abs, round, sorted, any, all, next, float, int, str,
    range, enumerate, zip, list, dict, set, tuple); no imports, no file or network access, no dunder attributes.
  * the claim holds when round(scale * eval(expr), decimals) == value (scale defaults to 1; use 100 for percents,
    1e-0 for counts). `value` is the number exactly as written in the report prose (after the same rounding).
Exit status is non-zero when any claim fails or cannot be evaluated. Prints claim ids and numbers only (never record text).
"""
from __future__ import annotations

import ast
import glob
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                 for n in ("len", "sum", "min", "max", "abs", "round", "sorted", "any", "all", "next", "float", "int", "str",
                           "range", "enumerate", "zip", "list", "dict", "set", "tuple", "bool", "isinstance")}


def load_records() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "records").glob("*.json"))}


def check_expr_safe(expr: str) -> None:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("dunder/private attribute access is not allowed")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are not allowed")


def evaluate(expr: str, records: dict):
    check_expr_safe(expr)
    return eval(compile(ast.parse(expr, mode="eval"), "<claim>", "eval"), {"__builtins__": SAFE_BUILTINS}, {"R": records})


def check_claim(claim: dict, records: dict) -> tuple[bool, str]:
    try:
        got = evaluate(claim["expr"], records)
        scale = float(claim.get("scale", 1))
        dec = int(claim["decimals"])
        shown = round(scale * float(got), dec)
        want = float(claim["value"])
    except Exception as e:  # unreadable claims fail loudly
        return False, f"cannot evaluate ({type(e).__name__}: {str(e)[:80]})"
    ok = math.isclose(shown, want, rel_tol=0, abs_tol=0.5 * 10 ** (-dec) * 1e-6 + 1e-12)
    return ok, f"report says {want} ; record gives {shown}"


def main(argv=None) -> int:
    files = (argv if argv is not None else sys.argv[1:]) or sorted(glob.glob(str(ROOT / "reports" / "sections" / "*.claims.json")))
    if not files:
        print("no claims files found")
        return 2
    records = load_records()
    n_ok = n_bad = 0
    for f in files:
        claims = json.loads(pathlib.Path(f).read_text())
        for c in claims:
            ok, msg = check_claim(c, records)
            n_ok += ok
            n_bad += not ok
            if not ok:
                print(f"[FAIL] {pathlib.Path(f).name}:{c.get('id')}: {msg}")
    print(f"{n_ok} claims OK, {n_bad} failing, across {len(files)} files")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
