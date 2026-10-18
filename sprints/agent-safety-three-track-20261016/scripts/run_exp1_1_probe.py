"""Layer-48 probe on Exp 1.1 (Paper 1 salvage, Task 2).

Pairs the activations from shared/modal_paper1_salvage.py (final prompt token for cot_on / cot_off prompts, and
`</think>` for saved cot_on transcripts) with the strict Exp 1.1 labels from scripts/verify_tracks1_and_2.py
(truncated trials excluded). Evaluates under 5-fold leave-two-templates-out CV:
    probe      StandardScaler + RidgeClassifierCV (alpha by efficient LOO on the training fold)
    r_hat_1d   logistic regression on h·r̂ (Phase 1 layer-48 refusal direction)
    tfidf      TF-IDF (1-2 grams) logistic regression on the prompt text (post: prompt + think text)
    design     one-hot cost × consequence × rule × arm cell
Regimes: pooled per mode; within each arm; within-arm cells that split behaviour.

Usage:
    uv run --no-project --with numpy --with scikit-learn --with tokenizers python scripts/run_exp1_1_probe.py \
        --acts-dir <dir with exp1_1_l48_acts_{intact,refusal_ablated}.npz> --exp1-1-dir <dir with exp1_1_records_*.json> \
        --tokenizer <qwen tokenizer.json> --directions <phase1_directions.npz>
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tokenizers import Tokenizer

SPRINT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_tracks1_and_2 as vt  # noqa: E402

warnings.filterwarnings("ignore")
ARMS = ("intact", "refusal_ablated")
METHODS = ("probe", "r_hat_1d", "tfidf", "design")


def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else None


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", type=Path, required=True)
    ap.add_argument("--exp1-1-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--directions", type=Path, required=True)
    args = ap.parse_args()
    vt.TOK = Tokenizer.from_file(str(args.tokenizer))
    r_hat = np.load(args.directions)["r_hat"].astype(np.float64)
    r_hat /= np.linalg.norm(r_hat)

    factorial = json.loads((SPRINT_ROOT / "data/exp1_1/factorial_360_trials.json").read_text())
    fac = {t["trial_id"]: t for t in factorial}
    templates = sorted({t["template_id"] for t in factorial})
    assert len(templates) == 10
    fold_of = {tp: i // 2 for i, tp in enumerate(templates)}

    rows = []  # one row per (arm, mode, position)
    last_tokens = {}
    for arm in ARMS:
        z = np.load(args.acts_dir / f"exp1_1_l48_acts_{arm}.npz")
        last_tokens[arm] = json.loads(str(z["last_tokens"]))
        idx = {t: i for i, t in enumerate(z["trial_ids"].tolist())}
        pidx = {t: i for i, t in enumerate(z["post_trial_ids"].tolist())}
        recs = json.loads((args.exp1_1_dir / f"exp1_1_records_{arm}.json").read_text())
        for r in recs:
            t = fac[r["trial_id"]]
            mode = r["reasoning_mode"]
            out = vt.strict_tool_outcome(t, r["raw_response"], mode, 768 if mode == "cot_on" else 512)
            if out in vt.TRUNC_KINDS or out == "think_unclosed_no_cap":
                continue
            base = {"arm": arm, "mode": mode, "trial_id": t["trial_id"], "template": t["template_id"],
                    "scenario": t["scenario_id"], "cell": f"{t['cost_regime']}__{t['consequence']}__{t['rule_level']}",
                    "y": int(out == "shortcut"),
                    "prompt": t["messages"][0]["content"] + " " + t["messages"][1]["content"]}
            X_pre = z["X_pre_on"] if mode == "cot_on" else z["X_pre_off"]
            rows.append({**base, "pos": "pre", "x": X_pre[idx[t["trial_id"]]], "text": base["prompt"]})
            if mode == "cot_on" and t["trial_id"] in pidx:
                think = r["raw_response"].split("</think>", 1)[0]
                rows.append({**base, "pos": "post", "x": z["X_post"][pidx[t["trial_id"]]], "text": base["prompt"] + " " + think})

    def oof(sub: list[dict]) -> dict[str, np.ndarray]:
        y = np.array([r["y"] for r in sub])
        X = np.stack([r["x"] for r in sub]).astype(np.float64)
        proj = (X @ r_hat)[:, None]
        cells = sorted({(r["cell"], r["arm"]) for r in sub})
        D = np.array([[1.0 if (r["cell"], r["arm"]) == c else 0.0 for c in cells] for r in sub])
        texts = [r["text"] for r in sub]
        folds = np.array([fold_of[r["template"]] for r in sub])
        preds = {m: np.full(len(sub), np.nan) for m in METHODS}
        for f in range(5):
            te, tr = folds == f, folds != f
            if len(set(y[tr])) < 2:
                for m in METHODS:
                    preds[m][te] = y[tr].mean()
                continue
            sc = StandardScaler().fit(X[tr])
            clf = RidgeClassifierCV(alphas=np.logspace(0, 6, 13))
            preds["probe"][te] = clf.fit(sc.transform(X[tr]), y[tr]).decision_function(sc.transform(X[te]))
            preds["r_hat_1d"][te] = LogisticRegression().fit(proj[tr], y[tr]).predict_proba(proj[te])[:, 1]
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            Ttr = vec.fit_transform([texts[i] for i in np.where(tr)[0]])
            preds["tfidf"][te] = LogisticRegression(C=1.0, max_iter=3000).fit(Ttr, y[tr]).predict_proba(
                vec.transform([texts[i] for i in np.where(te)[0]]))[:, 1]
            preds["design"][te] = LogisticRegression(C=1.0, max_iter=3000).fit(D[tr], y[tr]).predict_proba(D[te])[:, 1]
        return {"y": y, **preds}

    md = []
    w = md.append
    w("### Task 2 detail — Exp 1.1 layer-48 probe (5-fold leave-two-templates-out)\n")
    w(f"Captured last tokens: `{json.dumps(last_tokens)}`\n")
    w("Labels: strict Exp 1.1 re-score (complete `<tool_call>` JSON), truncated trials dropped. "
      "Pooled out-of-fold AUROC; `pos/neg` = shortcut / non-shortcut.\n")
    w("| regime | position | subset | pos/neg | probe | r̂ 1-D | TF-IDF | design cell |")
    w("|---|---|---|---|---|---|---|---|")
    results = {}

    def report(regime, pos, label, sub, restrict=None):
        if not sub:
            return
        o = oof(sub)
        mask = np.ones(len(sub), bool) if restrict is None else np.array([restrict(r) for r in sub])
        y = o["y"][mask]
        if restrict is None:
            vals = {m: auroc(y, o[m][mask]) for m in METHODS}
        else:
            # Remove per-fold offsets: OOF scores from differently calibrated fold models are not comparable
            fm = np.array([fold_of[r["template"]] for r in sub])[mask]
            vals = {}
            for m in METHODS:
                sc_ = o[m][mask].astype(float).copy()
                for f in set(fm):
                    sc_[fm == f] -= sc_[fm == f].mean()
                vals[m] = auroc(y, np.round(sc_, 12))
        results[(regime, pos, label)] = (int(y.sum()), int(len(y) - y.sum()), vals)
        w(f"| {regime} | {pos} | {label} | {int(y.sum())}/{int(len(y) - y.sum())} | " + " | ".join(fmt(vals[m]) for m in METHODS) + " |")

    for mode in ("cot_on", "cot_off"):
        for pos in (("pre", "post") if mode == "cot_on" else ("pre",)):
            sub = [r for r in rows if r["mode"] == mode and r["pos"] == pos]
            report("1 pooled", pos, f"{mode}, both arms", sub)
            for arm in ARMS:
                report("2 within-arm", pos, f"{mode}, {arm}", [r for r in sub if r["arm"] == arm])

    # Regime 3: cells that split behaviour within one arm (>=3 of each class), evaluated two ways
    split_cells = []
    for mode in ("cot_on", "cot_off"):
        for arm in ARMS:
            c = Counter((r["cell"], r["y"]) for r in rows if r["mode"] == mode and r["arm"] == arm and r["pos"] == "pre")
            for cell in sorted({k[0] for k in c}):
                if c[(cell, 1)] >= 3 and c[(cell, 0)] >= 3:
                    split_cells.append((mode, arm, cell, c[(cell, 1)], c[(cell, 0)]))
    w("")
    w("Within-arm cells with ≥3 shortcut and ≥3 non-shortcut trials: " +
      "; ".join(f"`{m}/{a}/{c}` ({p}/{n})" for m, a, c, p, n in split_cells) + ".\n")
    w("| regime | position | subset | pos/neg | probe | r̂ 1-D | TF-IDF | design cell |")
    w("|---|---|---|---|---|---|---|---|")
    for mode, arm, cell, _, _ in split_cells:
        for pos in (("pre", "post") if mode == "cot_on" else ("pre",)):
            arm_rows = [r for r in rows if r["mode"] == mode and r["arm"] == arm and r["pos"] == pos]
            report("3a within-cell (arm-level model)", pos, f"{mode}/{arm}/{cell}", arm_rows, lambda r, c=cell: r["cell"] == c)
    for mode in ("cot_on", "cot_off"):
        cells = [(a, c) for m, a, c, _, _ in split_cells if m == mode]
        if not cells:
            continue
        for pos in (("pre", "post") if mode == "cot_on" else ("pre",)):
            sub = [r for r in rows if r["mode"] == mode and r["pos"] == pos and (r["arm"], r["cell"]) in cells]
            report("3b trained on split cells only", pos, f"{mode}: " + " + ".join(f"{a}/{c}" for a, c in cells), sub)
    w("\n3a scores are centred within each test fold before computing AUROC (removes fold-model offsets); the "
      "design-cell baseline is then 0.5 by construction. "
      "TF-IDF sees only the prompt (pre) or prompt + think text (post).\n")

    out = SPRINT_ROOT / "records" / "exp1_1_probe_salvage_results.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
