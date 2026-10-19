"""Exp 3.2: pre-action probing of layer-48 activations (CPU side: position contract, tables, probes, regimes).

What is probed. The activation at the PRE-THINK position: the last token of the chat-template generation prompt
BEFORE the `<think>` token (the "\\n" after "assistant"). cot_on prompts end `...assistant\\n<think>\\n`, cot_off prompts
end `...assistant\\n<think>\\n\\n</think>\\n\\n`: the POSITION is the same token in both (`check_pre_think_position`
asserts it at text and token level: same last token id, next token `<think>` in both). The CONTEXT in front of it is
not always the same, though. Checked against the real Qwen3.8 chat template (the donors' tokenizer_audit copy): when
thinking is on and the reasoning effort is xhigh (also the default) or low, the template prepends an effort block
("Reasoning effort is set to ...") to the system message; the cot_off render has no such block. So the cot_on and
cot_off pre-think states are states of two different prompts. `relate_prefixes` classifies the pair as "identical"
(effort 'medium': one forward pass serves both modes) or "effort_block" (the cot_off prefix is the cot_on prefix with
exactly that block removed; each mode needs its own forward pass, on the prompt that mode really generated from);
anything else raises. The activations come from each arm's own FP8 checkpoint (v3.modal_hf swap loader, no runtime
ablation hooks); see v3.modal_acts. Sites: "out" = output of layers[48] (Run 1 / Exp 3.2 convention, primary),
"in" = input (where r_hat was fit).

Labels. Strict outcomes of the matching sweeps (Exp 1.1 main grid; optionally Exp 3.1), REPLAYED from raw_output via
`episodes.recompute_record`: y = direct_taken. TRUNCATED_OR_MALFORMED episodes have no label and are dropped
(counted). The same prompt in two sweeps (Exp 3.1 narrative == Exp 1.1 prompts) is one row (Exp 1.1 wins).

Regimes (every probe is evaluated leave-templates-out: the 10 templates are the unit, 5 folds of 2, folds assigned
round-robin over (domain, template) order; every CI is a percentile bootstrap that resamples SCENARIOS)
  A  full grid, per cot mode: standardised Ridge probe on h48 vs the full-interaction design baseline
     arm x cost x consequence x boundary (`design`: saturated cell lookup, the strongest function of those four
     factors), a stronger tabular `design_plus` (cell + domain + tool order, logistic), prompt TF-IDF and the 1-D
     h48.r_hat projection. Pooled out-of-fold AUROC + paired AUROC-difference CIs.
  B  within Intact: the 1-D projection h48.r_hat as predictor. `rhat_neg` is the pre-specified direction (higher
     refusal-direction projection -> fewer shortcuts: score = -projection, nothing fitted); `rhat_cv` learns the sign
     leave-templates-out. Compared with Intact's own cell lookup, a full-activation probe and TF-IDF.
  C  within-cell: only cells whose labels split (>= min_minority of each class among DISTINCT scenarios; a prompt is
     never resampled). Probe features are DEMEANED per cell (arm, cost, consequence, boundary) and the training labels
     are centred per cell, so neither arm nor cell identity reaches the probe; held-out scores are centred per
     (cell, fold). The AUROC is the stratified (same-cell pairs only) concordance, so the design baseline, constant
     inside a cell, is exactly 0.500 by construction. Baselines get the same treatment (TF-IDF demeaned per cell,
     r_hat projection demeaned per cell, and `order_domain`: one-hot tool order + domain demeaned per cell, the layout
     information a pre-think state certainly encodes and that varies inside a cell). Scopes: training pooled over eligible cells of one (arm, cot) [headline],
     of one cot mode (all arms), or per single cell. Optional label-permutation null (labels shuffled inside cells).

CI caveat. The bootstrap resamples scenarios while the out-of-fold scores stay fixed, so it ignores the variance of
the cross-validated fit itself and under-covers a little (simulated null: ~85% of the 95% intervals cover 0.5). The
label-permutation null (`permutation_null.p_perm`, Regime C headline scope) is the calibrated test of "any signal".

`recompute(raw_rows, acts, config)` is the receipt function: raw episode rows + the saved activation arrays ->
summary. `config` (stored in the record) selects the cot_on reasoning effort of the labels ('auto' = the only one
present; it must equal the effort the activations were captured at), the site, folds, bootstrap size, permutation count, `min_minority`,
the datasets ({name: [sources]}), optionally `c_scopes` / `c_methods` (restrict Regime C) and `threads` (BLAS threads, default
4: the ridge fits are 5120 wide with a few hundred to a few thousand rows, and the library default (one thread per core) was
measured 5-8x slower than 4 threads on a loaded 12-core machine. Full size (6 arms x 768 prompts, D=5120, n_boot 2000,
n_perm 50, both datasets and sites): 21 min with threads=4, while the unpinned run was still going after 34 min and was
stopped; results with different thread counts agree to ~1e-12 and are rounded to 6 decimals; None = library default).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

from v3 import arms, episodes, exp31, stats
from v3.sandbox import TRUNCATED_OR_MALFORMED

EXP_ID = "exp32"
LAYER = 48
SITES = ("in", "out")
PRIMARY_SITE = "out"
COT_MODES = ("cot_on", "cot_off")
THINK_OPEN = "<think>\n"
THINK_EMPTY = "<think>\n\n</think>\n\n"
RHAT_KEYS = {"r1": "r_hat", "r2": "refusal"}      # v3.arms.VECTORS file layouts
FORBIDDEN_IN_PROMPTS = ("</think>", "DECLARED_TOOL_NAME")
ALPHAS = np.logspace(0, 6, 13)
DEFAULT_CONFIG = {"effort": "auto", "site": PRIMARY_SITE, "sensitivity_sites": ["in"], "n_folds": 5, "n_boot": 2000, "seed": stats.DEFAULT_SEED,
                  "min_minority": 4, "n_perm": 50, "threads": 4,
                  "datasets": {"exp11": ["exp11"], "exp11+exp31": ["exp11", "exp31"]}}


# ------------------------------------------------------------------------------------------ position contract
EFFORT_BLOCK_START = "Reasoning effort is set to"


def render_generation_prompt(tok, messages: list, cot_on: bool, effort) -> str:
    """The generation prompt exactly as v3.modal_serve._render builds it for Qwen (effort only affects cot_on)."""
    kw = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": cot_on}
    if effort and cot_on:
        kw["reasoning_effort"] = effort
    return tok.apply_chat_template(messages, **kw)


def pre_think_prefix(full_prompt: str, cot_on: bool) -> str:
    """The generation prompt without its trailing think block: everything up to the pre-think position."""
    suffix = THINK_OPEN if cot_on else THINK_EMPTY
    if not full_prompt.endswith(suffix):
        raise ValueError(f"generation prompt does not end with the expected think block (cot_on={cot_on})")
    prefix = full_prompt[: -len(suffix)]
    if "<think>" in prefix or "</think>" in prefix:
        raise ValueError("think token inside the pre-think prefix")
    return prefix


def relate_prefixes(prefix_on: str, prefix_off: str) -> str:
    """'identical', or 'effort_block' when the cot_off prefix is the cot_on prefix minus ONE contiguous reasoning-effort
    block ("Reasoning effort is set to ..." plus its separator). Any other difference raises ValueError."""
    if prefix_on == prefix_off:
        return "identical"
    k, n = prefix_on.find(EFFORT_BLOCK_START), len(prefix_on) - len(prefix_off)
    if k >= 0 and n > 0 and prefix_on[:k] + prefix_on[k + n:] == prefix_off:
        return "effort_block"
    raise ValueError("cot_on and cot_off prompts differ before <think> by more than the reasoning-effort block")


def check_pre_think_position(tok, text_on: str, text_off: str, think_id=None) -> dict:
    """Assert that the pre-think POSITION is the same token in both cot modes and classify the contexts in front of it.

    Per mode the prefix token ids must be a prefix of the full prompt's ids and the next token must be `<think>`; the
    two modes must agree on the last pre-think token id and on the next token id; the prefixes must be related by
    `relate_prefixes`. Raises ValueError (never echoing prompt text); returns the position facts."""
    prefix_on, prefix_off = pre_think_prefix(text_on, True), pre_think_prefix(text_off, False)
    relation = relate_prefixes(prefix_on, prefix_off)

    def ids(text):
        return list(tok(text, add_special_tokens=False)["input_ids"])
    facts = {}
    for mode, prefix, full in (("on", prefix_on, text_on), ("off", prefix_off, text_off)):
        pre, whole = ids(prefix), ids(full)
        n = len(pre)
        if n == 0 or whole[:n] != pre:
            raise ValueError(f"cot_{mode}: token ids of the pre-think prefix are not a prefix of the full prompt tokenisation")
        if len(whole) <= n or (think_id is not None and whole[n] != think_id):
            raise ValueError(f"cot_{mode}: the token after the pre-think prefix is not <think>")
        facts[mode] = {"n": n, "last": pre[-1], "next": whole[n]}
    if facts["on"]["last"] != facts["off"]["last"] or facts["on"]["next"] != facts["off"]["next"]:
        raise ValueError("the pre-think token differs between cot_on and cot_off")
    return {"relation": relation, "n_pre_tokens": facts["on"]["n"], "n_pre_tokens_off": facts["off"]["n"],
            "last_pre_id": facts["on"]["last"], "next_id": facts["on"]["next"]}


def plan_pre_think(tok, prompts: list, effort, think_id=None) -> dict:
    """Render every prompt for both cot modes and check the position contract. Returns, per mode, the prefix texts to
    forward and their token counts, the relation of the two contexts ('identical' for all prompts, else
    'effort_block'), the shared last-token id and a hash of the rendered prompts (identical across checkpoints)."""
    plan: dict = {"prefix_on": [], "prefix_off": [], "n_pre_on": [], "n_pre_off": []}
    relations, last_ids, next_ids, digest = [], set(), set(), hashlib.sha256()
    for p in prompts:
        if any(lit in json.dumps(p["messages"]) for lit in FORBIDDEN_IN_PROMPTS):
            raise ValueError(f"prompt {p['id']} contains a forbidden literal")
        on = render_generation_prompt(tok, p["messages"], True, effort)
        off = render_generation_prompt(tok, p["messages"], False, effort)
        try:
            facts = check_pre_think_position(tok, on, off, think_id)
        except ValueError as e:
            raise ValueError(f"prompt {p['id']}: {e}") from None
        plan["prefix_on"].append(pre_think_prefix(on, True))
        plan["prefix_off"].append(pre_think_prefix(off, False))
        plan["n_pre_on"].append(facts["n_pre_tokens"])
        plan["n_pre_off"].append(facts["n_pre_tokens_off"])
        relations.append(facts["relation"])
        last_ids.add(facts["last_pre_id"])
        next_ids.add(facts["next_id"])
        digest.update(on.encode() + b"\x00" + off.encode() + b"\x00")
    kinds = sorted(set(relations))
    return {**plan, "relation": kinds[0] if len(kinds) == 1 else "mixed", "last_pre_ids": sorted(last_ids),
            "next_ids": sorted(next_ids), "rendered_sha256": digest.hexdigest()}


# ------------------------------------------------------------------------------------------ prompts and files
def prompt_id(messages: list) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


def build_prompt_set(sources: dict) -> tuple[list, dict]:
    """sources: {name: [instance]} -> (prompts [{id, messages}] unique by content, in first-seen order;
    index {id: [{source, trial_id, structure fields}]}). The index carries no prompt text."""
    prompts, index = [], {}
    for name, instances in sources.items():
        for inst in instances:
            pid = prompt_id(inst["messages"])
            if pid not in index:
                index[pid] = []
                prompts.append({"id": pid, "messages": inst["messages"]})
            index[pid].append({"source": name, "trial_id": inst["trial_id"], "template_id": inst["template_id"],
                               "scenario_id": inst["scenario_id"], "domain": inst["domain"], "cost_regime": inst["cost_regime"],
                               "consequence": inst["consequence"], "boundary": inst["boundary"], "order_index": inst["order_index"],
                               "fmt": inst.get("meta", {}).get("stake_format", "narrative")})
    return prompts, index


def load_r_hat(path, key: str, dim: int | None = None) -> np.ndarray:
    """Unit-norm float64 refusal direction from a phase-1 directions npz; the shape is verified (and `dim` if given)."""
    v = np.asarray(np.load(path)[key], dtype=np.float64)
    if v.ndim != 1 or (dim is not None and v.shape[0] != dim):
        raise ValueError(f"r_hat {key!r} has shape {v.shape}, expected ({dim},)")
    if not np.isfinite(v).all() or abs(float(np.linalg.norm(v)) - 1.0) > 1e-6:
        raise ValueError(f"r_hat {key!r} is not a finite unit vector")
    return v


def r_hat_path(family: str) -> str:
    return arms.VECTORS[family]


COT_OFF_STATES = ("same", "separate", "absent")


def save_acts(path, prompt_ids: list, pre_in, pre_out, n_tokens, last_ids, r_hat, meta: dict, *, pre_in_off=None,
              pre_out_off=None, n_tokens_off=None) -> str:
    """Write one arm's float32 pre-think activations + metadata; returns the file's sha256.

    pre_in / pre_out: the cot_on prefix states, [N, D]. meta["cot_off"] says where the cot_off states are: "same" (the
    contexts are identical, the cot_on arrays serve), "separate" (pre_in_off / pre_out_off / n_tokens_off are given) or
    "absent" (this arm was not captured for cot_off)."""
    pre_in, pre_out = np.asarray(pre_in, np.float32), np.asarray(pre_out, np.float32)
    if not (len(prompt_ids) == len(pre_in) == len(pre_out) == len(n_tokens) == len(last_ids)) or pre_in.shape[1] != len(r_hat):
        raise ValueError("inconsistent activation arrays")
    if meta.get("cot_off") not in COT_OFF_STATES:
        raise ValueError(f"meta['cot_off'] must be one of {COT_OFF_STATES}")
    extra = {}
    if meta["cot_off"] == "separate":
        if pre_in_off is None or pre_out_off is None or n_tokens_off is None:
            raise ValueError("cot_off 'separate' needs pre_in_off, pre_out_off and n_tokens_off")
        extra = {"pre_in_off": np.asarray(pre_in_off, np.float32), "pre_out_off": np.asarray(pre_out_off, np.float32),
                 "n_tokens_off": np.asarray(n_tokens_off, np.int64)}
        if any(len(v) != len(prompt_ids) for v in extra.values()):
            raise ValueError("inconsistent cot_off arrays")
    elif pre_in_off is not None or pre_out_off is not None:
        raise ValueError("cot_off arrays given although meta['cot_off'] is not 'separate'")
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, prompt_ids=np.array(prompt_ids), pre_in=pre_in, pre_out=pre_out, n_tokens=np.asarray(n_tokens, np.int64),
             last_ids=np.asarray(last_ids, np.int64), r_hat=np.asarray(r_hat, np.float64),
             meta_json=np.array(json.dumps(meta, sort_keys=True)), **extra)
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_acts(directory) -> dict:
    """{arm: {ids, index, in, out, in_off, out_off, r_hat, n_tokens, last_ids, meta, sha256}} from <dir>/acts_<arm>.npz.
    `in` / `out` are the cot_on prefix states, `in_off` / `out_off` the cot_off ones (the same arrays when the contexts
    are identical, None when the arm was not captured for cot_off)."""
    out = {}
    for f in sorted(pathlib.Path(directory).glob("acts_*.npz")):
        z = np.load(f)
        ids = [str(i) for i in z["prompt_ids"]]
        meta = json.loads(str(z["meta_json"]))
        pin, pout = z["pre_in"], z["pre_out"]          # NpzFile reads on every access: load once, alias for "same"
        state = meta.get("cot_off", "absent")
        off = {"separate": lambda: (z["pre_in_off"], z["pre_out_off"]), "same": lambda: (pin, pout), "absent": lambda: (None, None)}[state]()
        out[f.stem[len("acts_"):]] = {
            "ids": ids, "index": {i: k for k, i in enumerate(ids)}, "in": pin, "out": pout, "in_off": off[0], "out_off": off[1],
            "r_hat": z["r_hat"], "n_tokens": z["n_tokens"], "last_ids": z["last_ids"], "meta": meta,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    return out


# ------------------------------------------------------------------------------------------ labels and table
def label_rows(raw_rows: list) -> tuple[list, int]:
    """Episode rows -> label rows (strict outcome replayed from raw_output). Returns (rows, n_unreplayable)."""
    out, bad = [], 0
    for r in raw_rows:
        if "turns" not in r or r.get("family") != "main":
            continue
        try:
            fresh = episodes.recompute_record(r)
        except (KeyError, TypeError, AttributeError, ValueError):
            bad += 1
            continue
        meta = r.get("meta") or {}
        truncated = fresh["route"] == TRUNCATED_OR_MALFORMED
        out.append({"source": "exp31" if meta.get("exp") == "exp31" else "exp11", "arm": r["arm"], "cot_mode": r["cot_mode"],
                    "trial_id": r["trial_id"], "template": r["template_id"], "scenario": r["scenario_id"], "domain": r["domain"],
                    "cost": r["cost_regime"], "cons": r["consequence"], "bnd": r["boundary"], "order": r["order_index"],
                    "fmt": meta.get("stake_format", "narrative"), "pid": prompt_id(r["messages"]), "effort": r.get("reasoning_effort"),
                    "text": "\n".join(m["content"] for m in r["messages"]), "y": None if truncated else int(fresh["direct_taken"]),
                    "route": fresh["route"], "sub_status": fresh["sub_status"]})
    return out, bad


def select_effort(labels: list, requested="auto"):
    """The reasoning effort of the cot_on labels the probes use (cot_off rows always record None: the worker ignores
    effort without thinking). 'auto' demands exactly one cot_on effort in the data; a label ('low', 'default', ...)
    picks one of several. Returns the effort as stored (None = template default)."""
    efforts = {l["effort"] for l in labels if l["cot_mode"] == "cot_on"}
    if requested == "auto":
        if len(efforts) > 1:
            raise ValueError(f"cot_on labels come from several reasoning efforts {sorted(map(str, efforts))}; set config['effort']")
        return next(iter(efforts), None)
    chosen = None if requested == "default" else requested
    if chosen not in efforts:
        raise ValueError(f"no cot_on labels with reasoning effort {requested!r} (present: {sorted(map(str, efforts))})")
    return chosen


def _state_key(site: str, cot_mode: str) -> str:
    """Key of the activation array a row's label belongs to: the state of the prompt THAT MODE generated from."""
    return site if cot_mode == "cot_on" else site + "_off"


def build_table(labels: list, acts: dict, *, site: str = PRIMARY_SITE, sources=("exp11",)) -> dict:
    """Join labelled rows with activations: one row per (arm, cot_mode, prompt). Rows without a label (truncated),
    without activations, or whose prompt already came from a higher-priority source are dropped and counted."""
    rank = {s: i for i, s in enumerate(sources)}
    best: dict = {}
    n_trunc = n_no_acts = n_dup = 0
    for r in labels:
        if r["source"] not in rank:
            continue
        if r["y"] is None:
            n_trunc += 1
            continue
        a = acts.get(r["arm"])
        if a is None or r["pid"] not in a["index"] or a[_state_key(site, r["cot_mode"])] is None:
            n_no_acts += 1
            continue
        key = (r["arm"], r["cot_mode"], r["pid"])
        if key in best:
            n_dup += 1
            if rank[r["source"]] >= rank[best[key]["source"]]:
                continue
        best[key] = r
    rows = sorted(best.values(), key=lambda r: (r["arm"], r["cot_mode"], r["trial_id"]))
    if not rows:
        return {"n": 0, "dropped": {"truncated": n_trunc, "no_acts": n_no_acts, "duplicate_prompt": n_dup}}
    r_hats = [a["r_hat"] for a in acts.values()]
    if any(not np.array_equal(r_hats[0], v) for v in r_hats):
        raise ValueError("arms were captured with different r_hat vectors")
    X = np.stack([acts[r["arm"]][_state_key(site, r["cot_mode"])][acts[r["arm"]]["index"][r["pid"]]] for r in rows]).astype(np.float32)
    if r_hats[0].shape != (X.shape[1],):
        raise ValueError(f"r_hat has shape {r_hats[0].shape}, activations have width {X.shape[1]}")
    multi_fmt = len({r["fmt"] for r in rows}) > 1
    col = lambda k: np.array([r[k] for r in rows])
    tab = {"n": len(rows), "rows": rows, "X": X, "proj": X.astype(np.float64) @ r_hats[0], "y": col("y").astype(int),
           "site": site, "dropped": {"truncated": n_trunc, "no_acts": n_no_acts, "duplicate_prompt": n_dup}}
    for k in ("arm", "cot_mode", "template", "scenario", "domain", "cost", "cons", "bnd", "fmt", "source", "text"):
        tab[k] = col(k)
    tab["order"] = np.array([str(r["order"]) for r in rows])
    tab["cell"] = np.array([f"{r['arm']}|{r['cost']}|{r['cons']}|{r['bnd']}" + (f"|{r['fmt']}" if multi_fmt else "") for r in rows])
    return tab


def subset(tab: dict, mask) -> dict:
    mask = np.asarray(mask, bool)
    out = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in tab.items() if k != "rows"}
    out["rows"] = [r for r, m in zip(tab["rows"], mask) if m]
    out["n"] = int(mask.sum())
    return out


def template_folds(tab: dict, n_folds: int) -> np.ndarray:
    """Fold per row; templates are the unit (round-robin over (domain, template) order)."""
    keys = sorted(set(zip(tab["domain"], tab["template"])))
    n_folds = max(2, min(n_folds, len(keys)))
    fold_of = {t: i % n_folds for i, (_, t) in enumerate(keys)}
    return np.array([fold_of[t] for t in tab["template"]])


# ------------------------------------------------------------------------------------------ learners
# scikit-learn is imported inside the learners: the Modal image that runs v3.modal_acts does not ship it, and the
# capture job only needs the position-contract half of this module.
def _standardise(train, *others):
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return [(a - mu) / sd for a in (train, *others)]


def ridge_classifier_scores(X_tr, y_tr, X_te) -> np.ndarray:
    """Standardised RidgeClassifierCV (alpha by efficient leave-one-out on the training fold) decision values."""
    if len(set(y_tr)) < 2:
        return np.zeros(len(X_te))
    from sklearn.linear_model import RidgeClassifierCV

    Z_tr, Z_te = _standardise(np.asarray(X_tr, np.float64), np.asarray(X_te, np.float64))
    return RidgeClassifierCV(alphas=ALPHAS).fit(Z_tr, y_tr).decision_function(Z_te)


def ridge_regression_scores(X_tr, y_centered, X_te) -> np.ndarray:
    """Standardised RidgeCV on (already centred) targets; used by Regime C."""
    if np.allclose(y_centered, 0):
        return np.zeros(len(X_te))
    from sklearn.linear_model import RidgeCV

    Z_tr, Z_te = _standardise(np.asarray(X_tr, np.float64), np.asarray(X_te, np.float64))
    return RidgeCV(alphas=ALPHAS).fit(Z_tr, y_centered).predict(Z_te)


def logistic_scores(F_tr, y_tr, F_te) -> np.ndarray:
    if len(set(y_tr)) < 2:
        return np.zeros(len(F_te))
    from sklearn.linear_model import LogisticRegression

    Z_tr, Z_te = _standardise(np.asarray(F_tr, np.float64), np.asarray(F_te, np.float64))
    return LogisticRegression(max_iter=1000).fit(Z_tr, y_tr).decision_function(Z_te)


def lookup_scores(cells_tr, y_tr, cells_te, shrink: float = 1.0) -> np.ndarray:
    """Saturated cell model: smoothed training rate of each cell, (k + shrink * p0) / (n + shrink); unseen cell -> p0."""
    p0 = float(np.mean(y_tr)) if len(y_tr) else 0.5
    k, n = defaultdict(float), defaultdict(float)
    for c, y in zip(cells_tr, y_tr):
        k[c] += y
        n[c] += 1
    return np.array([(k[c] + shrink * p0) / (n[c] + shrink) if c in n else p0 for c in cells_te])


def one_hot_blocks(feature_cols: list) -> np.ndarray:
    """Concatenated one-hot encodings (levels sorted) of the given label columns, float [n, sum of levels]."""
    blocks = []
    for col in feature_cols:
        levels = sorted(set(col))
        blocks.append(np.array([[v == lv for lv in levels] for v in col], dtype=float))
    return np.hstack(blocks)


def design_plus_scores(feature_cols: list, y_tr, tr, te) -> np.ndarray:
    """A stronger tabular baseline than the cell lookup: L2 logistic regression on one-hot(cell) + one-hot(domain) +
    one-hot(tool order) (feature_cols = [cell, domain, order] arrays over all rows; tr / te are row indices)."""
    from sklearn.linear_model import LogisticRegression

    if len(set(y_tr)) < 2:
        return np.zeros(len(te))
    F = one_hot_blocks(feature_cols)
    return LogisticRegression(C=1.0, max_iter=2000).fit(F[tr], y_tr).decision_function(F[te])


def fit_vectorizer(texts):
    """TF-IDF (1-2 grams) fitted on `texts`; None when the corpus is too small for any term (score 0 then)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=3000)
    try:
        return vec.fit(texts)
    except ValueError:
        return None


def tfidf_scores(text_tr, y_tr, text_te) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    vec = fit_vectorizer(text_tr) if len(set(y_tr)) == 2 else None
    if vec is None:
        return np.zeros(len(text_te))
    return LogisticRegression(max_iter=3000).fit(vec.transform(text_tr), y_tr).decision_function(vec.transform(text_te))


def oof_scores(n: int, folds: np.ndarray, fit_predict) -> np.ndarray:
    """Out-of-fold scores: fit_predict(train_idx, test_idx) -> scores for test_idx, once per fold."""
    out = np.zeros(n)
    for f in sorted(set(folds)):
        te = np.flatnonzero(folds == f)
        out[te] = fit_predict(np.flatnonzero(folds != f), te)
    return out


@contextlib.contextmanager
def _threads(n: int | None):
    if not n:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=n):
        yield


# ------------------------------------------------------------------------------------------ AUROC helpers
def _auc_record(y, s, clusters, cfg) -> dict:
    r = stats.auroc_bootstrap_ci(y, s, cfg["n_boot"], cfg["seed"], clusters)
    return {k: r[k] for k in ("auroc", "ci_low", "ci_high", "n_pos", "n_neg", "n_clusters")}


def _diff_record(y, a, b, clusters, cfg) -> dict:
    r = stats.auroc_diff_bootstrap_ci(y, a, b, cfg["n_boot"], cfg["seed"], clusters)
    return {k: r[k] for k in ("diff", "ci_low", "ci_high", "p_boot")}


def _stratified_u(y: np.ndarray, s: np.ndarray, codes: np.ndarray, n_strata: int) -> float:
    """Same-stratum concordance in O(n log n) with one rank computation: scores are replaced by their dense ranks and
    shifted into contiguous per-stratum blocks, so the global mid-ranks minus the block offsets are the within-stratum
    mid-ranks (ties keep 1/2)."""
    dense = np.unique(s, return_inverse=True)[1].ravel()
    ranks = rankdata(codes * (int(dense.max()) + 1) + dense)
    n_k = np.bincount(codes, minlength=n_strata)
    offset = np.concatenate([[0], np.cumsum(n_k)[:-1]])
    n1 = np.bincount(codes, weights=y, minlength=n_strata)
    n0 = n_k - n1
    u = np.bincount(codes, weights=ranks * y, minlength=n_strata) - offset * n1 - n1 * (n1 + 1) / 2
    ok = (n1 > 0) & (n0 > 0)
    return float(u[ok].sum() / (n1 * n0)[ok].sum()) if ok.any() else float("nan")


def stratified_auroc(y, s, sid) -> float:
    """Concordance over same-stratum (positive, negative) pairs only; ties count 1/2. A score that is constant inside
    every stratum is exactly 0.5; with one stratum this is the ordinary AUROC."""
    sid = np.asarray(sid)
    codes = np.unique(sid, return_inverse=True)[1].ravel()
    return _stratified_u(np.asarray(y, float), np.asarray(s, float), codes, int(codes.max()) + 1)


def stratified_auroc_ci(y, scores: dict, sid, clusters, n_boot: int, seed: int, ref: str | None = None, alpha: float = 0.05) -> dict:
    """Stratified AUROC per method with a percentile bootstrap that resamples whole clusters (scenarios); with
    `ref`, paired difference ref - other on the same resamples. One stratum reproduces stats.auroc_bootstrap_ci."""
    y = np.asarray(y).astype(float)
    codes = np.unique(np.asarray(sid), return_inverse=True)[1].ravel()
    n_strata = int(codes.max()) + 1
    scores = {m: np.asarray(s, float) for m, s in scores.items()}
    draw = stats._boot_index_sampler(len(y), clusters, np.random.default_rng(seed))
    point = {m: _stratified_u(y, s, codes, n_strata) for m, s in scores.items()}
    boots: dict = {m: [] for m in scores}
    for _ in range(n_boot):
        idx = draw()
        vals = {m: _stratified_u(y[idx], s[idx], codes[idx], n_strata) for m, s in scores.items()}
        if all(np.isfinite(v) for v in vals.values()):
            for m, v in vals.items():
                boots[m].append(v)
    out = {}
    for m in scores:
        b = np.array(boots[m])
        lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(b) else (float("nan"),) * 2
        out[m] = {"auroc": point[m], "ci_low": float(lo), "ci_high": float(hi), "n_boot_used": int(len(b))}
    if ref in scores:
        for m in scores:
            if m != ref:
                d = np.array(boots[ref]) - np.array(boots[m])
                lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(d) else (float("nan"),) * 2
                out[m]["diff_ref_minus_this"] = {"diff": point[ref] - point[m], "ci_low": float(lo), "ci_high": float(hi)}
    return out


# ------------------------------------------------------------------------------------------ Regime A / B
def _scores_pooled(tab: dict, folds, methods) -> dict:
    n, y = tab["n"], tab["y"]
    fns = {
        "probe": lambda tr, te: ridge_classifier_scores(tab["X"][tr], y[tr], tab["X"][te]),
        "design": lambda tr, te: lookup_scores(tab["cell"][tr], y[tr], tab["cell"][te]),
        "design_plus": lambda tr, te: design_plus_scores([tab["cell"], tab["domain"], tab["order"]], y[tr], tr, te),
        "tfidf": lambda tr, te: tfidf_scores(tab["text"][tr], y[tr], tab["text"][te]),
        "rhat_cv": lambda tr, te: logistic_scores(tab["proj"][tr, None], y[tr], tab["proj"][te, None]),
    }
    out = {m: oof_scores(n, folds, fns[m]) for m in methods if m in fns}
    if "rhat_neg" in methods:
        out["rhat_neg"] = -tab["proj"]       # pre-specified direction, nothing fitted
    return out


def _compare(tab, scores: dict, cfg, pairs) -> dict:
    y, clusters = tab["y"], tab["scenario"]
    out = {"n": tab["n"], "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()), "n_scenarios": int(len(set(clusters))),
           "auroc": {m: _auc_record(y, s, clusters, cfg) for m, s in scores.items()}}
    out["diff"] = {f"{a}-{b}": _diff_record(y, scores[a], scores[b], clusters, cfg) for a, b in pairs if a in scores and b in scores}
    return out


def regime_a(tab: dict, cfg: dict) -> dict:
    """Full grid, one cot mode, all arms pooled: probe vs the arm x cost x consequence x boundary lookup (`design`, the
    pre-specified baseline), a stronger tabular `design_plus` (cell + domain + tool order), TF-IDF and the 1-D r_hat."""
    y = tab["y"]
    if tab["n"] < 20 or len(set(y)) < 2:
        return {"status": "too_little_data", "n": tab["n"]}
    folds = template_folds(tab, cfg["n_folds"])
    scores = _scores_pooled(tab, folds, ("probe", "design", "design_plus", "tfidf", "rhat_cv"))
    out = _compare(tab, scores, cfg, (("probe", "design"), ("probe", "design_plus"), ("probe", "tfidf"), ("probe", "rhat_cv")))
    out["per_arm"] = {a: {m: _auc_value(y[tab["arm"] == a], s[tab["arm"] == a]) for m, s in scores.items()}
                      for a in sorted(set(tab["arm"]))}
    out["n_cells"] = int(len(set(tab["cell"])))
    out["probe_exceeds_design_ci"] = bool(out["diff"]["probe-design"]["ci_low"] > 0)
    return out


def _auc_value(y, s) -> float | None:
    v = stats.auroc(y, s) if len(set(y)) == 2 else float("nan")
    return None if v != v else float(v)


def regime_b(tab_all: dict, cfg: dict) -> dict:
    """Within Intact (one cot mode): the 1-D h48.r_hat projection as predictor."""
    tab = subset(tab_all, tab_all["arm"] == "intact")
    y = tab["y"] if tab["n"] else np.array([0])
    if tab["n"] < 20 or len(set(y)) < 2:
        return {"status": "too_little_data", "n": tab["n"]}
    folds = template_folds(tab, cfg["n_folds"])
    scores = _scores_pooled(tab, folds, ("rhat_neg", "rhat_cv", "design", "design_plus", "probe", "tfidf"))
    out = _compare(tab, scores, cfg, (("rhat_neg", "design"), ("rhat_neg", "design_plus"), ("rhat_cv", "design"), ("probe", "design"),
                                      ("rhat_neg", "tfidf")))
    # the same projection judged inside cells only (pairs of the same cell), pre-specified sign; the design score is
    # centred per (cell, fold) like Regime C's, so it is exactly 0.5 there (rhat_neg is fitted nowhere: no offsets to remove)
    sid = tab["cell"]
    design = _center_scores(scores["design"], np.array([f"{c}#{f}" for c, f in zip(sid, folds)]))
    out["rhat_neg_within_cell"] = stratified_auroc_ci(y, {"rhat_neg": scores["rhat_neg"], "design": design}, sid,
                                                      tab["scenario"], cfg["n_boot"], cfg["seed"])
    out["orientation"] = "rhat_neg = -(h48 . r_hat): higher refusal-direction projection predicts fewer shortcuts"
    return out


# ------------------------------------------------------------------------------------------ Regime C
def _demean_by(F: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.array(F, dtype=np.float64, copy=True)
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean(0)
    return out


def _center_scores(s: np.ndarray, groups: np.ndarray) -> np.ndarray:
    s = np.array(s, dtype=np.float64, copy=True)
    for g in np.unique(groups):
        m = groups == g
        s[m] -= s[m].mean()
    return np.round(s, 10)      # a score constant inside a group becomes exactly 0.0


def _within_features(tab: dict, idx: np.ndarray, methods) -> dict:
    """Cell-demeaned features of the rows `idx` (label-free, so shared by every permutation)."""
    strata = tab["stratum"][idx]
    feats = {}
    if "probe" in methods:
        feats["probe"] = _demean_by(tab["X"][idx], strata)
    if "rhat" in methods:
        feats["rhat"] = _demean_by(tab["proj"][idx, None], strata)
    if "order_domain" in methods:    # what the prompt layout alone offers inside a cell: tool order and domain, same learner
        feats["order_domain"] = _demean_by(one_hot_blocks([tab["order"][idx], tab["domain"][idx]]), strata)
    return feats


def _within_scores(tab: dict, idx: np.ndarray, y: np.ndarray, folds: np.ndarray, methods, feats: dict | None = None) -> dict:
    """Out-of-fold within-cell scores for the rows `idx` (one scope group). Features are demeaned per cell, training
    labels centred per cell, held-out scores centred per (cell, fold). `y` may be a permuted copy (null)."""
    strata = tab["stratum"][idx]
    yg, fg, cells, texts = y[idx], folds[idx], tab["cell"][idx], tab["text"][idx]
    feats = feats if feats is not None else _within_features(tab, idx, methods)
    out = {m: np.zeros(len(idx)) for m in methods}
    if "tfidf" in methods:       # vocabulary / IDF from the group's prompts: label-free, so shared by all folds
        vec = fit_vectorizer(texts)
        feats = {**feats, **({"tfidf": _demean_by(vec.transform(texts).toarray(), strata)} if vec is not None else {})}
    for f in sorted(set(fg)):
        te, tr = np.flatnonzero(fg == f), np.flatnonzero(fg != f)
        if len(te) == 0 or len(set(yg[tr])) < 2:
            continue
        y_tr = yg[tr].astype(float)
        yc = y_tr.copy()
        for s_ in np.unique(strata[tr]):
            m = strata[tr] == s_
            yc[m] -= y_tr[m].mean()
        for name, F in feats.items():
            out[name][te] = ridge_regression_scores(F[tr], yc, F[te])
        if "design" in methods:
            out["design"][te] = lookup_scores(cells[tr], yg[tr], cells[te])
    cell_fold = np.array([f"{a}#{b}" for a, b in zip(strata, fg)])
    return {m: _center_scores(s_, cell_fold) for m, s_ in out.items()}


_SCOPES = {"arm_cot": ("arm", "cot_mode"), "cot": ("cot_mode",), "cell": ("stratum",)}
_C_METHODS = ("probe", "design", "tfidf", "rhat", "order_domain")


def regime_c(tab: dict, cfg: dict) -> dict:
    """Within-cell split trials (see module docstring). `tab` may hold both cot modes; strata never mix them."""
    if tab["n"] == 0:
        return {"status": "no_data"}
    tab = {**tab, "stratum": np.char.add(np.char.add(tab["cot_mode"].astype(str), "|"), tab["cell"].astype(str))}
    y = tab["y"]
    counts = {s: (int(((tab["stratum"] == s) & (y == 1)).sum()), int(((tab["stratum"] == s) & (y == 0)).sum())) for s in np.unique(tab["stratum"])}
    eligible = {s for s, (p, n) in counts.items() if min(p, n) >= cfg["min_minority"]}
    out = {"min_minority": cfg["min_minority"], "n_strata": len(counts), "n_eligible": len(eligible),
           "eligible": [{"stratum": s, "n_pos": counts[s][0], "n_neg": counts[s][1]} for s in sorted(eligible)], "scopes": {}}
    if not eligible:
        out["status"] = "no_eligible_cell"
        return out
    folds = template_folds(tab, cfg["n_folds"])
    keep = np.isin(tab["stratum"], sorted(eligible))
    methods = tuple(m for m in _C_METHODS if m in cfg.get("c_methods", _C_METHODS))
    for scope, keys in _SCOPES.items():
        if scope not in cfg.get("c_scopes", _SCOPES):
            continue
        labels = np.array(["|".join(str(tab[k][i]) for k in keys) for i in range(tab["n"])])
        groups = []
        for g in sorted(set(labels[keep])):
            idx = np.flatnonzero(keep & (labels == g))
            scores = _within_scores(tab, idx, y, folds, methods)
            sid, yy, cl = tab["stratum"][idx], y[idx], tab["scenario"][idx]
            entry = {"group": g, "n": int(len(idx)), "n_strata": int(len(set(sid))),
                     "pooled": stratified_auroc_ci(yy, scores, sid, cl, cfg["n_boot"], cfg["seed"], ref="probe")}
            if scope == "arm_cot":
                entry["strata"] = [{"stratum": s, "n": int((sid == s).sum()), "n_pos": int(yy[sid == s].sum()),
                                    "auroc": {m: _auc_record(yy[sid == s], sc[sid == s], cl[sid == s], cfg) for m, sc in scores.items()}}
                                   for s in sorted(set(sid))]
            groups.append(entry)
            if scope == "arm_cot" and cfg["n_perm"] > 0 and len(set(sid)) > 0:
                entry["permutation_null"] = _permutation_null(tab, idx, y, folds, cfg, entry["pooled"]["probe"]["auroc"])
        out["scopes"][scope] = groups
    d = [e["pooled"]["design"]["auroc"] for g in out["scopes"].values() for e in g if "design" in e["pooled"]]
    out["design_exactly_half_everywhere"] = bool(d) and all(v == 0.5 for v in d)
    return out


def _permutation_null(tab, idx, y, folds, cfg, observed) -> dict:
    """Shuffle labels among the scenarios of each cell (the activations stay), refit, recompute the stratified AUROC."""
    rng = np.random.default_rng(cfg["seed"])
    sid, null = tab["stratum"][idx], []
    feats = _within_features(tab, idx, ("probe",))
    for _ in range(cfg["n_perm"]):
        yp = y.copy()
        for s in np.unique(sid):
            rows = idx[sid == s]
            yp[rows] = y[rng.permutation(rows)]
        sc = _within_scores(tab, idx, yp, folds, ("probe",), feats)
        null.append(stratified_auroc(yp[idx], sc["probe"], sid))
    null = np.array(null)
    return {"n_perm": int(len(null)), "observed": observed, "null_mean": float(null.mean()),
            "null_q95": float(np.percentile(null, 95)), "p_perm": float((1 + (null >= observed).sum()) / (len(null) + 1))}


# ------------------------------------------------------------------------------------------ cost
def n_unique_prompts(scale: float = 1.0, *, cells: str = "all12", bases_per_domain: int | None = None, orders: int | None = None,
                     n_exp11: int = 480) -> int:
    """Prompts one arm forwards: the Exp 1.1 main grid plus the Exp 3.1 prompts minus those the two share (every
    order-0 narrative Exp 3.1 prompt IS an Exp 1.1 grid prompt)."""
    design = exp31.scale_design(scale)
    bpd, orders = bases_per_domain or design["bases_per_domain"], orders or design["orders"]
    n_cells = len(exp31.CELL_SETS[cells])
    return n_exp11 + 4 * bpd * n_cells * orders * 2 - 4 * bpd * n_cells


def estimate_cost(scale: float = 1.0, *, n_arms: int = len(arms.ARMS_27B), n_cot_off_arms: int | None = None,
                  separate_cot_off: bool = True, n_shards: int | None = None, cells: str = "all12", n_prompts: int | None = None,
                  load_s: float = 150.0, build_s: float = 150.0, forward_s: float = 0.25, overhead_s: float = 120.0,
                  gpu: str = "H200", timeout_s: float = 7200.0, max_arms_per_shard: int = 2) -> dict:
    """Forwards, GPU-hours and USD of the activation capture (one container per shard, one H200 each).

    Every arm is forwarded on the cot_on prefixes; the `n_cot_off_arms` arms that have cot_off labels (default
    min(3, n_arms): Exp 3.1's arms) need a second pass when the cot_off context differs from the cot_on one
    (`separate_cot_off`, true at reasoning effort xhigh / low / default, false at 'medium').
    ASSUMED, not measured: swap load 150 s per checkpoint (FP8 read + dequantisation), model build + mount 150 s per
    container, 0.25 s per padding-free prompt forward (~900 tokens through 48 of 64 layers), 120 s container start.
    `scale` sets the Exp 3.1 half of the prompt set."""
    from v3 import ledger

    n_prompts = n_unique_prompts(scale, cells=cells) if n_prompts is None else n_prompts
    n_cot_off_arms = min(3, n_arms) if n_cot_off_arms is None else n_cot_off_arms
    n_shards = n_shards or -(-n_arms // max_arms_per_shard)
    forwards = n_prompts * (n_arms + (n_cot_off_arms if separate_cot_off else 0))
    hours = (n_shards * (build_s + overhead_s) + n_arms * load_s + forwards * forward_s) / 3600
    per_shard_arms = -(-n_arms // n_shards)
    return {"prompts_per_arm": n_prompts, "forwards": forwards, "n_shards": n_shards, "gpu_hours": round(hours, 3),
            "usd": ledger.est_cost(gpu, 1, hours),
            "wall_hours_if_parallel": round(((build_s + overhead_s) + per_shard_arms * load_s
                                             + forwards / n_arms * per_shard_arms * forward_s) / 3600, 3),
            "worst_case_usd": ledger.est_cost(gpu, n_shards, timeout_s / 3600)}


# ------------------------------------------------------------------------------------------ receipt function
def recompute(raw_rows: list, acts: dict, config: dict | None = None) -> dict:
    """Raw episode rows (Exp 1.1 main grid, optionally Exp 3.1) + saved activations -> the whole Exp 3.2 summary.
    Deterministic (seeded; BLAS pinned to `threads`). Floats are rounded to 6 decimals."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    labels, n_unreplayable = label_rows(raw_rows)
    effort = select_effort(labels, cfg["effort"])
    labels = [l for l in labels if l["cot_mode"] == "cot_off" or l["effort"] == effort]
    used = {l["arm"] for l in labels if l["cot_mode"] == "cot_on" and l["arm"] in acts}
    wrong = sorted(a for a in used if acts[a]["meta"].get("effort") != effort)
    if wrong:
        raise ValueError(f"activations of {wrong} were captured at reasoning effort {acts[wrong[0]]['meta'].get('effort')!r}, "
                         f"but the cot_on labels come from effort {effort!r}: the pre-think states are of different prompts")
    out: dict = {"schema": "exp32_v1", "config": {k: v for k, v in cfg.items()}, "n_label_rows": len(labels), "effort": effort,
                 "notes": ["labels: strict replayed outcome, y = direct_taken; truncated episodes are unlabelled and dropped",
                           "Regime C design AUROC is 0.5 by construction (constant inside a cell); bootstrap CIs resample scenarios "
                           "and slightly under-cover cross-validated AUROCs: use permutation_null.p_perm as the calibrated test"],
                 "n_unreplayable": n_unreplayable,
                 "acts": {a: {"sha256": v["sha256"], "n_prompts": len(v["ids"]), "dim": int(v["out"].shape[1]),
                              "last_token_ids": sorted({int(i) for i in v["last_ids"]}), "cot_off": v["meta"].get("cot_off"),
                              "meta": v["meta"]} for a, v in sorted(acts.items())},
                 "datasets": {}}
    with _threads(cfg["threads"]):
        present = {l["source"] for l in labels}
        for name, sources in cfg["datasets"].items():
            ds: dict = {"sources": sources, "sites": {}}
            if not set(sources) <= present:
                ds["status"] = f"missing sources {sorted(set(sources) - present)}"
                out["datasets"][name] = ds
                continue
            for site in [cfg["site"], *cfg["sensitivity_sites"]]:
                tab = build_table(labels, acts, site=site, sources=tuple(sources))
                if tab["n"] == 0:
                    ds["sites"][site] = {"status": "no_rows", "dropped": tab["dropped"]}
                    continue
                res: dict = {"n": tab["n"], "dropped": tab["dropped"],
                             "label_counts": [{"arm": a, "cot_mode": c, "n": int(((tab["arm"] == a) & (tab["cot_mode"] == c)).sum()),
                                               "n_shortcut": int(tab["y"][(tab["arm"] == a) & (tab["cot_mode"] == c)].sum())}
                                              for a in sorted(set(tab["arm"])) for c in sorted(set(tab["cot_mode"][tab["arm"] == a]))],
                             "regimes": {}}
                for cot in sorted(set(tab["cot_mode"])):
                    sub = subset(tab, tab["cot_mode"] == cot)
                    reg = {"A": regime_a(sub, cfg), "B": regime_b(sub, cfg)}
                    if site == cfg["site"]:
                        reg["C"] = regime_c(sub, cfg)
                    res["regimes"][cot] = reg
                ds["sites"][site] = res
            out["datasets"][name] = ds
    return stats.to_jsonable(_round(out))


def _round(x, nd: int = 6):
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    return x
