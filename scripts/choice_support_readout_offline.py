#!/usr/bin/env python3
"""One admitted offline support decoder; prepare never reads hidden feature arrays."""

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import zipfile

import numpy as np

ASSAY = "choice_support_readout_offline"
DELIVERIES = ("decisive_evidence", "redundant_evidence", "independent_novelty")
LAMBDA = 1 / 12
PLAN = "reports/choice-support-readout-offline-v1-plan.md"
CODE_FILES = (
    PLAN,
    "scripts/choice_support_readout_offline.py",
    "scripts/audit_choice_support_readout_offline.py",
    "tests/test_choice_support_readout_offline.py",
    "tests/test_audit_choice_support_readout_offline.py",
)
METHOD = dict(
    layer=57, position="decision", features=5120, fit_worlds=12, evaluation_worlds=12,
    fit_rows=72, evaluation_rows=72, dtype="float64", ridge_lambda=LAMBDA,
    normalization="fit_only_center_and_scalar_rms_euclidean_norm",
    threshold=0.5, exact_set_floor_numerator=9, exact_set_floor_denominator=10,
    shortcut="fit_modal_mask_given_oracle_cardinality_lexical_ties",
    shortcut_gain_numerator=1, shortcut_gain_denominator=4,
    retrospective_boundary=True, causal_admission=False,
)
PINS = {
    "v1": dict(
        manifest="c33d76eba3931aeacadf68f6282c7c1471be8edccec83e7ca2f37ab076e68feb",
        source="7e005358661c0de588faba66c9e9e5384984810abbc27385351ca5a2bdf8a911",
        archive="23254d727f0788da9a353f36c9adaf51e92aefaed19f8ecec4e96b131a2b79cf",
        audit="d1a4bfdbdc68ee1ce63d18709c6640d805d6fa065623f3c8ea0b452c3d9bf4bf",
        assignments="1052a14bf9e73cd00697e51e4c1602b1f026ac4a4ef0f00c7f120340997b289e",
        labels="b7fd0e2336cb8091c8e74d769d1e03bc132f2b88dea8826923a308d63eaa6165",
        captures="2277f187965240341451ebabbfe5ad871e3d14ff957e5e24054dcf66323768ca",
        inventory="c6c5635101a71b4c324ee5a380686cb9f7474c552a89060176d120acb408c70f",
        attempt="81fc07b16979410eba445ebb7a873ef8",
    ),
    "v2": dict(
        manifest="b2ef2c264176003c9a0bcedf03a3e7bd80668cef38167d29f9ce3d9005861ab6",
        source="88a91e516e2580b23b9468bb1d38ba1c8a2b1e818e5f4e363ded74874a666623",
        archive="042a017d07d97ee149a7648608d7c51aeac4ef3f2f6d241f83179249ffe9a174",
        audit="aeec95021b15206e992ecf2a29655d128860b1c2230754d99d8fcac767a3412b",
        assignments="adde817547c7f6dee5a125fa616a3cf1cb98a56708ec89e01280ce05f0f6330d",
        labels="4508c0497c0b3a271c7c40ec2a21f0f99a60bd365661fa81e5a4cb1ca1f742e6",
        captures="a0c2f530df644c2994a92869dd41a5836ef6c977496a7350f1bd65a30e6ab0db",
        inventory="aaf41f020927602e2326ce00a2c30dedff991b4e002d0f98afc0da7fe35b5bc6",
        attempt="eeb0e3a93d644bf188d4c0dcfb1eee88",
    ),
}
SHARED_VISIBLE = "5e85fe2ddf66df3c53d08d031bc5c720402e6ead207a6cbe69ac2b26f68c5d3c"
SOURCE_FILES = (
    "src/curiosity_study/voluntary_resolution.py",
    "src/curiosity_study/information_state.py",
    "src/curiosity_study/reasoning_trace.py",
    "src/curiosity_study/batched.py",
    "src/curiosity_study/thinking_information.py",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(canonical(value) + "\n")


def row_key(a):
    return a["world_slot"], DELIVERIES.index(a["delivery"]), a["menu_layout"]


def displayed_oracle(text):
    """Only visible table/facts; no answer text or realized-truth shortcut."""
    require(text.count("Candidate | ") == 1, "Ambiguous displayed candidate table")
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("Candidate | "))
    attributes = [s.strip() for s in lines[index].split("|")[1:]]
    require(len(attributes) == 2 and len(set(attributes)) == 2, "Invalid attributes")
    table = []
    for offset, name in enumerate("WXYZ", 1):
        cells = [s.strip() for s in lines[index + offset].split("|")]
        require(len(cells) == 3 and cells[0] == name, "Invalid displayed row")
        table.append(cells[1:])
    require(len(set(map(tuple, table))) == 4, "Duplicate candidates")
    require(text.count("Initial observation:") == 1, "Ambiguous initial observation")
    section = text.split("Initial observation:", 1)[1].split("The hidden candidate does not change.", 1)[0]
    facts = {}
    for line in section.splitlines():
        match = re.fullmatch(r"The (.+?) is (.+)\.", line)
        if match:
            name, value = match.groups()
            require(name in attributes and (name not in facts or facts[name] == value), "Conflicting fact")
            facts[name] = value
    require(len(facts) in (1, 2), "Unexpected evidence inventory")
    bits = [int(all(row[attributes.index(k)] == v for k, v in facts.items())) for row in table]
    require(sum(bits) in (1, 2), "Unexpected compatible support")
    order = sorted(range(2), key=attributes.__getitem__)
    visible = dict(
        attributes=[attributes[i] for i in order],
        candidates=sorted([[row[i] for i in order] for row in table]),
        observed=sorted([list(item) for item in facts.items()]),
    )
    return dict(bits=bits, attributes=attributes, candidates=table, visible=visible,
                visible_problem_sha256=digest(visible))


def capture_metadata(path):
    """Read the hidden NPY header, never its feature values; metadata arrays only."""
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist().count("hidden.npy") == 1, "Missing/duplicate hidden array")
        with archive.open("hidden.npy") as stream:
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), "Unsupported hidden NPY header version")
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(stream)
    require(shape == (32, 4, 7, 5120) and dtype == np.dtype("float32") and not fortran,
            "Unexpected hidden header")
    names = ("trial_ids", "layers", "positions", "available", "token_positions", "closure_counts")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in names}
    require(arrays["layers"].tolist() == [15, 31, 47, 57], "Capture layers changed")
    require(arrays["positions"].tolist() == ["record", "prefill", "token_32", "token_128",
                                            "token_512", "think_end", "decision"], "Positions changed")
    require(arrays["available"].shape == (32, 4, 7)
            and arrays["token_positions"].shape == (32, 7)
            and arrays["closure_counts"].shape == (32,), "Capture metadata shape")
    return arrays


def archived_tasks(archive_path, assignments):
    """Cross-check with the immutable task source; no model construction or inference."""
    name = "_offline_support_source_" + sha(archive_path)[:16]
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary)
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                if member.name.startswith("src/curiosity_study/") and member.name.endswith(".py"):
                    require(member.isfile() and ".." not in Path(member.name).parts, "Unsafe source member")
                    target = destination / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
        import types
        package = types.ModuleType(name)
        package.__path__ = [str(destination / "src/curiosity_study")]
        sys.modules[name] = package
        try:
            module = importlib.import_module(name + ".voluntary_resolution")
            return {a["trial_id"]: dict(task=module.task(a), context=module.context(a)) for a in assignments}
        finally:
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    del sys.modules[key]


def prepare_collection(project, version):
    pin = PINS[version]
    run = project / "runs" / f"qwen32-voluntary-resolution-{version}"
    audit_path = project / "reports" / f"qwen32-voluntary-resolution-{version}-audit.json"
    manifest, freeze, audit = read(run / "manifest.json"), read(run / "freeze.json"), read(audit_path)
    require(digest(manifest) == pin["manifest"] == freeze["manifest_sha256"] == audit["manifest_sha256"],
            "Source manifest mismatch")
    require(manifest["source_sha256"] == pin["source"] == freeze["source_sha256"] == audit["source_sha256"],
            "Source identity mismatch")
    require(sha(run / "source.tar.gz") == pin["archive"] == freeze["archive_sha256"]
            and sha(audit_path) == pin["audit"], "Source archive/audit changed")
    require(audit["complete"] is True and audit["integrity_passed"] is True, "Incomplete source audit")
    assignments = sorted([a for a in manifest["diagnostics"] if a["family"] == "objects" and a["kind"] == "choice"], key=row_key)
    require(len(assignments) == 72 and digest(assignments) == pin["assignments"], "Fixed source rows changed")
    expected_grid = {(w, d, menu) for w in range(12) for d in DELIVERIES for menu in (0, 1)}
    require({(a["world_slot"], a["delivery"], a["menu_layout"]) for a in assignments} == expected_grid,
            "Incomplete world factorial")
    records, captures, files = {}, {}, {}
    paths = [run / name for name in ("manifest.json", "freeze.json", "source.tar.gz")] + [audit_path]
    for offset in range(0, 288, 32):
        cohort_path = run / "cohorts" / pin["attempt"] / f"{offset:07d}.json"
        array_path = run / "activations" / pin["attempt"] / f"{offset:07d}.npz"
        paths.extend((cohort_path, array_path))
        cohort = read(cohort_path)
        require(cohort["manifest_hash"] == pin["manifest"] and digest(cohort["records"]) == cohort["sha256"],
                "Source cohort binding mismatch")
        require([r["assignment"] for r in cohort["records"]] == manifest["diagnostics"][offset:offset + 32],
                "Source cohort row order mismatch")
        meta = capture_metadata(array_path)
        array_sha = sha(array_path)
        for row, record in enumerate(cohort["records"]):
            trial = record["trial_id"]
            require(trial not in records and str(meta["trial_ids"][row]) == trial, "Duplicate/misbound source row")
            desc = record["activation_artifact"]
            require(desc["path"] == str(array_path.relative_to(run)) and desc["row"] == row
                    and desc["sha256"] == array_sha, "Source activation binding mismatch")
            require(desc["available"] == meta["available"][row].tolist()
                    and desc["token_positions"] == meta["token_positions"][row].tolist(), "Capture metadata mismatch")
            records[trial] = record
            captures[trial] = dict(
                path=str(array_path.relative_to(project)), byte_sha256=array_sha, row=row,
                layer=57, layer_index=3, position="decision", position_index=6,
                token_position=int(meta["token_positions"][row, 6]),
                available=bool(meta["available"][row, 3, 6]),
                closure_count=int(meta["closure_counts"][row]),
            )
    require(len(records) == 288, "Incomplete source records")
    audit_rows = {r["trial_id"]: r for r in audit["outcomes"]}
    require(set(audit_rows) == set(records), "Source audit record inventory changed")
    for trial, record in records.items():
        require(audit_rows[trial]["record_sha256"] == digest(record), "Source audited record bytes changed")
    for path in paths:
        files[str(path.relative_to(project))] = sha(path)
    require(len(files) == 22 and digest(files) == pin["inventory"], "Closed input inventory changed")
    archived = archived_tasks(run / "source.tar.gz", assignments)
    rows, label_ledger, capture_ledger = [], [], []
    for a in assignments:
        trial = a["trial_id"]
        record, capture = records[trial], captures[trial]
        require(a == record["assignment"], "Source assignment mismatch")
        prompt = record["inference"]["rendered_prompt"]
        oracle = displayed_oracle(prompt)
        original = archived[trial]
        require(original["context"] in prompt and canonical(original["task"]) == canonical(record["task"]),
                "Archived source reconstruction disagrees")
        require([i for i, bit in enumerate(oracle["bits"]) if bit] == original["task"]["compatible"],
                "Independent displayed oracle disagrees")
        label = dict(trial_id=trial, assignment_sha256=digest(a), world_slot=a["world_slot"],
                     semantic_world_id=a["semantic_world_id"], delivery=a["delivery"],
                     menu_layout=a["menu_layout"], candidate_labels=list("WXYZ"), compatible_bits=oracle["bits"])
        label_ledger.append(label)
        capture_ledger.append({**label, "record_sha256": digest(record),
            "activation_path": record["activation_artifact"]["path"], "activation_byte_sha256": capture["byte_sha256"],
            "activation_row": capture["row"], "layer": 57, "position": "decision", "available": capture["available"],
            "token_position": capture["token_position"], "closure_count": capture["closure_count"]})
        rows.append(dict(
            split="fit" if version == "v1" else "evaluation", collection=version,
            trial_id=trial, assignment_sha256=digest(a), record_sha256=digest(record),
            world_slot=a["world_slot"], semantic_world_id=a["semantic_world_id"],
            delivery=a["delivery"], menu_layout=a["menu_layout"], known_attribute=a["known"][0],
            oracle_bits=oracle["bits"], visible_problem_sha256=oracle["visible_problem_sha256"],
            available=capture["available"], capture=capture,
            source_flags=dict(valid=record["valid"], cap_hit=record["cap_hit"],
                              thinking_closed=record["thinking_closed"], closure_count=capture["closure_count"]),
        ))
    require(digest(label_ledger) == pin["labels"] and digest(capture_ledger) == pin["captures"], "Fixed label/capture ledger changed")
    with tarfile.open(run / "source.tar.gz") as archive:
        source_files = {path: hashlib.sha256(archive.extractfile(path).read()).hexdigest() for path in SOURCE_FILES}
    return dict(version=version, pins=pin, files=files, source_files=source_files, rows=rows)


def prepare(project):
    """Return a JSON-ready ledger. No hidden feature values, norms or fits are read."""
    project = Path(project).resolve()
    code_root = Path(__file__).resolve().parents[1]
    code_files = {path: sha(code_root / path) for path in CODE_FILES}
    collections = [prepare_collection(project, v) for v in ("v1", "v2")]
    rows = [row for collection in collections for row in collection.pop("rows")]
    train_visible = {r["visible_problem_sha256"] for r in rows if r["split"] == "fit"}
    for row in rows:
        row["seen_in_fit"] = row["visible_problem_sha256"] in train_visible
    overlap = [r for r in rows if r["split"] == "evaluation" and r["seen_in_fit"]]
    require(len(overlap) == 4 and {r["visible_problem_sha256"] for r in overlap} == {SHARED_VISIBLE}
            and {r["world_slot"] for r in overlap} == {3}, "Prespecified visible overlap changed")
    worlds = [{r["semantic_world_id"] for r in rows if r["split"] == split} for split in ("fit", "evaluation")]
    require(all(len(s) == 12 for s in worlds) and not worlds[0] & worlds[1], "Exact world split changed")
    result = dict(schema_version=1, assay=ASSAY, method=METHOD, code_files=code_files,
                  collections=collections, rows=rows, rows_sha256=digest(rows),
                  visible_overlap=dict(sha256=SHARED_VISIBLE, evaluation_trial_ids=[r["trial_id"] for r in overlap]),
                  runtime=dict(python=list(sys.version_info[:3]), numpy=np.__version__),
                  hidden_feature_values_read=False, causal_admission=False)
    verify_bound_files(project, result)
    return result


def verify_bound_files(project, ledger):
    for collection in ledger["collections"]:
        for path, expected in collection["files"].items():
            require(sha(Path(project) / path) == expected, "Input file changed: " + path)
    code_root = Path(__file__).resolve().parents[1]
    for path, expected in ledger["code_files"].items():
        require(sha(code_root / path) == expected, "Code/plan changed: " + path)


def fit_ridge(X, Y):
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    require(X.ndim == 2 and Y.shape == (len(X), 4) and np.isfinite(X).all()
            and np.isfinite(Y).all() and np.isin(Y, (0, 1)).all(), "Invalid fit arrays")
    mu, ybar = X.mean(axis=0), Y.mean(axis=0)
    centered = X - mu
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    require(np.isfinite(scale) and scale > 0, "Degenerate training scale")
    Z = centered / scale
    K = Z @ Z.T
    B = np.linalg.solve(K + len(X) * LAMBDA * np.eye(len(X)), Y - ybar)
    W = Z.T @ B
    singular = np.linalg.svd(Z, compute_uv=False)
    eig = np.linalg.eigvalsh(K)
    diagnostics = dict(rank=int(np.count_nonzero(singular > singular[0] * 1e-10)),
                       effective_df=float(np.sum(eig / (eig + len(X) * LAMBDA))),
                       n_features=X.shape[1], train_rows=len(X), ridge_lambda=LAMBDA)
    require(diagnostics["effective_df"] <= 12 + 1e-9, "Effective complexity bound failed")
    return {**dict(mu=mu, scale=np.asarray(scale), ybar=ybar, W=W, B=B, available=np.asarray(True)),
            "lambda": np.asarray(LAMBDA, dtype=np.float64)}, diagnostics


def predict(model, X):
    return model["ybar"] + ((np.asarray(X, dtype=np.float64) - model["mu"]) / model["scale"]) @ model["W"]


def shortcut(train_rows):
    masks = {}
    for k in (1, 2):
        counts = Counter("".join(map(str, r["oracle_bits"])) for r in train_rows if sum(r["oracle_bits"]) == k)
        require(bool(counts), "Missing training cardinality")
        chosen = min(counts, key=lambda mask: (-counts[mask], mask))
        masks[str(k)] = dict(bits=list(map(int, chosen)), counts=dict(sorted(counts.items())))
    return masks


def exact(row):
    return row["predicted_bits"] is not None and row["predicted_bits"] == row["oracle_bits"]


def count_metric(rows):
    n, correct = len(rows), sum(exact(r) for r in rows)
    return dict(assigned=n, correct=correct, required=(9 * n + 9) // 10,
                passed=n > 0 and 10 * correct >= 9 * n)


def assess_predictions(rows, *, model_available, integrity_passed=True):
    require(type(model_available) is bool and type(integrity_passed) is bool, "Admission flags must be booleans")
    require(len(rows) == 144 and len({r["trial_id"] for r in rows}) == 144, "All144 assigned rows required")
    for row in rows:
        require(type(row["available"]) is bool and type(row["seen_in_fit"]) is bool
                and all(type(row["source_flags"][key]) is bool for key in ("valid", "cap_hit", "thinking_closed")),
                "Row flags must be booleans")
        for name in ("oracle_bits", "shortcut_bits", "predicted_bits"):
            bits = row[name]
            require((name == "predicted_bits" and bits is None) or
                    (isinstance(bits, list) and len(bits) == 4 and all(type(b) is int and b in (0, 1) for b in bits)),
                    "Invalid support bits")
        scores = row["scores"]
        if model_available and row["available"]:
            require(isinstance(scores, list) and len(scores) == 4
                    and all(type(s) in (int, float) for s in scores) and np.isfinite(scores).all(),
                    "Available prediction requires four finite scores")
            require(row["predicted_bits"] == [int(s >= 0.5) for s in scores]
                    and row["prediction_status"] == "predicted", "Prediction does not match fixed threshold")
        else:
            expected_status = "capture_unavailable" if not row["available"] else "model_unavailable"
            require(scores is None and row["predicted_bits"] is None
                    and row["prediction_status"] == expected_status, "Unavailable prediction must remain explicit")
    parts = {split: [r for r in rows if r["split"] == split] for split in ("fit", "evaluation")}
    for group in parts.values():
        require(len(group) == 72 and {(r["world_slot"], r["delivery"], r["menu_layout"]) for r in group}
                == {(world, delivery, menu) for world in range(12) for delivery in DELIVERIES for menu in (0, 1)},
                "Incomplete fixed split")
    evaluation = parts["evaluation"]
    primary, unseen = [], []
    for delivery in DELIVERIES:
        for menu in (0, 1):
            selected = [r for r in evaluation if r["delivery"] == delivery and r["menu_layout"] == menu]
            nonoverlap = [r for r in selected if not r["seen_in_fit"]]
            require(len(selected) == 12 and len(nonoverlap) == (12 if delivery == DELIVERIES[0] else 11),
                    "Fixed sensitivity denominator changed")
            primary.append(dict(delivery=delivery, menu_layout=menu, **count_metric(selected)))
            unseen.append(dict(delivery=delivery, menu_layout=menu, **count_metric(nonoverlap)))
    strata = []
    for k, n in ((1, 24), (2, 48)):
        selected = [r for r in evaluation if sum(r["oracle_bits"]) == k]
        require(len(selected) == n, "Cardinality denominator changed")
        correct = sum(exact(r) for r in selected)
        reference = sum(r["shortcut_bits"] == r["oracle_bits"] for r in selected)
        strata.append(dict(cardinality=k, assigned=n, correct=correct, shortcut_correct=reference,
                           additional_correct=correct-reference, required_additional=n//4,
                           passed=4*(correct-reference) >= n))
    lookup = {(r["world_slot"], r["delivery"], r["menu_layout"]): r for r in evaluation}
    paired = {}
    for kind in ("evidence_sensitivity", "novelty_invariance", "menu_invariance"):
        entries = []
        conditions = DELIVERIES if kind == "menu_invariance" else (0, 1)
        for condition in conditions:
            pairs = []
            for world in range(12):
                if kind == "menu_invariance":
                    pair = [lookup[world, condition, menu] for menu in (0, 1)]
                else:
                    ds = DELIVERIES[:2] if kind == "evidence_sensitivity" else DELIVERIES[1:]
                    pair = [lookup[world, d, condition] for d in ds]
                ok = all(exact(r) for r in pair)
                if kind == "evidence_sensitivity":
                    ok = ok and sum(pair[0]["oracle_bits"]) == 1 and all(a <= b for a, b in zip(pair[0]["oracle_bits"], pair[1]["oracle_bits"]))
                else:
                    ok = ok and pair[0]["oracle_bits"] == pair[1]["oracle_bits"]
                pairs.append(dict(world_slot=world, correct=bool(ok), nonoverlap=all(not r["seen_in_fit"] for r in pair)))
            un = [p for p in pairs if p["nonoverlap"]]
            entries.append(dict(condition=condition, assigned=12, correct=sum(p["correct"] for p in pairs),
                                required=11, passed=sum(p["correct"] for p in pairs) >= 11,
                                nonoverlap_assigned=len(un), nonoverlap_correct=sum(p["correct"] for p in un)))
        paired[kind] = entries
    summaries = {}
    for split, group in parts.items():
        summaries[split] = dict(assigned=72, predicted=sum(r["predicted_bits"] is not None for r in group),
            exact_correct=sum(exact(r) for r in group),
            cardinality_correct=sum(r["predicted_bits"] is not None and sum(r["predicted_bits"]) == sum(r["oracle_bits"]) for r in group),
            all_six_correct_worlds=sum(all(exact(r) for r in group if r["world_slot"] == w) for w in range(12)),
            source_invalid=sum(not r["source_flags"]["valid"] for r in group),
            source_caps=sum(r["source_flags"]["cap_hit"] for r in group),
            capture_unavailable=sum(not r["available"] for r in group))
    bits = []
    for j, label in enumerate("WXYZ"):
        counts = Counter()
        for r in evaluation:
            if r["predicted_bits"] is None:
                counts["missing"] += 1
            else:
                y, p = r["oracle_bits"][j], r["predicted_bits"][j]
                counts[{(1, 1): "tp", (0, 0): "tn", (0, 1): "fp", (1, 0): "fn"}[y, p]] += 1
        bits.append(dict(candidate=label, **{key: counts[key] for key in ("tp", "tn", "fp", "fn", "missing")}))
    roles = [dict(known_attribute=k, **count_metric([r for r in evaluation if r["known_attribute"] == k])) for k in (0, 1)]
    checks = dict(provenance_and_grouping_passed=bool(integrity_passed), complete_captures=all(r["available"] for r in rows),
                  model_available=bool(model_available), primary_exact_sets=all(c["passed"] for c in primary),
                  nonoverlap_exact_sets=all(c["passed"] for c in unseen), shortcut_improvement=all(c["passed"] for c in strata),
                  **{kind: all(c["passed"] for c in cells) for kind, cells in paired.items()})
    return dict(checks=checks, instrument_feasibility_passed=all(checks.values()),
                metrics=dict(summaries=summaries, primary_cells=primary, nonoverlap_cells=unseen,
                             cardinality_strata=strata, paired=paired, per_bit=bits, known_property_roles=roles))


def load_features(project, rows):
    """Called only by execute after exact admitted ledger reproduction."""
    result = np.full((len(rows), METHOD["features"]), np.nan, dtype=np.float64)
    grouped = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row["capture"]["path"], []).append((index, row))
    for path, selected in grouped.items():
        require(sha(Path(project) / path) == selected[0][1]["capture"]["byte_sha256"], "Activation changed before feature read")
        with np.load(Path(project) / path, allow_pickle=False) as archive:
            hidden = archive["hidden"]
            for index, row in selected:
                if row["available"]:
                    c = row["capture"]
                    vector = hidden[c["row"], c["layer_index"], c["position_index"]]
                    require(vector.shape == (METHOD["features"],) and np.isfinite(vector).all(), "Available feature is nonfinite")
                    result[index] = vector
    return result


def build_predictions(rows, features):
    require(len(rows) == 144 and features.shape == (144, METHOD["features"]), "Fixed feature shape required")
    train, evaluation = rows[:72], rows[72:]
    require(all(r["split"] == "fit" for r in train) and all(r["split"] == "evaluation" for r in evaluation), "Split order changed")
    reference = shortcut(train)
    available = all(r["available"] for r in train)
    diagnostics = dict(reason="missing_training_capture", rank=None, effective_df=None)
    if available:
        try:
            model, diagnostics = fit_ridge(features[:72], [r["oracle_bits"] for r in train])
        except (ValueError, np.linalg.LinAlgError) as error:
            available = False
            diagnostics = dict(reason="fixed_fit_unavailable", error_type=type(error).__name__,
                               error=str(error), rank=None, effective_df=None)
    if not available:
        model = {key: np.empty((0,), dtype=np.float64) for key in ("mu", "scale", "ybar", "W", "B")}
        model["available"] = np.asarray(False)
    output = []
    keys = ("split", "trial_id", "world_slot", "semantic_world_id", "delivery", "menu_layout", "known_attribute",
            "oracle_bits", "visible_problem_sha256", "seen_in_fit", "available", "source_flags")
    for index, row in enumerate(rows):
        score = predict(model, features[index:index + 1])[0] if available and row["available"] else None
        if score is not None:
            require(np.isfinite(score).all(), "Nonfinite regression prediction")
        output.append({**{k: row[k] for k in keys}, "scores": None if score is None else score.tolist(),
                       "predicted_bits": None if score is None else (score >= 0.5).astype(int).tolist(),
                       "shortcut_bits": reference[str(sum(row["oracle_bits"]))]["bits"],
                       "prediction_status": "predicted" if score is not None else (
                           "capture_unavailable" if not row["available"] else "model_unavailable")})
    return model, diagnostics, reference, output


def execute(project, frozen_plan_or_manifest, output_dir):
    """Root-only invocation after immutable admission. Never retries or reuses a directory."""
    project, output_dir = Path(project).resolve(), Path(output_dir)
    manifest = read(frozen_plan_or_manifest) if isinstance(frozen_plan_or_manifest, (str, Path)) else frozen_plan_or_manifest
    require(manifest.get("schema_version") == 1 and manifest.get("assay") == ASSAY
            and manifest.get("admitted") is True, "Explicit frozen root admission required")
    relative = manifest.get("output_relative_path")
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "One project-relative output must be frozen")
    require(output_dir.resolve() == (project / relative).resolve(), "Output differs from the sole admitted destination")
    ledger = manifest["input_ledger"]
    require(digest(ledger) == manifest["input_ledger_sha256"], "Frozen ledger digest mismatch")
    require(prepare(project) == ledger, "Reprepared inputs/code/plan differ from frozen ledger")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_new(output_dir / "started.json", dict(manifest_sha256=digest(manifest), input_ledger_sha256=digest(ledger)))
    write_new(output_dir / "input-ledger.json", ledger)
    try:
        features = load_features(project, ledger["rows"])
        model, diagnostics, reference, predictions = build_predictions(ledger["rows"], features)
        assessment = assess_predictions(predictions, model_available=bool(model["available"]))
        verify_bound_files(project, ledger)
        if bool(model["available"]):
            with (output_dir / "model.npz").open("xb") as stream:
                np.savez_compressed(stream, **model)
        prediction_object = dict(schema_version=1, rows=predictions)
        write_new(output_dir / "predictions.json", prediction_object)
        names = ["input-ledger.json", "predictions.json", "started.json"]
        if bool(model["available"]):
            names.append("model.npz")
        artifacts = {name: dict(path=name, byte_sha256=sha(output_dir / name)) for name in names}
        artifacts["predictions.json"]["content_sha256"] = digest(prediction_object)
        artifacts["input-ledger.json"]["content_sha256"] = digest(ledger)
        result = dict(schema_version=1, assay=ASSAY, complete=True, model_available=bool(model["available"]),
                      manifest_sha256=digest(manifest), input_ledger_sha256=digest(ledger),
                      method=METHOD, artifacts=artifacts, model_diagnostics=diagnostics, shortcut=reference,
                      **assessment, causal_admission=False, treated_distribution_validated=False,
                      retrospective_boundary=True)
        verify_bound_files(project, ledger)
        write_new(output_dir / "result.json", result)
        return result
    except BaseException as error:
        write_new(output_dir / "failure.json", dict(manifest_sha256=digest(manifest),
                  input_ledger_sha256=digest(ledger), complete=False, instrument_feasibility_passed=False,
                  causal_admission=False, error_type=type(error).__name__, error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("execute")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_new(args.out, prepare(args.project))
    else:
        execute(args.project, args.manifest, args.out)


if __name__ == "__main__":
    main()
