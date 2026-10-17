"""Independent, read-only checks for the fixed retrospective support decoder.

No producer imports, regression fit, model inference, or parameter selection.
The displayed evidence defines normative support, not internal confidence.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

import numpy as np

DELIVERIES = ("decisive_evidence", "redundant_evidence", "independent_novelty")
LABELS = ("W", "X", "Y", "Z")
SITE = (57, "decision")
RIDGE = 1 / 12
OVERLAP = "5e85fe2ddf66df3c53d08d031bc5c720402e6ead207a6cbe69ac2b26f68c5d3c"
RULES = (
    "The hidden candidate does not change. All property observations and reports are accurate, "
    "with no measurement noise. Decorative patterns are generated independently of the hidden "
    "candidate and give no evidence about it."
)
METHOD = dict(layer=57, position="decision", features=5120, fit_worlds=12, evaluation_worlds=12,
    fit_rows=72, evaluation_rows=72, dtype="float64", ridge_lambda=RIDGE,
    normalization="fit_only_center_and_scalar_rms_euclidean_norm", threshold=.5,
    exact_set_floor_numerator=9, exact_set_floor_denominator=10,
    shortcut="fit_modal_mask_given_oracle_cardinality_lexical_ties",
    shortcut_gain_numerator=1, shortcut_gain_denominator=4,
    retrospective_boundary=True, causal_admission=False)
CODE_FILES = {"reports/choice-support-readout-offline-v1-plan.md",
    "scripts/choice_support_readout_offline.py", "scripts/audit_choice_support_readout_offline.py",
    "tests/test_choice_support_readout_offline.py", "tests/test_audit_choice_support_readout_offline.py"}
SOURCE_FILES = {"src/curiosity_study/voluntary_resolution.py", "src/curiosity_study/information_state.py",
    "src/curiosity_study/reasoning_trace.py", "src/curiosity_study/batched.py",
    "src/curiosity_study/thinking_information.py"}
# Literal source identities from the prospective note; no producer imports.
PINS = {
    "v1": dict(manifest="c33d76eba3931aeacadf68f6282c7c1471be8edccec83e7ca2f37ab076e68feb",
        source="7e005358661c0de588faba66c9e9e5384984810abbc27385351ca5a2bdf8a911",
        archive="23254d727f0788da9a353f36c9adaf51e92aefaed19f8ecec4e96b131a2b79cf",
        audit="d1a4bfdbdc68ee1ce63d18709c6640d805d6fa065623f3c8ea0b452c3d9bf4bf",
        assignments="1052a14bf9e73cd00697e51e4c1602b1f026ac4a4ef0f00c7f120340997b289e",
        labels="b7fd0e2336cb8091c8e74d769d1e03bc132f2b88dea8826923a308d63eaa6165",
        captures="2277f187965240341451ebabbfe5ad871e3d14ff957e5e24054dcf66323768ca",
        inventory="c6c5635101a71b4c324ee5a380686cb9f7474c552a89060176d120acb408c70f",
        attempt="81fc07b16979410eba445ebb7a873ef8"),
    "v2": dict(manifest="b2ef2c264176003c9a0bcedf03a3e7bd80668cef38167d29f9ce3d9005861ab6",
        source="88a91e516e2580b23b9468bb1d38ba1c8a2b1e818e5f4e363ded74874a666623",
        archive="042a017d07d97ee149a7648608d7c51aeac4ef3f2f6d241f83179249ffe9a174",
        audit="aeec95021b15206e992ecf2a29655d128860b1c2230754d99d8fcac767a3412b",
        assignments="adde817547c7f6dee5a125fa616a3cf1cb98a56708ec89e01280ce05f0f6330d",
        labels="4508c0497c0b3a271c7c40ec2a21f0f99a60bd365661fa81e5a4cb1ca1f742e6",
        captures="a0c2f530df644c2994a92869dd41a5836ef6c977496a7350f1bd65a30e6ab0db",
        inventory="aaf41f020927602e2326ce00a2c30dedff991b4e002d0f98afc0da7fe35b5bc6",
        attempt="eeb0e3a93d644bf188d4c0dcfb1eee88"),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def displayed_support(prompt):
    """Parse only the four displayed rows and explicit prequestion observations."""
    require(isinstance(prompt, str) and prompt.count("\nEnd of record.") == 1,
        "One displayed record boundary required")
    context = prompt.split("\nEnd of record.", 1)[0]
    intro = "One hidden object is one of the four rows below. Every row was equally possible initially.\n"
    require(context.startswith(intro + "Candidate | "), "Object table/equal prior differs")
    header, body = context[len(intro + "Candidate | "):].split("\n", 1)
    attributes = header.split(" | ")
    require(len(attributes) == len(set(attributes)) == 2 and set(attributes) == {"color", "shape"},
        "Unexpected named attributes")
    table, observations = body.split("\nInitial observation:\n", 1)
    table = [line.split(" | ") for line in table.splitlines()]
    require(len(table) == 4 and all(len(row) == 3 for row in table)
        and [row[0] for row in table] == list(LABELS), "Candidate display coordinates differ")
    candidates = [row[1:] for row in table]
    require(len({tuple(row) for row in candidates}) == 4
        and all(len({row[col] for row in candidates}) == 2 for col in range(2)),
        "Require four distinct two-property candidates")
    initial, additional = observations.split("\nAdditional delivery:\n", 1)
    require(additional.endswith("\n" + RULES), "Noiseless independent-evidence rules differ")
    additional = additional[:-(len(RULES) + 1)]
    facts = {}
    initial_attribute = None
    for index, text in enumerate((initial, additional)):
        if index == 1 and text.startswith("Independent decorative pattern:\n"):
            lines = text.splitlines()[1:]
            require(len(lines) == 4 and all(re.fullmatch(r"[.*]( [.*]){3}", line) for line in lines),
                "Malformed independent pattern")
            continue
        match = re.fullmatch(r"The (\w+) is (\w+)\.", text)
        require(match is not None, "Only explicit displayed facts define support")
        attribute, value = match.groups()
        require(attribute in attributes and value in {row[attributes.index(attribute)] for row in candidates},
            "Fact does not name a table value")
        require(attribute not in facts or facts[attribute] == value, "Contradictory displayed evidence")
        facts[attribute] = value
        if index == 0:
            initial_attribute = attribute
    bits = [int(all(row[attributes.index(key)] == value for key, value in facts.items())) for row in candidates]
    require(sum(bits) in (1, 2), "Displayed support must be singleton or pair")
    order = sorted(range(2), key=attributes.__getitem__)
    named_table = dict(attributes=[attributes[i] for i in order],
        candidates=sorted([[row[i] for i in order] for row in candidates]))
    visible = dict(**named_table, observed=sorted([[key, value] for key, value in facts.items()]))
    return dict(candidate_labels=list(LABELS), compatible_bits=bits, cardinality=sum(bits),
        attributes=attributes, candidates=candidates, observations=facts,
        initial_attribute=initial_attribute, visible_problem=visible,
        visible_problem_sha256=digest(visible), named_table_sha256=digest(named_table))


def bit_mask(value, *, optional=False):
    if value is None and optional:
        return None
    require(isinstance(value, list) and len(value) == 4
        and all(type(x) is int and x in (0, 1) for x in value), "Malformed four-bit support mask")
    return tuple(value)


def shortcut_masks(training_rows):
    result = {}
    for cardinality in (1, 2):
        counts = Counter(bit_mask(row["compatible_bits"]) for row in training_rows
            if sum(row["compatible_bits"]) == cardinality)
        require(bool(counts), "Missing training cardinality")
        chosen = min(counts, key=lambda bits: (-counts[bits], bits))
        result[str(cardinality)] = dict(mask=list(chosen), count=counts[chosen],
            assigned=sum(counts.values()), frequencies={"".join(map(str, k)): v for k, v in sorted(counts.items())})
    return result


def row_inventory(rows):
    """World clustering and fixed 3x2 crossing; no answer-dependent filtering."""
    require(len(rows) == 72 and len({r["trial_id"] for r in rows}) == 72, "Require every assigned split row")
    groups = defaultdict(list)
    for row in rows:
        require(type(row["world_slot"]) is int and row["world_slot"] in range(12)
            and type(row["menu_layout"]) is int and row["menu_layout"] in (0, 1)
            and row["delivery"] in DELIVERIES, "Invalid assigned cell")
        bit_mask(row["compatible_bits"])
        bit_mask(row.get("predicted_bits"), optional=True)
        require(sum(row["compatible_bits"]) == (1 if row["delivery"] == DELIVERIES[0] else 2),
            "Delivery and independently reconstructed support differ")
        groups[row["semantic_world_id"]].append(row)
    require(len(groups) == 12 and {r["world_slot"] for r in rows} == set(range(12)),
        "Require twelve distinct assigned semantic worlds")
    for group in groups.values():
        require(len(group) == 6 and len({r["world_slot"] for r in group}) == 1
            and {(r["delivery"], r["menu_layout"]) for r in group}
            == {(d, m) for d in DELIVERIES for m in (0, 1)}, "World variants were split, duplicated or dropped")
        cells = {(r["delivery"], r["menu_layout"]): r for r in group}
        for d in DELIVERIES:
            require(cells[d, 0]["compatible_bits"] == cells[d, 1]["compatible_bits"],
                "Menu changed the oracle")
        decisive = set(np.flatnonzero(cells[DELIVERIES[0], 0]["compatible_bits"]).tolist())
        redundant = set(np.flatnonzero(cells[DELIVERIES[1], 0]["compatible_bits"]).tolist())
        require(decisive < redundant and cells[DELIVERIES[1], 0]["compatible_bits"]
            == cells[DELIVERIES[2], 0]["compatible_bits"], "Evidence/pattern oracle mismatch")
    return groups


def counted(rows, shortcut):
    exact = sum(row.get("predicted_bits") == row["compatible_bits"] for row in rows)
    modal = sum(shortcut[str(sum(row["compatible_bits"]))]["mask"] == row["compatible_bits"] for row in rows)
    cardinality = sum(row.get("predicted_bits") is not None
        and sum(row["predicted_bits"]) == sum(row["compatible_bits"]) for row in rows)
    return dict(assigned=len(rows), exact_correct=exact, cardinality_correct=cardinality,
        shortcut_correct=modal, paired_advantage_count=exact-modal)


def assess_rows(training_rows, evaluation_rows, *, integrity=True):
    """Independent integer gates, including unseen inputs and paired worlds."""
    require(type(integrity) is bool, "Integrity must be an explicit boolean")
    train_groups, test_groups = row_inventory(training_rows), row_inventory(evaluation_rows)
    require(not set(train_groups) & set(test_groups), "Training/evaluation exact worlds overlap")
    train_seen = {r["visible_problem_sha256"] for r in training_rows}
    masks = shortcut_masks(training_rows)
    cells, unseen_cells, strata, roles = [], [], [], []
    for delivery in DELIVERIES:
        for menu in (0, 1):
            selected = [r for r in evaluation_rows if (r["delivery"], r["menu_layout"]) == (delivery, menu)]
            unseen = [r for r in selected if r["visible_problem_sha256"] not in train_seen]
            n = 12 if delivery == DELIVERIES[0] else 11
            cells.append(dict(delivery=delivery, menu_layout=menu, **counted(selected, masks), minimum=11))
            unseen_cells.append(dict(delivery=delivery, menu_layout=menu, **counted(unseen, masks), minimum=math.ceil(.9*n)))
            require(len(unseen) == n, "Fixed visible-overlap denominators differ")
    for cardinality, gain in ((1, 6), (2, 12)):
        rows = [r for r in evaluation_rows if sum(r["compatible_bits"]) == cardinality]
        require(len(rows) == (24 if cardinality == 1 else 48), "Cardinality stratum denominator differs")
        strata.append(dict(cardinality=cardinality, **counted(rows, masks), minimum_advantage=gain))
    for attribute in ("color", "shape"):
        rows = [r for r in evaluation_rows if r["initial_attribute"] == attribute]
        require(len(rows) == 36, "Initially known property roles differ")
        roles.append(dict(initial_attribute=attribute, **counted(rows, masks)))
    paired = []
    relations = [("evidence_sensitivity", menu, [(DELIVERIES[0], menu), (DELIVERIES[1], menu)]) for menu in (0, 1)]
    relations += [("pattern_invariance", menu, [(DELIVERIES[1], menu), (DELIVERIES[2], menu)]) for menu in (0, 1)]
    relations += [("menu_invariance", delivery, [(delivery, 0), (delivery, 1)]) for delivery in DELIVERIES]
    for kind, stratum, keys in relations:
        success, unseen_success, unseen_count = 0, 0, 0
        for group in test_groups.values():
            by_cell = {(r["delivery"], r["menu_layout"]): r for r in group}
            pair = [by_cell[key] for key in keys]
            good = all(r.get("predicted_bits") == r["compatible_bits"] for r in pair)
            if good and kind == "evidence_sensitivity":
                good = set(np.flatnonzero(pair[0]["predicted_bits"]).tolist()) < set(np.flatnonzero(pair[1]["predicted_bits"]).tolist())
            elif good:
                good = pair[0]["predicted_bits"] == pair[1]["predicted_bits"]
            no_overlap = all(r["visible_problem_sha256"] not in train_seen for r in pair)
            success += int(good)
            unseen_count += int(no_overlap)
            unseen_success += int(good and no_overlap)
        paired.append(dict(kind=kind, stratum=stratum, assigned_worlds=12, correct_worlds=success,
            minimum=11, nonoverlap_worlds=unseen_count, nonoverlap_correct=unseen_success))
    confusion = []
    for j, label in enumerate(LABELS):
        counts = Counter("unavailable" if r.get("predicted_bits") is None else
            ("tp" if r["compatible_bits"][j] else "fp") if r["predicted_bits"][j] else
            ("fn" if r["compatible_bits"][j] else "tn") for r in evaluation_rows)
        confusion.append(dict(candidate=label, assigned=72, **{key: counts[key] for key in ("tp", "fp", "fn", "tn", "unavailable")}))
    gates = dict(integrity_passed=integrity,
        complete_predictions=all(r.get("predicted_bits") is not None for r in (*training_rows, *evaluation_rows)),
        primary_cells_passed=all(r["exact_correct"] >= r["minimum"] for r in cells),
        nonoverlap_cells_passed=all(r["exact_correct"] >= r["minimum"] for r in unseen_cells),
        shortcut_advantage_passed=all(r["paired_advantage_count"] >= r["minimum_advantage"] for r in strata),
        paired_checks_passed=all(r["correct_worlds"] >= r["minimum"] for r in paired))
    return dict(assigned_rows=144, train=counted(training_rows, masks), evaluation=counted(evaluation_rows, masks),
        shortcut=masks, primary_cells=cells, nonoverlap_cells=unseen_cells, cardinality_strata=strata,
        known_property_roles=roles, paired=paired, per_bit_confusion=confusion,
        all_six_correct_worlds=sum(all(r.get("predicted_bits") == r["compatible_bits"] for r in group)
            for group in test_groups.values()), gates=gates, instrument_feasibility_passed=all(gates.values()))


def verify_stored_model(x_train, y_train, x_test, model, train_scores, test_scores):
    """Verify one stored ridge solution without solve/lstsq or another fit."""
    x, y, xt = (np.asarray(value, dtype=np.float64) for value in (x_train, y_train, x_test))
    require(x.ndim == xt.ndim == 2 and x.shape[0] == xt.shape[0] == 72
        and x.shape[1] == xt.shape[1] and y.shape == (72, 4), "Fixed regression dimensions differ")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Missing training features make the fit unavailable")
    mu, ybar = x.mean(0), y.mean(0)
    scale = float(np.sqrt(np.mean(np.sum((x-mu)**2, axis=1))))
    require(math.isfinite(scale) and scale > 0, "Training scale is unavailable")
    w = np.asarray(model["W"], dtype=np.float64)
    require(w.shape == (x.shape[1], 4) and np.isfinite(w).all(), "Stored slopes differ")
    for key, expected in (("mu", mu), ("ybar", ybar), ("scale", scale)):
        actual = np.asarray(model[key], dtype=np.float64)
        require(actual.shape == np.asarray(expected).shape and np.isfinite(actual).all()
            and np.allclose(actual, expected, rtol=1e-12, atol=1e-12), "Train-only normalization differs: " + key)
    require(float(model["lambda"]) == RIDGE, "Ridge penalty changed")
    z, yc = (x-mu)/scale, y-ybar
    rhs = z.T @ yc
    residual = z.T @ (z @ w-yc) + (72*RIDGE)*w
    error = float(np.linalg.norm(residual))
    relative = error / max(1., float(np.linalg.norm(rhs)), float(np.linalg.norm(72*RIDGE*w)))
    require(relative <= 1e-10, "Stored model fails the fixed ridge normal equation")
    if "B" in model:
        b = np.asarray(model["B"], dtype=np.float64)
        require(b.shape == (72, 4) and np.isfinite(b).all()
            and np.allclose(z.T@b, w, atol=1e-11, rtol=1e-10)
            and np.allclose((z@z.T+6*np.eye(72))@b, yc, atol=1e-11, rtol=1e-10),
            "Stored dual solution differs from its fixed equation")
    predictions = []
    for data, supplied in ((x, train_scores), (xt, test_scores)):
        available = np.isfinite(data).all(1)
        expected = ybar + ((data[available]-mu)/scale)@w
        actual = np.asarray(supplied, dtype=np.float64)
        require(actual.shape == (72, 4) and np.isnan(actual[~available]).all()
            and np.isfinite(actual[available]).all()
            and np.allclose(actual[available], expected, rtol=1e-10, atol=1e-11)
            and np.array_equal(actual[available] >= .5, expected >= .5),
            "Stored prediction differs from fixed coefficients or imputes an unavailable row")
        independent = np.full((72, 4), np.nan)
        independent[available] = expected
        predictions.append([list(map(int, row >= .5)) if ok else None
            for row, ok in zip(independent, available, strict=True)])
    spectrum = np.linalg.eigvalsh(z@z.T)
    require(spectrum.min() >= -1e-10 and math.isclose(float(spectrum.sum()), 72., abs_tol=1e-9),
        "Training Gram matrix normalization differs")
    effective_df = float(np.sum(spectrum/(spectrum+6)))
    require(effective_df <= 12+1e-10, "Fixed ridge effective-df bound differs")
    singular = np.linalg.svd(z, compute_uv=False)
    return dict(normal_equation_relative_residual=relative, effective_df=effective_df,
        rank=int(np.count_nonzero(singular > singular[0]*1e-10)),
        train_predicted_bits=predictions[0], evaluation_predicted_bits=predictions[1])


class Reader:
    """Bind before parsing, reject changed rereads, and recheck all inputs at exit."""
    def __init__(self):
        self.files = {}

    def bytes(self, path, expected=None):
        path = Path(path).resolve()
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        require(expected is None or actual == expected, "Input byte hash differs: " + str(path))
        require(str(path) not in self.files or self.files[str(path)] == actual,
            "Input changed after first binding: " + str(path))
        self.files[str(path)] = actual
        return content

    def json(self, path, expected=None):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                require(key not in result, "Duplicate JSON key")
                result[key] = value
            return result
        return json.loads(self.bytes(path, expected), object_pairs_hook=unique,
            parse_constant=lambda value: require(False, "Nonfinite JSON: " + value))

    def npz(self, path, expected=None, names=None):
        import io
        with np.load(io.BytesIO(self.bytes(path, expected)), allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), "Duplicate NPZ arrays")
            return {key: archive[key] for key in (archive.files if names is None else names)}

    def unchanged(self):
        for path, expected in tuple(self.files.items()):
            self.bytes(path, expected)


def relative(root, path):
    path = Path(path)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe relative artifact path")
    result = (Path(root)/path).resolve()
    require(result.is_relative_to(Path(root).resolve()), "Artifact escapes its bound root")
    return result


def source_rows(project, collection, reader):
    """Narrow binding/label reconstruction; no original audit/scorer is rerun."""
    version = collection["version"]
    require(version in PINS and collection["pins"] == PINS[version], "Closed source pins changed")
    pin = PINS[version]
    prefix = f"runs/qwen32-voluntary-resolution-{version}"
    files = collection["files"]
    expected_paths = {f"{prefix}/{name}" for name in ("manifest.json", "freeze.json", "source.tar.gz")}
    expected_paths.add(f"reports/qwen32-voluntary-resolution-{version}-audit.json")
    for offset in range(0, 288, 32):
        for folder, suffix in (("cohorts", "json"), ("activations", "npz")):
            expected_paths.add(f"{prefix}/{folder}/{pin['attempt']}/{offset:07d}.{suffix}")
    require(set(files) == expected_paths and digest(files) == pin["inventory"], "Exact22 source files differ")
    for path, expected in files.items():
        reader.bytes(relative(project, path), expected)
    def read(path):
        return reader.json(relative(project, path), files[path])
    manifest, freeze = (read(f"{prefix}/{name}.json") for name in ("manifest", "freeze"))
    old_audit = read(f"reports/qwen32-voluntary-resolution-{version}-audit.json")
    require(digest(manifest) == freeze["manifest_sha256"] == old_audit["manifest_sha256"] == pin["manifest"]
        and manifest["source_sha256"] == freeze["source_sha256"] == old_audit["source_sha256"] == pin["source"]
        and files[f"{prefix}/source.tar.gz"] == freeze["archive_sha256"] == pin["archive"]
        and files[f"reports/qwen32-voluntary-resolution-{version}-audit.json"] == pin["audit"]
        and old_audit["complete"] is True and old_audit["integrity_passed"] is True,
        "Original closed provenance differs")
    audited = {r["trial_id"]: r for r in old_audit["outcomes"]}
    assignments = manifest["diagnostics"]
    require(len(assignments) == 288 and len(audited) == 288
        and digest(assignments) == manifest["assignment_sha256"], "Original assignment inventory differs")
    selected = sorted([a for a in assignments if a["family"] == "objects" and a["kind"] == "choice"],
        key=lambda a: (a["world_slot"], DELIVERIES.index(a["delivery"]), a["menu_layout"]))
    require(len(selected) == 72 and digest(selected) == pin["assignments"], "Fixed selected assignments differ")
    selected_ids = {a["trial_id"] for a in selected}
    records, captures, vectors, bindings = {}, {}, {}, []
    for offset in range(0, 288, 32):
        path = f"{prefix}/cohorts/{pin['attempt']}/{offset:07d}.json"
        cohort = read(path)
        apath = f"{prefix}/activations/{pin['attempt']}/{offset:07d}.npz"
        arrays = reader.npz(relative(project, apath), files[apath], names=("hidden", "layers", "positions",
            "trial_ids", "available", "token_positions", "closure_counts"))
        require(cohort["manifest_hash"] == pin["manifest"] and cohort["sha256"] == digest(cohort["records"])
            and [r["assignment"] for r in cohort["records"]] == assignments[offset:offset+32],
            "Original cohort order or content differs")
        require(arrays["hidden"].dtype == np.float32 and arrays["hidden"].shape == (32, 4, 7, 5120)
            and arrays["layers"].tolist() == [15, 31, 47, 57]
            and arrays["positions"].tolist() == ["record", "prefill", "token_32", "token_128", "token_512", "think_end", "decision"]
            and arrays["trial_ids"].tolist() == [a["trial_id"] for a in assignments[offset:offset+32]],
            "Activation schema or physical row identity differs")
        for index, record in enumerate(cohort["records"]):
            tid = record["trial_id"]
            require(tid not in records and digest(record) == audited[tid]["record_sha256"], "Record/audit bytes differ")
            artifact = record["activation_artifact"]
            require(artifact == audited[tid]["activation_artifact"]
                and artifact["path"] == f"activations/{pin['attempt']}/{offset:07d}.npz"
                and artifact["sha256"] == files[apath] and artifact["row"] == index
                and artifact["available"] == arrays["available"][index].tolist()
                and artifact["token_positions"] == arrays["token_positions"][index].tolist(),
                "Activation capture binding differs")
            bindings.append(dict(trial_id=tid, record_sha256=digest(record), artifact_sha256=files[apath],
                row=index, token_prefix_sha256=digest(record["inference"]["unpadded_input_ids"])))
            records[tid] = record
            if tid not in selected_ids:
                continue
            available = bool(arrays["available"][index, 3, 6])
            vector = arrays["hidden"][index, 3, 6]
            require(np.isfinite(vector).all() if available else np.isnan(vector).all(), "Decision availability differs")
            captures[tid] = dict(path=apath, byte_sha256=files[apath], row=index,
                layer=57, layer_index=3, position="decision", position_index=6,
                token_position=int(arrays["token_positions"][index, 6]), available=available,
                closure_count=int(arrays["closure_counts"][index]))
            vectors[tid] = vector.astype(np.float64)
    require(digest(bindings) == old_audit["generation_record_bindings_sha256"], "Original record/prefix ledger differs")
    import io
    require(set(collection["source_files"]) == SOURCE_FILES, "Archived source inventory differs")
    with tarfile.open(fileobj=io.BytesIO(reader.bytes(relative(project, f"{prefix}/source.tar.gz"))), mode="r:gz") as archive:
        for path, expected in collection["source_files"].items():
            member = archive.getmember(path)
            require(member.isfile() and hashlib.sha256(archive.extractfile(member).read()).hexdigest() == expected,
                "Version-specific archived source bytes differ")
    rows, labels, capture_ledger, xs = [], [], [], []
    for a in selected:
        tid = a["trial_id"]
        record, cap = records[tid], captures[tid]
        messages = record["inference"]["messages"]
        require(len(messages) == 2 and messages[0]["role"] == "system" and messages[1]["role"] == "user",
            "Source conversation schema differs")
        oracle = displayed_support(messages[1]["content"])
        require(messages[1]["content"] in record["inference"]["rendered_prompt"]
            and record["task"]["attributes"] == oracle["attributes"]
            and record["task"]["candidates"] == oracle["candidates"]
            and record["task"]["compatible"] == np.flatnonzero(oracle["compatible_bits"]).tolist()
            and a["known"] == [oracle["attributes"].index(oracle["initial_attribute"])],
            "Displayed-only reconstruction disagrees with bound task metadata")
        label = dict(trial_id=tid, assignment_sha256=digest(a), world_slot=a["world_slot"],
            semantic_world_id=a["semantic_world_id"], delivery=a["delivery"], menu_layout=a["menu_layout"],
            candidate_labels=list(LABELS), compatible_bits=oracle["compatible_bits"])
        labels.append(label)
        capture_ledger.append(dict(**label, record_sha256=digest(record),
            activation_path=record["activation_artifact"]["path"], activation_byte_sha256=cap["byte_sha256"],
            activation_row=cap["row"], layer=57, position="decision", available=cap["available"],
            token_position=cap["token_position"], closure_count=cap["closure_count"]))
        rows.append(dict(split="fit" if version == "v1" else "evaluation", collection=version,
            trial_id=tid, assignment_sha256=digest(a), record_sha256=digest(record), world_slot=a["world_slot"],
            semantic_world_id=a["semantic_world_id"], delivery=a["delivery"], menu_layout=a["menu_layout"],
            known_attribute=a["known"][0], oracle_bits=oracle["compatible_bits"],
            visible_problem_sha256=oracle["visible_problem_sha256"], available=cap["available"], capture=cap,
            source_flags=dict(valid=record["valid"], cap_hit=record["cap_hit"], thinking_closed=record["thinking_closed"],
                closure_count=cap["closure_count"])))
        xs.append(vectors[tid])
    require(digest(labels) == pin["labels"] and digest(capture_ledger) == pin["captures"],
        "Prespecified label/capture ledgers differ")
    return rows, np.asarray(xs, dtype=np.float64)


def reported_assessment(predictions, independent, model_available):
    """Translate independently counted quantities to the producer's public schema."""
    def cell(row):
        n, good = row["assigned"], row["exact_correct"]
        return dict(assigned=n, correct=good, required=(9*n+9)//10,
            passed=n > 0 and 10*good >= 9*n)
    primary = [dict(delivery=r["delivery"], menu_layout=r["menu_layout"], **cell(r))
        for r in independent["primary_cells"]]
    unseen = [dict(delivery=r["delivery"], menu_layout=r["menu_layout"], **cell(r))
        for r in independent["nonoverlap_cells"]]
    strata = [dict(cardinality=r["cardinality"], assigned=r["assigned"], correct=r["exact_correct"],
        shortcut_correct=r["shortcut_correct"], additional_correct=r["paired_advantage_count"],
        required_additional=r["minimum_advantage"], passed=r["paired_advantage_count"] >= r["minimum_advantage"])
        for r in independent["cardinality_strata"]]
    paired = {}
    for kind, saved_kind in (("evidence_sensitivity", "evidence_sensitivity"),
            ("pattern_invariance", "novelty_invariance"), ("menu_invariance", "menu_invariance")):
        paired[saved_kind] = [dict(condition=r["stratum"], assigned=r["assigned_worlds"],
            correct=r["correct_worlds"], required=r["minimum"], passed=r["correct_worlds"] >= r["minimum"],
            nonoverlap_assigned=r["nonoverlap_worlds"], nonoverlap_correct=r["nonoverlap_correct"])
            for r in independent["paired"] if r["kind"] == kind]
    summaries = {}
    for split in ("fit", "evaluation"):
        rows = [r for r in predictions if r["split"] == split]
        good = [r["predicted_bits"] == r["oracle_bits"] for r in rows]
        summaries[split] = dict(assigned=72, predicted=sum(r["predicted_bits"] is not None for r in rows),
            exact_correct=sum(good), cardinality_correct=sum(r["predicted_bits"] is not None
                and sum(r["predicted_bits"]) == sum(r["oracle_bits"]) for r in rows),
            all_six_correct_worlds=sum(all(r["predicted_bits"] == r["oracle_bits"] for r in rows
                if r["world_slot"] == slot) for slot in range(12)),
            source_invalid=sum(not r["source_flags"]["valid"] for r in rows),
            source_caps=sum(r["source_flags"]["cap_hit"] for r in rows),
            capture_unavailable=sum(not r["available"] for r in rows))
    bits = [dict(candidate=r["candidate"], **{k: r[k] for k in ("tp", "tn", "fp", "fn")},
        missing=r["unavailable"]) for r in independent["per_bit_confusion"]]
    roles = [dict(known_attribute=i, **cell(next(r for r in independent["known_property_roles"]
        if r["initial_attribute"] == name))) for i, name in enumerate(("color", "shape"))]
    checks = dict(provenance_and_grouping_passed=True,
        complete_captures=all(r["available"] for r in predictions), model_available=model_available,
        primary_exact_sets=all(r["passed"] for r in primary), nonoverlap_exact_sets=all(r["passed"] for r in unseen),
        shortcut_improvement=all(r["passed"] for r in strata),
        **{kind: all(r["passed"] for r in cells) for kind, cells in paired.items()})
    return dict(checks=checks, instrument_feasibility_passed=all(checks.values()),
        metrics=dict(summaries=summaries, primary_cells=primary, nonoverlap_cells=unseen,
            cardinality_strata=strata, paired=paired, per_bit=bits, known_property_roles=roles))


def audit(project, manifest_path, output_dir, *, code_root=None):
    """Read-only postrun audit. Root must authorize before any actual feature read."""
    project, output = Path(project).resolve(), Path(output_dir).resolve()
    code_root = Path(code_root).resolve() if code_root else Path(__file__).resolve().parents[1]
    reader = Reader()
    manifest = reader.json(manifest_path)
    require(manifest.get("schema_version") == 1 and manifest.get("assay") == "choice_support_readout_offline"
        and manifest.get("admitted") is True, "Require the frozen root-admitted manifest")
    require(relative(project, manifest["output_relative_path"]) == output,
        "Output directory differs from the one-shot frozen destination")
    ledger = manifest["input_ledger"]
    require(digest(ledger) == manifest["input_ledger_sha256"]
        and reader.json(output/"input-ledger.json") == ledger
        and ledger["method"] == METHOD and ledger["hidden_feature_values_read"] is False
        and ledger["causal_admission"] is False and set(ledger["code_files"]) == CODE_FILES,
        "Fixed method/ledger/code inventory differs")
    for path, expected in ledger["code_files"].items():
        reader.bytes(relative(code_root, path), expected)
    started = reader.json(output/"started.json")
    require(started == dict(manifest_sha256=digest(manifest), input_ledger_sha256=digest(ledger)),
        "One-shot attempt binding differs")
    require([c["version"] for c in ledger["collections"]] == ["v1", "v2"], "Fixed source split differs")
    reconstructed, matrices = [], []
    for collection in ledger["collections"]:
        rows, values = source_rows(project, collection, reader)
        reconstructed.extend(rows)
        matrices.append(values)
    train_seen = {r["visible_problem_sha256"] for r in reconstructed[:72]}
    for row in reconstructed:
        row["seen_in_fit"] = row["visible_problem_sha256"] in train_seen
    require(reconstructed == ledger["rows"] and digest(reconstructed) == ledger["rows_sha256"],
        "Independent source reconstruction differs from the admitted ledger")
    overlap = [r for r in reconstructed[72:] if r["seen_in_fit"]]
    require(len(overlap) == 4 and {r["visible_problem_sha256"] for r in overlap} == {OVERLAP}
        and {r["world_slot"] for r in overlap} == {3}
        and {(r["delivery"], r["menu_layout"]) for r in overlap}
            == {(d, m) for d in DELIVERIES[1:] for m in (0, 1)}
        and ledger["visible_overlap"] == dict(sha256=OVERLAP, evaluation_trial_ids=[r["trial_id"] for r in overlap]),
        "Exact input-defined overlap identity differs")
    if not (output/"result.json").exists():
        failure = reader.json(output/"failure.json")
        require(failure["manifest_sha256"] == digest(manifest) and failure["input_ledger_sha256"] == digest(ledger)
            and failure["complete"] is False and failure["instrument_feasibility_passed"] is False
            and failure["causal_admission"] is False, "Malformed failed-attempt receipt")
        reader.unchanged()
        return dict(schema_version=1, assay="choice_support_readout_offline_audit", complete=False,
            passed=False, computation_agrees=False, assigned_rows=144, retained_assignment_rows=reconstructed,
            instrument_feasibility_passed=False, causal_admission=False, failure=failure,
            input_files=reader.files, input_files_sha256=digest(reader.files),
            auditor_byte_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    require(not (output/"failure.json").exists(), "Successful result coexists with a terminal failure")
    result = reader.json(output/"result.json")
    require(result["schema_version"] == 1 and result["assay"] == "choice_support_readout_offline"
        and result["manifest_sha256"] == digest(manifest) and result["input_ledger_sha256"] == digest(ledger)
        and result["complete"] is True and type(result["model_available"]) is bool
        and result["method"] == METHOD and result["causal_admission"] is False
        and result["treated_distribution_validated"] is False and result["retrospective_boundary"] is True,
        "Result scope/admission differs")
    available = result["model_available"]
    expected_artifacts = {"input-ledger.json", "predictions.json", "started.json"} | ({"model.npz"} if available else set())
    require(set(result["artifacts"]) == expected_artifacts and (output/"model.npz").exists() is available,
        "Unavailable fit cannot invent a model, or available fit omit one")
    for name, binding in result["artifacts"].items():
        require(binding["path"] == name, "Output artifact path differs")
        reader.bytes(output/name, binding["byte_sha256"])
        if "content_sha256" in binding:
            require(digest(reader.json(output/name)) == binding["content_sha256"], "Output content digest differs")
    payload = reader.json(output/"predictions.json")
    require(payload["schema_version"] == 1 and len(payload["rows"]) == 144, "All144 predictions required")
    predictions = payload["rows"]
    base_keys = ("split", "trial_id", "world_slot", "semantic_world_id", "delivery", "menu_layout", "known_attribute",
        "oracle_bits", "visible_problem_sha256", "seen_in_fit", "available", "source_flags")
    internal = []
    for saved, row in zip(predictions, reconstructed, strict=True):
        require({k: saved[k] for k in base_keys} == {k: row[k] for k in base_keys},
            "Prediction row lost or changed assigned/source metadata")
        require(type(saved["available"]) is bool and type(saved["seen_in_fit"]) is bool
            and all(type(saved["source_flags"][k]) is bool for k in ("valid", "cap_hit", "thinking_closed")),
            "Source flags must be explicit booleans")
        bit_mask(saved["predicted_bits"], optional=True)
        expected_status = "predicted" if available and row["available"] else (
            "capture_unavailable" if not row["available"] else "model_unavailable")
        require(saved["prediction_status"] == expected_status
            and ((saved["scores"] is not None and saved["predicted_bits"] is not None) if expected_status == "predicted"
                else saved["scores"] is saved["predicted_bits"] is None), "Prediction availability differs")
        internal.append(dict(**saved, compatible_bits=saved["oracle_bits"],
            initial_attribute=("color", "shape")[saved["known_attribute"]]))
    if available:
        model = reader.npz(output/"model.npz", result["artifacts"]["model.npz"]["byte_sha256"])
        require(set(model) == {"mu", "scale", "ybar", "W", "B", "available", "lambda"}
            and model["available"].shape == () and model["available"].dtype == np.bool_
            and model["available"].item() is True
            and model["lambda"].shape == () and model["lambda"].item() == RIDGE
            and all(model[k].dtype == np.float64 for k in model if k != "available"), "Stored model dtype/inventory differs")
        scores = np.asarray([r["scores"] if r["scores"] is not None else [np.nan]*4 for r in predictions], dtype=np.float64)
        numerical = verify_stored_model(matrices[0], [r["oracle_bits"] for r in reconstructed[:72]], matrices[1],
            model, scores[:72], scores[72:])
        require([r["predicted_bits"] for r in predictions] == numerical["train_predicted_bits"]
            + numerical["evaluation_predicted_bits"], "Saved bits differ from independent model thresholding")
        diagnostics = result["model_diagnostics"]
        require(diagnostics["rank"] == numerical["rank"]
            and math.isclose(diagnostics["effective_df"], numerical["effective_df"], rel_tol=1e-10, abs_tol=1e-10)
            and diagnostics["n_features"] == 5120 and diagnostics["train_rows"] == 72
            and diagnostics["ridge_lambda"] == RIDGE, "Reported model diagnostics differ")
    else:
        numerical = dict(model_available=False, normal_equation_relative_residual=None,
            reason=result["model_diagnostics"].get("reason"))
        require(numerical["reason"] in ("missing_training_capture", "fixed_fit_unavailable")
            and result["model_diagnostics"]["rank"] is None and result["model_diagnostics"]["effective_df"] is None,
            "Unavailable model requires an explicit disposition")
        if numerical["reason"] == "missing_training_capture":
            require(any(not row["available"] for row in reconstructed[:72]), "Unavailable training state was invented")
        else:
            require(result["model_diagnostics"].get("error_type") in ("ValueError", "LinAlgError")
                and isinstance(result["model_diagnostics"].get("error"), str), "Unrecognized fixed-fit failure")
    independent = assess_rows(internal[:72], internal[72:])
    expected_shortcut = {k: dict(bits=v["mask"], counts=v["frequencies"]) for k, v in independent["shortcut"].items()}
    require(result["shortcut"] == expected_shortcut
        and all(r["shortcut_bits"] == expected_shortcut[str(sum(r["oracle_bits"]))]["bits"] for r in predictions),
        "Shortcut uses test labels, repairs predictions, or changes tie rule")
    recomputed = reported_assessment(predictions, independent, available)
    for key, value in recomputed.items():
        require(digest(result[key]) == digest(value), "Reported metrics/decision differ: " + key)
    reader.unchanged()
    return dict(schema_version=1, assay="choice_support_readout_offline_audit", complete=True,
        passed=True, computation_agrees=True, assigned_rows=144, fit_rows=72, evaluation_rows=72,
        manifest_sha256=digest(manifest), input_ledger_sha256=digest(ledger),
        result_byte_sha256=reader.files[str(output/"result.json")],
        auditor_byte_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        instrument_feasibility_passed=recomputed["instrument_feasibility_passed"], causal_admission=False,
        model_available=available, numerical_checks={k: v for k, v in numerical.items() if not k.endswith("predicted_bits")},
        independent_metrics=independent, reported_schema_recomputation=recomputed,
        input_files=reader.files, input_files_sha256=digest(reader.files),
        scope="Independent displayed support, fixed row/provenance bindings, model equation and all fixed gates. No refit, original model inference, old study re-audit, belief calibration or causal admission.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True, help="Immutable producer output directory")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.out.exists(), "Audit receipt is immutable")
    receipt = audit(arguments.project, arguments.manifest, arguments.analysis)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: receipt[k] for k in ("complete", "passed", "instrument_feasibility_passed")}))

