"""Read-only residual captures of independently authored, matched descriptions.

These are representations of narrative text, not induced states or self-reports.
No contrast direction or semantic outcome is fitted or scored by this module.
"""

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import random
import re
import time
import uuid

import numpy as np

from .artifacts import RunStore, digest, write_once
from .state_backend import StateBackend

REVISION = "short-text-concept-capture-v1"
SUFFIX = "\nEnd of description."
LAYERS = [15, 31, 47, 57]
POSITIONS = ["record", "final"]
MODEL = dict(model_id="Qwen/Qwen3-32B", revision="9216db5781bf21249d130ec9da846c4624c16137")
ASSIGNMENT_ORDER = dict(
    schema_version=1,
    rule="scenario_pair_polarity_counterbalance_v1",
    seed=2026101433,
    corpus_order=["A", "B"],
    scenario_order="lexicographic scenario_id",
    member_order="Exactly six positive-first and six negative-first scenarios per corpus and contrast; independently keyed by corpus and contrast.",
    pair_order="Seeded shuffle of the eight contrast pairs independently within each corpus and scenario.",
    random_stream="Python Random with integer SHA256 of the canonical seed, purpose, corpus and contrast/scenario identity.",
    scenario_rows=16,
    scenarios_per_cohort=2,
    padding_assignments=False,
)
CONTRASTS = (
    "curiosity_vs_indifference",
    "satisfaction_vs_neutral",
    "confidence_high_vs_low",
    "positive_valence_vs_neutral",
    "completion_vs_ongoing",
    "generic_motivation_vs_indifference",
    "novelty_interest_vs_indifference",
    "task_demand_vs_no_demand",
)
PAIR_FIELDS = {
    "id",
    "scenario_id",
    "domain",
    "contrast",
    "shared_facts",
    "positive",
    "negative",
    "intended_difference",
    "residual_confounds",
}
BANNED = re.compile(r"\b(?:curious|curiosity|satisfied|satisfaction)\b", re.IGNORECASE)
WORDS = re.compile(r"\b\w+(?:['’\-]\w+)*\b")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate_corpora(corpora):
    require(isinstance(corpora, dict) and set(corpora) == {"A", "B"}, "Require exactly corpora A and B")
    authors, all_texts, summaries = [], set(), []
    for name in ("A", "B"):
        corpus = corpora[name]
        require(
            isinstance(corpus, dict)
            and corpus.get("schema_version") == 1
            and corpus.get("corpus_id") == name,
            "Corpus identity/schema mismatch",
        )
        require(
            all(
                isinstance(corpus.get(key), str) and corpus[key].strip()
                for key in ("author_id", "authorship")
            ),
            "Require author_id and an independent-authorship declaration",
        )
        authors.append(corpus["author_id"])
        pairs = corpus.get("pairs")
        require(
            isinstance(pairs, list) and len(pairs) == 96, "Each corpus requires 12 scenarios by eight pairs"
        )
        ids, scenarios, domains, word_counts = set(), defaultdict(Counter), defaultdict(set), []
        for pair in pairs:
            require(isinstance(pair, dict) and PAIR_FIELDS <= pair.keys(), "Incomplete matched-pair metadata")
            require(
                all(
                    isinstance(pair[key], str) and pair[key].strip()
                    for key in PAIR_FIELDS - {"residual_confounds"}
                ),
                "Matched-pair fields must be nonempty strings",
            )
            confounds = pair["residual_confounds"]
            require(
                (isinstance(confounds, str) and bool(confounds.strip()))
                or (
                    isinstance(confounds, list)
                    and bool(confounds)
                    and all(isinstance(item, str) and item.strip() for item in confounds)
                ),
                "Residual confounds must be explicit nonempty prose or a list",
            )
            require(
                pair["id"] not in ids and pair["contrast"] in CONTRASTS, "Duplicate pair or unknown contrast"
            )
            ids.add(pair["id"])
            scenarios[pair["scenario_id"]][pair["contrast"]] += 1
            domains[pair["scenario_id"]].add(pair["domain"])
            lengths = []
            for polarity in ("positive", "negative"):
                text = pair[polarity]
                require(
                    text == text.strip() and text.startswith(pair["shared_facts"]),
                    "Every narrative must begin with its exact shared facts and have no outer whitespace",
                )
                require(not BANNED.search(text), "Explicit target concept terms are banned in narrative text")
                require(text not in all_texts, "Narrative texts must be distinct within and across corpora")
                require(SUFFIX.strip() not in text, "Narrative must not contain the capture suffix")
                all_texts.add(text)
                lengths.append(len(WORDS.findall(text)))
            require(all(35 <= length <= 75 for length in lengths), "Narratives must contain 35 to 75 words")
            require(
                max(lengths) <= 1.25 * min(lengths), "Within-pair word lengths must differ by at most 25%"
            )
            word_counts.extend(lengths)
        require(
            len(scenarios) == 12 and all(counts == Counter(CONTRASTS) for counts in scenarios.values()),
            "Every one of 12 scenarios requires exactly one pair for every contrast",
        )
        require(
            all(len(values) == 1 for values in domains.values()),
            "A scenario must retain its domain across contrasts",
        )
        summaries.append(
            dict(
                corpus_id=name,
                author_id=corpus["author_id"],
                pairs=96,
                texts=192,
                scenarios=12,
                minimum_words=min(word_counts),
                maximum_words=max(word_counts),
            )
        )
    require(len(set(authors)) == 2, "Corpora A and B require distinct author identifiers")
    return dict(
        passed=True,
        corpora=summaries,
        texts=384,
        independence="Authorship is declared and bound; independent material review is still required.",
        semantic_validity="Mechanical material constraints do not establish the intended psychological contrast.",
    )


def assignments(corpora):
    validate_corpora(corpora)
    rows = []
    for corpus_id in ("A", "B"):
        pairs = {(p["scenario_id"], p["contrast"]): p for p in corpora[corpus_id]["pairs"]}
        scenarios = sorted({key[0] for key in pairs})

        def stream(purpose, **identity):
            seed = int(
                digest(dict(seed=ASSIGNMENT_ORDER["seed"], purpose=purpose, corpus_id=corpus_id, **identity)),
                16,
            )
            return random.Random(seed)

        positive_first = {
            contrast: set(stream("pair_member_order", contrast=contrast).sample(scenarios, 6))
            for contrast in CONTRASTS
        }
        for scenario_index, scenario in enumerate(scenarios):
            contrast_order = list(CONTRASTS)
            stream("contrast_pair_order", scenario_id=scenario).shuffle(contrast_order)
            for contrast in contrast_order:
                pair = pairs[(scenario, contrast)]
                # Trial identity is stable under execution-order counterbalancing.
                index = scenario_index * len(CONTRASTS) + CONTRASTS.index(contrast)
                polarities = (
                    ("positive", "negative")
                    if scenario in positive_first[contrast]
                    else ("negative", "positive")
                )
                for polarity in polarities:
                    text = pair[polarity]
                    rows.append(
                        dict(
                            trial_id=f"concept-{corpus_id}-{index:03d}-{polarity}",
                            task_revision=REVISION,
                            kind="concept_capture",
                            corpus_id=corpus_id,
                            pair_id=pair["id"],
                            scenario_id=pair["scenario_id"],
                            domain=pair["domain"],
                            contrast=pair["contrast"],
                            polarity=polarity,
                            text=text,
                            text_sha256=text_sha(text),
                            pair_sha256=digest(pair),
                        )
                    )
    return rows


def validate_manifest(manifest):
    rows = assignments(manifest["materials"])
    require(
        manifest.get("assay") == "concept_capture"
        and manifest.get("task_revision") == REVISION
        and manifest.get("operation") == "capture",
        "Unsupported concept-capture assay",
    )
    require(manifest.get("model") == MODEL, "Require the pinned Qwen3-32B model")
    require(
        manifest.get("execution") == dict(backend="torch_batched_v1", batch_size=32)
        and manifest.get("capture_backend") == "state_backend_v1",
        "Require the frozen batch32 capture backend",
    )
    require(
        manifest.get("capture_plan") == dict(layers=LAYERS, positions=POSITIONS)
        and manifest.get("max_new_tokens") == 0
        and manifest.get("suffix") == SUFFIX,
        "Concept capture requires all frozen sites and no generated tokens",
    )
    require(
        manifest.get("assignments") == []
        and manifest.get("diagnostics") == rows
        and manifest.get("assignment_sha256") == digest(rows),
        "Concept assignment coverage/order mismatch",
    )
    require(
        manifest.get("assignment_order") == ASSIGNMENT_ORDER
        and manifest.get("assignment_order_sha256") == digest(ASSIGNMENT_ORDER),
        "Concept assignment counterbalance policy differs from the frozen layout",
    )
    bindings = manifest.get("material_bindings", [])
    require(
        len(bindings) == 2 and {b["corpus_id"] for b in bindings} == {"A", "B"}, "Bind both raw corpus files"
    )
    hashes = {b["corpus_id"]: b["byte_sha256"] for b in bindings}
    require(
        all(b["canonical_sha256"] == digest(manifest["materials"][b["corpus_id"]]) for b in bindings),
        "Embedded corpus differs from frozen material binding",
    )
    review = manifest.get("material_review", {})
    require(
        review.get("schema_version") == 1
        and review.get("passed") is True
        and review.get("corpus_byte_sha256") == hashes
        and review.get("reviewer") == "design_review"
        and review.get("scope") == "structural_and_semantic_prefreeze_review"
        and isinstance(review.get("limitations"), list)
        and review["limitations"]
        and all(isinstance(item, str) and item.strip() for item in review["limitations"])
        and manifest.get("material_review_binding", {}).get("canonical_sha256") == digest(review),
        "Require the exact independently reviewed materials, with review limitations",
    )
    return rows


def inference_request(assignment, tokenizer):
    text = assignment["text"]
    require(
        text and text == text.strip() and assignment["text_sha256"] == text_sha(text),
        "Narrative text hash mismatch",
    )
    prompt = text + SUFFIX
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), [list(v) for v in encoded["offset_mapping"]]
    require(
        ids == tokenizer.encode(prompt, add_special_tokens=False) and len(ids) == len(offsets) and bool(ids),
        "Offset and exact-prefix tokenization must agree",
    )
    end = len(text) - 1
    candidates = [i for i, (start, stop) in enumerate(offsets) if start <= end < stop]
    require(len(candidates) == 1, "Ambiguous narrative-end token")
    record = candidates[0]
    require(
        not prompt[len(text) : offsets[record][1]].strip() and record < len(ids) - 1,
        "Narrative-end token includes nonwhitespace suffix content",
    )
    require(offsets[-1][1] == len(prompt), "Final token must reach the neutral suffix end")
    request = dict(
        token_prefix=ids,
        token_prefix_sha256=digest(ids),
        prefix_positions=dict(record=record, final=len(ids) - 1),
        labels=["A", "B"],
        seed=0,
    )
    binding = dict(
        raw_text_sha256=text_sha(text),
        prompt=prompt,
        prompt_text_sha256=text_sha(prompt),
        offset_mapping=offsets,
        add_special_tokens=False,
        apply_chat_template=False,
        token_prefix_sha256=digest(ids),
        prefix_positions=request["prefix_positions"],
        labels_role="Initial A/B scores are engineering diagnostics only; no semantic response is scored.",
    )
    return request, binding


def _logit_hashes(model, requests, layers):
    observed = []

    def observe(module, args, output):
        values = output.logits.detach().cpu().contiguous()
        observed.append(
            [
                digest(
                    dict(
                        dtype=str(values.dtype),
                        shape=list(row.shape),
                        bytes_sha256=hashlib.sha256(
                            row.contiguous().view(model.torch.uint8).numpy().tobytes()
                        ).hexdigest(),
                    )
                )
                for row in values
            ]
        )

    handle = model.model.register_forward_hook(observe)
    try:
        rows, hidden = StateBackend(model).forward(
            requests, layers=layers, positions=POSITIONS, max_new_tokens=0
        )
    finally:
        handle.remove()
    require(len(observed) == 1, "Concept capture must perform exactly one prefill and no decoding")
    return rows, hidden, observed[0]


def verify_capture(model, requests, layers):
    """Same actual row layout, without and with read-only capture hooks."""
    baseline, _, before = _logit_hashes(model, requests, [])
    captured, hidden, after = _logit_hashes(model, requests, layers)
    fields = (
        "unpadded_input_ids",
        "input_ids",
        "attention_mask",
        "position_ids",
        "label_logits",
        "greedy_first_token_id",
    )
    checks = dict(
        full_initial_logits_equal=before == after,
        every_row_fingerprint_equal=all(
            all(a[k] == b[k] for k in fields) for a, b in zip(baseline, captured)
        ),
        all_captures_finite=bool(np.isfinite(hidden).all()),
        capture_shape=list(hidden.shape) == [len(requests), len(layers), 2, model.model.config.hidden_size],
        no_intervention_or_generation=all(
            r["intervention"] is None and "completion" not in r for r in baseline + captured
        ),
        hook_cleanup=model.active_hooks == 0 and all(r["hooks_remaining"] == 0 for r in baseline + captured),
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        layers=list(layers),
        positions=POSITIONS,
        source_prefix_sha256=[r["token_prefix_sha256"] for r in requests],
        batch_size=len(requests),
        prefill_width=baseline[0]["prefill_width"],
        baseline_full_initial_logits_sha256=before,
        captured_full_initial_logits_sha256=after,
        capability="Read-only plain-text prefill captures only; does not certify natural generation or interventions.",
    )


def execute(root, manifest, model, checkpoint=None, deadline=None):
    rows = validate_manifest(manifest)
    root = Path(root)
    store = RunStore(root, manifest)
    completed = store.completed()
    require(set(completed) <= {a["trial_id"] for a in rows}, "Unexpected completed concept assignment")
    attempt, started = uuid.uuid4().hex, time.monotonic()
    metadata = model.metadata()
    metadata["concept_capture_hidden_size"] = model.model.config.hidden_size
    require(
        all(metadata.get(k) == v for k, v in manifest["model"].items()),
        "Loaded model differs from frozen concept model",
    )
    write_once(root / "environments" / f"{attempt}.json", metadata)
    prepared = [inference_request(a, model.tokenizer) for a in rows]
    integrity = verify_capture(model, [request for request, _ in prepared[:32]], LAYERS)
    integrity.update(manifest_sha256=digest(manifest), cohort_trial_ids=[a["trial_id"] for a in rows[:32]])
    write_once(root / "concept_integrity" / f"{attempt}.json", integrity)
    if checkpoint:
        checkpoint()
    require(integrity["passed"], "Read-only concept capture changed initial model output")
    backend = StateBackend(model)
    for offset in range(0, len(rows), 32):
        cohort = rows[offset : offset + 32]
        present = [a["trial_id"] in completed for a in cohort]
        require(not any(present) or all(present), "Cannot resume a partially published concept cohort")
        if all(present):
            continue
        if deadline is not None and time.time() >= deadline:
            break
        ids = [a["trial_id"] for a in cohort]
        selected = prepared[offset : offset + 32]
        cohort_id = f"{attempt}-{offset:06d}"
        try:
            measurements, hidden = backend.forward(
                [request for request, _ in selected], layers=LAYERS, positions=POSITIONS, max_new_tokens=0
            )
            require(
                hidden.shape == (32, 4, 2, model.model.config.hidden_size) and np.isfinite(hidden).all(),
                "Every text requires finite captures at every frozen site",
            )
            relative = Path("activations") / attempt / f"{offset:06d}.npz"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            positions = [[request["prefix_positions"][p] for p in POSITIONS] for request, _ in selected]
            with path.open("xb") as handle:
                np.savez_compressed(
                    handle,
                    hidden=hidden,
                    available=np.isfinite(hidden).all(-1),
                    layers=LAYERS,
                    positions=POSITIONS,
                    trial_ids=ids,
                    token_positions=positions,
                    token_prefix_sha256=[request["token_prefix_sha256"] for request, _ in selected],
                )
            artifact_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            cohort_records = []
            for row, (assignment, measurement, (_, tokenization)) in enumerate(
                zip(cohort, measurements, selected)
            ):
                layout = dict(
                    id=cohort_id,
                    offset=offset,
                    row=row,
                    size=32,
                    trial_ids=ids,
                    prefix_hashes=[r["token_prefix_sha256"] for r, _ in selected],
                    prefill_width=measurement["prefill_width"],
                )
                cohort_records.append(
                    dict(
                        trial_id=assignment["trial_id"],
                        assignment=assignment,
                        kind="concept_capture",
                        terminal=True,
                        measurement=measurement,
                        tokenization=tokenization,
                        cohort=layout,
                        cohort_sha256=digest(layout),
                        activation_artifact=dict(
                            path=str(relative),
                            sha256=artifact_sha,
                            row=row,
                            layers=LAYERS,
                            positions=POSITIONS,
                            token_positions=positions[row],
                            available=[[True, True]] * 4,
                        ),
                        scores_role="engineering_only_initial_scores; no semantic outcome",
                    )
                )
        except Exception as error:
            store.save_many(
                [
                    dict(
                        trial_id=a["trial_id"],
                        assignment=a,
                        kind="concept_capture",
                        terminal=True,
                        failure="execution_error",
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                    for a in cohort
                ],
                attempt,
                f"{offset:06d}",
            )
            if checkpoint:
                checkpoint()
            raise
        store.save_many(cohort_records, attempt, f"{offset:06d}")
        completed.update({r["trial_id"]: r for r in cohort_records})
        if checkpoint:
            checkpoint()
        print(f"{len(completed)}/{len(rows)} plain-text concept captures", flush=True)
    write_once(
        root / "profiles" / f"{attempt}.json",
        dict(
            completed=len(completed), assigned=len(rows), seconds=time.monotonic() - started, max_new_tokens=0
        ),
    )
    if checkpoint:
        checkpoint()
    return store.completed()


def render(manifest, records):
    return (
        "# Plain-text concept capture\n\n"
        f"Complete: {len(records)}/{len(manifest['diagnostics'])}. No answer or reasoning was generated.\n\n"
        "Captures describe model representations of authored narratives. Initial A/B scores are engineering diagnostics only. "
        "Material review and a separate frozen analysis are required before fitting directions. "
        "This collection does not induce, demonstrate, or measure subjective states.\n"
    )
