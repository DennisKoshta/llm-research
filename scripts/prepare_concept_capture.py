"""Validate reviewed materials and explicitly freeze a plain-text capture run.

This prepares artifacts only. It does not fabricate materials, load a model, fit
an axis, reserve a budget, or launch a worker.
"""

import argparse
import copy
import hashlib
from pathlib import Path
import sys

from curiosity_study import concept_capture as concept
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.runner import source_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402


def byte_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_review(review, corpus_hashes):
    concept.require(
        review.get("schema_version") == 1
        and review.get("passed") is True
        and review.get("corpus_byte_sha256") == corpus_hashes
        and review.get("reviewer") == "design_review"
        and review.get("scope") == "structural_and_semantic_prefreeze_review"
        and isinstance(review.get("limitations"), list)
        and review["limitations"]
        and all(isinstance(value, str) and value.strip() for value in review["limitations"]),
        "Require the passing, exactly corpus-bound independent material-review receipt",
    )


def build_manifest(corpora, model, material_bindings, review, review_binding, document_bindings):
    review_hashes = {row["corpus_id"]: row["byte_sha256"] for row in material_bindings}
    concept.require(
        len(material_bindings) == 2 and set(review_hashes) == {"A", "B"}, "Bind both corpus files"
    )
    validate_review(review, review_hashes)
    concept.require(model == concept.MODEL, "Concept capture requires the fixed Qwen3-32B revision")
    for row in material_bindings:
        concept.require(
            row["canonical_sha256"] == digest(corpora[row["corpus_id"]]), "Corpus canonical binding mismatch"
        )
    rows = concept.assignments(corpora)
    return dict(
        schema_version=3,
        assay="concept_capture",
        task_revision=concept.REVISION,
        phase="materials_capture",
        inference_scope="development",
        operation="capture",
        model=copy.deepcopy(model),
        execution_gpu="H200",
        execution=dict(backend="torch_batched_v1", batch_size=32),
        capture_backend="state_backend_v1",
        capture_plan=dict(layers=concept.LAYERS, positions=concept.POSITIONS),
        max_new_tokens=0,
        suffix=concept.SUFFIX,
        source_sha256=source_hash(),
        task_module_sha256=byte_sha(concept.__file__),
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        assignment_order=copy.deepcopy(concept.ASSIGNMENT_ORDER),
        assignment_order_sha256=digest(concept.ASSIGNMENT_ORDER),
        materials=copy.deepcopy(corpora),
        material_bindings=material_bindings,
        material_validation=concept.validate_corpora(corpora),
        material_review=copy.deepcopy(review),
        material_review_binding=review_binding,
        document_bindings=document_bindings,
        text_policy="Original HF tokenizer, no added special tokens, no chat template, no generated reasoning; narratives describe a narrator and do not assert model self-attribution.",
        cohort_order="Corpus A then B; sorted scenario IDs. Seed2026101433 independently assigns six positive-first and six negative-first scenarios per corpus/contrast and shuffles contrast-pair order within each scenario. Adjacent matched pairs,16-row scenarios and two complete scenarios per cohort are preserved. Exactly12 full cohorts; no padding assignments.",
        scores_policy="Initial A/B logits and probabilities are engineering diagnostics only. No correctness or concept-label outcome is scored.",
        analysis_policy="Capture all assigned texts at all frozen sites. No selection, fitting, mean pooling, or state-induction claim. A separately frozen analysis is required before fitting axes.",
        invalid_output_policy="terminal_execution_failure_no_retry",
        interpretation="Mechanical and independent material review do not establish psychological validity or subjective experience. This assay certifies only read-only narrative representation capture, separately from natural generation.",
    )


def prepare(corpus_a, corpus_b, model_config, material_review, documents, out, root=ROOT):
    root, out = Path(root).resolve(), Path(out).resolve()
    concept.require(not out.exists(), "Concept collections are immutable; choose a new output directory")
    paths = {"A": Path(corpus_a).resolve(), "B": Path(corpus_b).resolve()}
    model_config, material_review = Path(model_config).resolve(), Path(material_review).resolve()
    documents = [Path(path).resolve() for path in documents]
    concept.require(bool(documents), "Supply the prospective methods document explicitly")
    all_paths = [*paths.values(), model_config, material_review, *documents]
    concept.require(len(set(paths.values())) == 2, "A and B require distinct source files")
    names = sorted({str(path.relative_to(root)) for path in all_paths})
    corpora = {name: read(path) for name, path in paths.items()}
    configured = read(model_config)
    model = (
        dict(model_id=concept.MODEL["model_id"], revision=configured[concept.MODEL["model_id"]])
        if concept.MODEL["model_id"] in configured
        else configured
    )
    material_bindings = [
        dict(
            corpus_id=name,
            path=str(path.relative_to(root)),
            byte_sha256=byte_sha(path),
            canonical_sha256=digest(corpora[name]),
        )
        for name, path in paths.items()
    ]
    review = read(material_review)
    review_binding = dict(
        path=str(material_review.relative_to(root)),
        byte_sha256=byte_sha(material_review),
        canonical_sha256=digest(review),
    )
    bindings = [dict(path=name, byte_sha256=byte_sha(root / name)) for name in names]
    manifest = build_manifest(corpora, model, material_bindings, review, review_binding, bindings)
    concept.validate_manifest(manifest)
    # All validation precedes creating any run or execution artifacts.
    write_once(out / "manifest.json", manifest)
    frozen = freeze(root, out, names)
    return dict(
        run=str(out),
        assigned=len(manifest["diagnostics"]),
        manifest_sha256=digest(manifest),
        source_sha256=frozen["source_sha256"],
        gpu_launched=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-a", type=Path, required=True)
    parser.add_argument("--corpus-b", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--material-review", type=Path, required=True)
    parser.add_argument("--document", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        prepare(
            args.corpus_a, args.corpus_b, args.model_config, args.material_review, args.document, args.out
        )
    )
