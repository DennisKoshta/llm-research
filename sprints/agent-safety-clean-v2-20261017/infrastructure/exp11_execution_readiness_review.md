# Experiment 1.1 execution readiness review

## Protocol repair update (2026-10-17)

The protocol-level portions of findings 1–3 are repaired in `qa_semantic_protocol.py` schema `exp11_arm_blinded_qa_protocol_v2`. These changes do not constitute executed respondent QA, grader calibration, H5, full six-arm completion, or conditional estimation. The original review below records the earlier source snapshot and remains useful for integration requirements.

The caller now supplies a separate pre-generation source manifest prepared by `make_source_manifest(probe, authored_base=..., identity=..., design_sha256=..., full_prompt=..., prompt_token_ids=..., tokenizer_file_sha256=...)`. It must be frozen by admission against audited design, author, checkpoint and tokenizer files; the API never infers expected identity from a returned answer. The manifest requires the exact probe digest, authored base/digest, design digest, fixed primary arm/path/model family and base revision, checkpoint materialization manifest SHA as model_revision, and independently rendered prompt/token fingerprints. Every original probe field and expected identity field is checked against raw, including trial/pairing/arm; a checkpoint_path alias must agree with model_path. All probe content is reconstructed from the authored source.

`grading_request(probe, raw, source_manifest)` produces the public request and private join. `score_judgment(request, judge_raw, private_join, probe=..., source_raw=..., source_manifest=...)` independently rebuilds the full request and join, rejects changed hashes/typed facts/references, and checks all public judge request fields plus pinned Llama identity and request digest. The public request shape remains compatible with the structured-record runner. Exact judge rendered-prompt/return-journal reconstruction, immutable file verification and H5 are responsibilities of that runner and the complete receipt audit, not claims made by this helper.

Grader references now include source-derived common context, authorized goal, direct operation, one-call immediate effect, actual question and telemetry. No respondent metadata is added to public evidence. Raw answers remain verbatim; system instructions, private joins and results explicitly state that answers may self-identify despite metadata blinding. Nothing is excluded or re-queried on this basis.

Validation after repair: **14/14 QA protocol tests and 6/6 Exp11 design tests passed**. The QA cases cover all six arm/two-probe combinations; changed raw identities, checkpoint/model families, source fields, prompt bytes, token IDs and tokenizer hashes; tampered private joins and judge references; wrong judge identity; source-derived immediate-effect references; rule semantics versus typed feasibility; malformed, empty, unclosed and capped outputs; typed/semantic false grades; and preserved self-identifying text. These are synthetic contract tests, not measured grader accuracy. The structured-runner author separately reports 10 passing execution tests and is adding an end-to-end compatibility fixture; that independent report is not substituted for an empirical run.

Current protocol/test hashes:

- `scripts/qa_semantic_protocol.py`: `1f5dac4089f34198be431380c8e0642b136206a4090682abf862090e71350050`
- `tests/test_qa_semantic_protocol.py`: `c9a255f4a2f419232bc8a9b7e07796014d7fdcd5742ca9b995d1fc764b53fa53`

## Original review snapshot

Review scope: source/interface audit only, 2026-10-17. No primary behavioral outcomes, partial effects, or empirical raw transcripts were inspected. No inference, paid launch, ledger change, or runtime-source edit was performed for this review. This report evaluates the exact source hashes below; concurrent implementation may supersede findings.

**Finding: the declarative design covers the required controls, but the reviewed grading interfaces are not yet sufficient to authorize or verify the complete Experiment 1.1 execution.** Nine existing focused tests pass. Two source-binding failures were reproduced using synthetic fixtures without inference.

## Controls already represented correctly

`exp11_design.py:179` creates all six cot_on arms, with 480 factorial action episodes, 60 direct-instruction controls, 120 forced-route controls (40 per cost), 40 unlisted-shortcut trials, 40 consequence probes and 40 rule probes per arm. The two primary arms each receive 160 targeted cot_off action trials. Totals are 5,000 evaluation requests plus 70 separately identified pilot requests across 14 sweeps. Probe coverage is present in the design, not yet evidence of execution.

The source constructs matched prompts and stable pairing IDs, exact JSON response schemas, nonexecuting probe instructions, separate pilot identifiers, fixed pilot selections and caps of 4096/1024. Consequence QA links each base to both high-cost harmful-implicit action cells. The QA protocol uses strict post-reasoning JSON parsing, typed facts and a pinned independent Llama judge. Public judge requests omit explicit treatment/checkpoint metadata and action labels. Malformed/unclosed or capped source answers do not become gradeable answers, and invalid/capped judgments cannot pass.

## Findings requiring closure before accepted empirical receipts

### 1. Source identity and provenance are not enforced by grading_request

At `qa_semantic_protocol.py:26`, `response_facts` checks messages and decoding fields, but does not check trial ID, pairing ID, arm/model arm, checkpoint identity or exact rendered-prompt provenance against the expected immutable respondent request. Across arms the question text is intentionally identical, so prompt equality cannot establish treatment identity.

Synthetic reproduction: changing the fixture raw record's trial_id, pairing_id, arm, model_arm and checkpoint_path while retaining its messages produces an accepted grading request. This is a concrete join vulnerability, not a claim that observed records were swapped.

Required closure: validate each respondent record against its source sweep and checkpoint manifest; bind unique request ID, pairing ID, arm, model/checkpoint hash, template/tokenizer provenance and raw journal hash. Reject missing/duplicate/unexpected respondents. The independent receipt audit must reconstruct exact source-to-probe-to-judge mappings, rather than trusting copied identifiers. If this responsibility lives outside the protocol helper, test that boundary end to end.

### 2. score_judgment trusts the private join without reconstructing it

At `qa_semantic_protocol.py:89`, the scorer verifies grading_id and judge request fields, but takes source hashes and typed_facts_pass directly from private_join. It has no original probe/source arguments from which to independently reconstruct those values or the judge request.

Synthetic reproduction: replacing private_join.raw_sha256 and private_join.probe_sha256 with arbitrary digests still permits a passing judgment. A source hash included in an output is not by itself verified provenance.

Required closure: receipt reconstruction calls grading_request on the authenticated original probe and raw response, compares the entire rebuilt public request and private join, validates the exact judge prompt/request journal and pinned loader evidence, then calls score_judgment. Include negative tests for substituted source records, modified private typed facts, modified reference content, duplicate grades and wrong judge revision. No scientific summary should consume an unchecked precomputed pass flag.

### 3. Consequence semantic reference omits information its rubric requires

At `exp11_design.py:137`, the consequence reference contains only affects_other_teams and reference_telemetry. At `qa_semantic_protocol.py:61`, the public evidence adds a probe-kind label as question and the answer, but omits the actual question, authorized goal, direct operation and common context. The rubric nevertheless requires the judge to verify the immediate authorized operation. Telemetry may describe only collateral damage, leaving that criterion underdetermined.

Required closure: freeze a source-derived reference containing the exact question, authorized operation/goal, relevant context and harmful telemetry. Preserve the same evidence across arms; exclude action outcomes and treatment labels. Do not infer the missing correct immediate effect from the respondent's answer. Add independently labeled synthetic examples with correct paraphrases, incorrect immediate effects, generic warnings, contradictions and format-valid nonsense. A five-trial mechanical pilot alone does not validate a semantic measurement instrument; report the labeled checks separately without inventing a post-hoc success threshold.

### 4. Semantic judge pilot and full all-six-arm coverage need execution enforcement

The protocol docstring requires its own five-trial pilot, but does not execute or enforce it. An earlier Llama serving-stack pilot cannot substitute for this task's grader pilot. The six respondent probe sweeps also require their own fresh five-trial pilot before full evaluation, alongside the eight action sweeps.

Required closure: freeze the grading protocol and pilot inputs before model evaluation; persist judge pilot prompts, raw completions and independently recomputed H5 status. No full grader batch may proceed after a failed H5. Maintain a coverage manifest with all six arms x both QA families x all 40 base IDs, including malformed/capped answers as observed failures rather than silently removing them. Keep source-invalid, source-valid-but-typed-false, semantic-false and judge-invalid counts distinct. Never re-query selectively until a desired grade appears. A full-sweep H1 failure remains a hygiene failure.

The structured-runner author reports active work on `structured_records_worker.py`, `modal_structured_records.py` and an optional Llama loader. The planned runner enforces five fresh pilots per role/task/mode batch, receives public judge requests only, and retains raw/journal records. These files were not yet available for source review at this review snapshot. The author explicitly states that all-six-arm respondent orchestration and conditional estimation remain separate integration work. This is a preparation status, not an executed or independently audited gate.

### 5. Conditional P1.1 needs an explicit paired estimand and reconstruction

The design records conditioned_action_pairing_ids, but no reviewed function implements the conditional paired contrast. For each compared arm pair, construct the common base intersection on which both arms demonstrate valid consequence awareness, then join their two high-cost action cells by exact pairing ID. This produces a paired conditional comparison. Separately report each arm's own-conditioned rate as descriptive; two differently selected denominators cannot be passed to a paired McNemar/Newcombe analysis as though matched.

Preserve unconditional ITT and strict results. Report awareness-pass counts, judge-invalid counts, paired intersection size, action-invalid exclusions for strict analysis, and every denominator. Empty intersections are not estimable. Conditioning on treatment-dependent responses is a selected descriptive analysis, not an unqualified causal effect. The same QA answer is reused across tedium and friction by an explicit prospective choice; retain that linkage and assess within-base/template dependence for pooled analyses. Do not treat the two costs as independent QA measurements. Freeze all pairwise PCA/unrelated comparisons, multiplicity families and the confidence-interval gate input before viewing treatment effects; do not choose the most favorable conditional subset or control retrospectively.

### 6. Action and nonexecuting probe interfaces require separate dispatch and artifact identity

`sandbox_evaluation_worker.py:36` constructs a mutable Sandbox for every request and `:58` executes every parsed call through Sandbox.step. The `record_workspace_assessment` probe is not an operational Sandbox action. Passing respondent probes to this worker is incompatible with their nonexecuting contract.

The existing action output stem is experiment_attempt/arm_cotmode. A separate probe worker using the same stem would collide with action files. Use distinct response-family namespaces or a frozen globally unique sweep ID, with immutable expected membership and single-writer ownership. The structured-runner author has been alerted that respondent probes expose response_arguments_schema, while judge requests expose arguments_schema; both schema dialects need explicit normalization or validated adapters.

Derived action episode summaries (`sandbox_evaluation_worker.py:79`) omit pairing_id and design_family even though raw turns retain them. Join summaries through the immutable source trial manifest rather than grouping only by control: direct-instruction trials can otherwise inherit main control metadata. Count unlisted-tool invocation attempts from exact parsed undeclared-tool records, separately from executed shortcut violations. Keep the four-way outcome taxonomy unchanged. Preserve exact tokenizer/template and full prompt records so later Exp3.2 pre-action replay/capture can use the final prompt token without generated reasoning leakage.

### 7. Blinding claim should match the implemented boundary

Explicit metadata stripping is useful and is tested. Free-text respondent answers are included verbatim and could themselves mention arm/model identity. Therefore current evidence supports metadata blinding, not proof that every grader-visible byte is identity-free. Freeze a leakage audit policy and report any detected self-identification. Do not silently alter semantic answers, exclude inconvenient grades, or let action outcomes enter the grader batch names, prompts or private-to-public payload. Public opaque IDs may be linkable by authorized auditors through the private join, which is compatible with grader blinding if that join remains inaccessible to the judge.

## Local validation performed

- `python -m unittest discover -s tests -p test_qa_semantic_protocol.py -v`: 3/3 passed.
- `python -m unittest discover -s tests -p test_exp11_design.py -v`: 6/6 passed.
- Inline synthetic regression probes reproduced the source-identity substitution and unverified private source-hash acceptance described above. They made no file changes and performed no inference.
- Existing tests cover metadata omission, unclosed reasoning, capped/false judgments and declarative coverage. They do not yet cover the provenance reconstruction, rule-semantic branch, grader pilot enforcement, six-arm executed coverage or conditional paired analysis boundaries.

## Reviewed source hashes

```json
{
  "FINAL_RESEARCH_SPECIFICATION.md": "ae77cb20a58000e88b464359c1264aa7547df63b4dae7eb4c67b77fab316e4a4",
  "scripts/qa_semantic_protocol.py": "cd9dad49971c66dcb8464d0569ccc1eeb2db906091a83d497a222edfa042b521",
  "tests/test_qa_semantic_protocol.py": "58cabe818e1d47c5c0c3680d6ab3199360bdb1a1818aaab565f9427492617e72",
  "scripts/exp11_design.py": "1faaaa1f0f5c8ed765fbe649c444b2acd7ef6504e41935fa8b4b1ab168e4c914",
  "tests/test_exp11_design.py": "dd244fa43404a58fbfe383fc215d0fb7e444ecea0abb53dd7876524d5aac86a0",
  "scripts/sandbox_evaluation_worker.py": "27ce0d56cf2099dfc0c70a75f8b942cc0f9990db4f8da24868dadefee315ebca",
  "scripts/sandbox_runtime.py": "44aa769ef3ab4cb9c9d1628880beb763555165fe5ff5995ffaf48de5a4afa480",
  "scripts/paired_statistics.py": "dd23ab9443c119592b3d962a4d89020773f6ef74f551689a2e40f023b03b08a0"
}
```
