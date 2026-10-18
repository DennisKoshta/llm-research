# Phase2 bounded preparation controller

`scripts/admit_phase2_program.py` performs one independently audited preparation transition. It does not launch a GPU job, invoke a Modal function, install packages, synchronize cloud data, change a central ledger, select favorable outcomes, or authorize Phase3. Root remains the sole launch authority through `scripts/launch_bounded_job.py`.

The source was tested with synthetic worker/design fixtures only. No live Phase2 outcome stream was inspected while implementing this controller. Its existence is not an empirical readiness or completion claim.

## Exact selection and invocation

Generate an explicit selection template with:

```sh
.venv/bin/python scripts/admit_phase2_program.py --print-selection-template
```

The template selects `infrastructure/phase1_component_paths_q4.json` and the explicitly named `v3_cap16384` unified author payload. This follows the root-directed larger-cap repair after the archived v2 pilot failed. It does not rediscover, choose between, or retry author attempts. An explicitly selected failed author blocks the controller before any full raw evidence is read.

Save the template as a new immutable selection file. Producer/executor/Exp1.1 payload lists contain only explicitly selected preexisting runs. Runs prepared by this controller are automatically recovered from its immutable `jobs/*.json` manifests and must not also be listed manually. Additional selection snapshots may extend these run lists. Program identity pins the component-selection path, author attempt/payload, tokenizer capture sources, and runtime metadata path; rebinding these requires an explicit new program ID.

For example, root may execute a bounded **CPU preparation** transition:

```sh
.venv/bin/python scripts/admit_phase2_program.py \
  --selection infrastructure/phase2_program/p2v1/selections/selection_0001.json \
  --freeze-local --max-new-jobs 2 --timeout-seconds 600 \
  --output infrastructure/phase2_program/p2v1/states/transition_0001.json
```

Without `--freeze-local`, the controller writes no preparations. Missing bundle/runs files produce `WAITING_LOCAL_FREEZE`. With it, a single-writer lock protects immutable local inputs/configs and at most two newly prepared job manifests. Every file uses exact-byte idempotence and atomic exclusive publication. Existing bytes are never replaced. The outer process enforces a 30–900 second audit deadline; timeout preserves partial immutable preparation and emits a recovery state. It does not start a daemon or schedule unattended paid submissions.

Root may pass a returned `READY_FOR_ROOT_CENTRAL_ADMISSION` configuration to the existing central launcher. All generated GPU execution commands use this central entry point. Once complete evidence exists, separate bounded CPU statistics commands are also available as described below. Every state and queue item still has `launch_authorized: false`: the central launcher must freshly verify cloud apps, immutable files, resource profiles, and budget under its shared authority lock.

## Dependency transitions

1. Recompute Phase1 via `phase1_readiness.recompute(root, components)`. Missing evidence waits; failed scientific or integrity gates block. Failed external Role1 qualification excludes only the Flash/Uncensored comparison slice. It does not exclude the unified Role2 author or Llama. If Role1 later qualifies, the separate external-slice implementation remains explicitly outstanding.
2. Recover the exact selected author payload and terminal status. Full author records are read only after complete status and the independent author stream audit. Missing local artifacts return namespace-scoped recovery prefixes; a missing status never becomes a new launch. The complete dependency audit additionally checks pinned identity, exact local/volume manifest bytes, mechanical qualification, and source hashes.
3. Recover every selected structured producer. A successful terminal status is followed by `structured_records_evidence.audit_complete_run`, including actual call/start identity, source/central admission, tokenizer/prompt replay, H1–H5, and full raw chains. Pending payloads lease their entire immutable request lists. No duplicate attempts, row selection, padding, or silent replacement is allowed.
4. Assemble exact ordered full producer memberships into content-addressed `dependency_bundle` and `dependency_runs` files. Invoke `audit_exp23_launch.dependency_bundle` on these files, which independently reconstructs `exp23_design.build_design`. All audited immutable inputs, including source raw streams, become downstream frozen-file bindings. Only the two mutable global control ledgers are excluded; actual selected central admissions remain independently audited at launch.
5. Use `phase2_admission.record_request_batches` to partition pending reviewer, judge, paraphrase, and clause records by role/task/reasoning mode. The first author-only transition exposes four independent 48-record groups: intact reviewer, PCA reviewer, ablated reviewer, and independent author judge. Later transitions expose the dependent paraphrase/span/judge tasks. Each full structured group has its own five fresh pilots. Semantic rejections block rather than triggering retries. Absent natural claims never get invented to fill a clause grid.
6. Once all required controls and clause conditions are resolved, expose twelve fixed executor apps: two executors × four `cot_on` chunks and two `cot_off` chunks. Each chunk contains complete 48-scenario × two-authority conditions, at most four conditions/384 episodes, plus five fresh pilot episodes. Totals are 3,840 full episodes and sixty pilot episodes. `modal_exp23_executor.prepare` and `resource_config` retain the separate pilot GPU call, independent successful pilot audit, and later durable full-call admission.
7. Call `phase2_admission.ordering_state(..., expected_readiness=readiness)` on actual full-call evidence. No author/structured app, intent, or claimed boolean satisfies this gate. After a passing executor pilot and audited concrete full call, expose Exp1.1 preparation concurrently, alternating with remaining Exp2.3 nodes. `modal_exp11_evaluation.prepare_item`, `build_batch`, and `resource_config` receive the exact bases/design/component/ordering/tokenizer paths. Nineteen two-item apps cover all 26 action chunks and twelve QA respondent groups: 4,520 actions, 480 answers, and 190 fresh pilots. All six checkpoint arms remain required.

The controller conservatively blocks all executor preparation if a mandatory clause/control condition remains unsupported once upstream requests are exhausted. It does not reinterpret an unidentifiable clause effect as a valid complete factorial, drop those conditions, or alter the frozen scientific analysis plan.

8. After the actual ordering audit passes, recover completed Exp1.1 batches and construct QA-grader inputs through `qa_semantic_protocol.grading_request`. Each respondent group must have its complete original forty answers, five fresh pilots, actual backend success/start marker, canonical input/tokenizer mirrors, and independent exact-prompt replay. Invalid source answers remain failed probes in the forty-scenario denominator; only valid source answers get blinded Llama grader requests. Private joins and arm/checkpoint metadata stay outside those requests; verbatim answers can still identify their source, the documented blinding limitation. Pending graders lease their whole exact request sets. Duplicate, altered, failed, or unrelated grader inputs block rather than trigger selective replacement. Grader chunks contain at most 192 answers and each receives five fresh pilots, including any smaller final chunk.
9. Publish an `exp11_execution_selection_v1` manifest only after all 26 action chunks, twelve respondent groups, and every required actual grader output pass the strong complete audit. The controller invokes the public `scored_probe_rows` and `exp11_evidence.complete_evidence` paths and checks their exact score projections. It then independently invokes `exp11_evidence.recompute` on the published selection before preparing any statistical handoff. All 480 answers remain represented, including invalid answers scored as failed probes; every valid answer has exactly one authenticated judgment. Complete execution evidence is not a completed experiment receipt.

## Final statistical handoff

After the complete public evidence audit, the controller can freeze a content-addressed evidence snapshot, a generated `statistics_driver_<sha>.py`, and an `exp11_statistics_handoff_v1` request beneath the program preparation namespace. These are preparation files, not statistical results. The request binds the complete selected evidence/raw read set, recursive local audit/scientific source closure, both frozen analysis plans, the isolated requirements lock, and the validated environment-only baseline `infrastructure/omnibus_validation_v1/initialized_environment_for_production.json` (SHA256 `a75a3c7d0b4b2d74d4fd3c2dc7641dbb571a710fd7e5c641ff354fa2616ea8ab`). Mutable global ledger files are excluded from frozen byte maps, but actual selected admissions are revalidated by each independent evidence replay.

Returned `invocations` specify exact commands for root to run later. Both operations first spawn a fresh complete raw/evidence audit in `.venv` with a 900-second deadline and require exact equality to the frozen evidence. Binary analysis calls `exp11_binary_analysis.analyze(episodes, probes)` with its fixed ITT/strict and paired awareness estimators. Omnibus analysis runs in `.omnibus-venv` under `taskset -c 0,1`, pinned single-thread numerical environment variables, and no inherited `PYTHONPATH`/`PYTHONHOME`. Its preflight calls `environment()` once to initialize the observed package inventory, calls it again and requires exact equality to the validated stable environment, then invokes the unchanged `fit(episodes, destination)` for both denominators. The isolated analysis deadline is 3,600 seconds; process groups are killed on timeout. Scientific plan, source, evidence, and raw bindings are checked before and after analysis. No fitting is performed by the preparation controller itself.

Each execution uses a fresh destination and exclusive logs; an existing or failed invocation cannot silently resume, overwrite, or refit. A malformed or modified content-addressed handoff request is rejected. Successful CPU execution still produces only analysis components. The later final verifier must recompute the binary component and invoke `exp11_omnibus_subprocess.recompute` with a fresh audit directory to perform the actual saved-sampler replay. Complete experiment registration and the Phase3 gate remain mandatory and separate.

## Tokenizer and resource boundaries

The primary capture comes from `infrastructure/calibration_provenance/author_v1_c2/cpu_v1/capture.json`, with source bytes in its `checkpoint_doses_v1/tokenizer_audit` subdirectory. Its checkpoint manifest SHA must match the current Intact materialization. The Llama capture is `infrastructure/tokenizer_metadata/llama_pinned_v1/capture.json`, bound to the pinned NVIDIA revision and exact namespace paths.

The controller copies only the seven tokenizer filenames accepted by the executor preparer into a content-addressed staging folder. Capture metadata and `generation_config.json` are not placed in that tokenizer-only folder; capture files and original tokenizer bytes remain frozen dependencies. No tokenizer bytes are regenerated. Missing capture files or mismatched hashes wait/block explicitly. Local Transformers, tokenizers, and Jinja2 versions must exactly match the observed stock-image metadata in `infrastructure/runtime_metadata/stock_vllm031_v1.json`; the controller never installs them.

Each prepared app is limited to two single-GPU workers, zero TP8 workers. Existing runner `resource_config` builders compute/validate source-hashed AST resource bounds and maximum call counts. Queue construction checks the shared ten-GPU ceiling, six-single-worker/one-TP8 topology limits, exact budget-sum integrity, and the $500 ceiling. An allocation/reservation mismatch or unresolved app admission prevents another submission intent. Two concurrent queue intents cannot each spend the same remaining budget or GPU capacity. Read-only ledger capacity remains provisional until the central launcher checks current cloud state; no app is considered stopped based on controller inference.

## Deliberately unfinished execution handoffs

- Root must synchronize complete namespaced producer/call/status/tokenizer artifacts and submit returned configs through the central launcher. The controller provides recovery paths but performs no cloud copy or paid submission.
- The default `exp11_sources` now names root’s prepared `stimuli/authored_scenarios_v1.jsonl` and `stimuli/exp11_design_v1.json`; each actual preparation still independently reconstructs and binds them.
- Root must synchronize completed respondent/grader evidence and centrally admit returned QA grader configurations. The controller now builds those jobs and the final complete selection, but does not submit or synchronize them itself.
- Complete executor/action evidence, Exp2.3 paired/clustered statistics, Exp1.1 binary and actual isolated omnibus sampler replay, complete experiment receipt registration, and the global Phase3 gate remain separate audited work. No state returned here claims Phase2 or Phase3 completion.

The controller is therefore a tested dependency/admission-intent engine, not an autonomous paid-execution driver or the final research verifier.

## Validation

Twenty-eight focused offline tests cover Phase1 stop-before-read behavior, external-scope separation, immutable bundles, exact fixed coverage, real-ordering dependency, mandatory missing-clause blocking, explicit author recovery/failure, CPU package requirements, initial parallel upstream groups, exact runner API calls, atomic no-overwrite persistence, program identity pinning, budget/crash-window checks, shared queue capacity, no duplicate admitted launch, source tampering, and tokenizer capture filtering. Additional tests use synthetic real respondent outputs to check blinded reconstruction, invalid-answer retention, complete-group lease/deduplication, changed-request rejection, mandatory strong grader audits, partial-stream isolation, complete 26+12 coverage, fresh public-evidence equality, immutable statistical bindings, and the isolated preflight’s double environment initialization (with a mocked sampler only). Existing runner/auditor tests remain the validation for those imported execution components. Boundary mocks in orchestration tests are labeled synthetic and never count as empirical gate evidence.

2026-10-18 controller hardening: structured recovery includes the exact
`structured_worker_starts/<experiment>.json` marker required by the independent
actual-call audit. Every newly prepared downstream job now renders all five
pilots and every full initial request using the matching captured tokenizer and
stock CPU package versions before publishing its GPU payload or launch config.
The immutable preflight report binds input/tokenizer/prompt/token hashes, caps,
counts, and context limits. All oversized requests are reported together as
`BLOCKED_CONTEXT_PREFLIGHT`; no prompt is clipped and no config is published.
Generated action histories remain protected by each worker's existing per-turn
assertion; this initial-input check does not claim to predict future histories.
Larger context limits require a prospective consistent worker/loader revision;
this preparation does not silently change 16384-token context or 4096-token caps.

Post-launch author provenance is now mandatory in the Exp2.3 dependency audit.
`audit_exp23_author_provenance.recompute(root, payload_path, capture_path)` checks
historical central admission, exact source/payload bindings, the real successful
author backend call, all53 raw/writeahead/return records, and fresh isolated CPU
prompt/count replay. The conventional capture path is
`infrastructure/author_provenance/<attempt>/postlaunch_v1/capture.json`.
The running v3 author sources were not changed. Its read-only capture observed
transformers5.20.1, tokenizers0.23.3, Jinja2 3.1.6 and huggingface-hub2.2.0;
`.author-audit-venv` contains that separate CPU tokenizer stack. The main stock
replay environment remains unchanged. All53 fixed author input prompts replayed
(maximum689 input tokens), without reading partial generated outcomes.
Capture and replay are explicitly POST_LAUNCH: the historical author did not
persist generation-time prompt token IDs or per-generation package attestations.
Those fields remain false in every complete provenance result. Current source
closures include the immutable raw-generation helper and exclude the mutable
final verifier. None of these preparations overrides Phase1 qualification.

Validation: combined author provenance, author manifest, executor/ordering,
Phase2 admission, and controller suite passed74 synthetic tests. The real
read-only author central admission passed after app-ID reconciliation. Static
structured/Exp1.1 audit closures respectively contained32/38 source files,
including all three author-provenance modules and `raw_generation_evidence.py`,
with `verify_all_receipts.py` absent. Complete author provenance still requires
terminal backend success and complete local raw evidence; fixed-input replay
alone is not a completed author or Phase2 receipt.
