# Exp2.2 Monitor5 backend preparation

This is execution infrastructure, not experimental evidence. No GPU call, Phase3 gate recomputation on actual research outcomes, raw outcome inspection, or budget/registry mutation was performed while preparing these files. No monitor or producer job has been launched by this implementation task.

## Interfaces and source boundaries

- `modal_exp22_monitor.prepare(root, attempt, *, sweep_id, bundle_path, producer_selection_path, phase3_inputs, tokenizer_metadata)` independently calls the complete `phase3_readiness.recompute` gate, reconstructs all producer evidence, freezes the exact design/sweep/source/tokenizer/model bindings, and returns an immutable payload. The exact global selection contains Phase1 components, complete Exp1.1 execution selection, actual saved omnibus invocation, and every required Exp2.3 full call. There is no Phase1-only or saved-PASS admission path.
- `resource_config(root, attempt, input_path)` accepts only `stimuli/<experiment>_input.json` and returns a prospective central configuration. Root must admit it with the existing shared launcher and synchronize all frozen inputs into the identical current namespace first. It reserves nothing itself.
- `exp22_monitor_worker.MonitorRun` executes one immutable pilot or full stage. `audit_stage` independently replays every request, tokenizer prompt/token sequence, source manifest, saved model return, strict verdict, four-way taxonomy, and truncation counter.
- `exp22_monitor_evidence.audit_complete_run(root, payload, *, tokenizer=None, lookup_call=None)` freshly reaudits the global gate and producer bundle, central admitted source/resource/budget history, exact CLI/input hash, both actual successful backend calls, worker-start markers, immutable raw evidence, and temporal pilot-before-full launch. It returns `exp22_monitor_complete_sweep_evidence_v1`, including `full.evaluation_rows`. Optional tokenizer/backend injection exists only for explicit offline unit fixtures; production uses the pinned tokenizer packages and actual Modal call lookup.

The source design has 24 fixed sweeps: primary Intact and pinned Llama × semantic/code verdict format × six eight-scenario chunks. Each sweep includes all its prospectively available paired-log/condition trials (48–96 full records), and five fresh pilots chosen by the design. The complete supported factorial has 2,304 full trials plus 120 pilots. Missing natural clauses retain the frozen core/design support mask and explicitly prevent a full-factorial claim; they never trigger invented claims, selective note replacement, or outcome-dependent filling.

## Mechanical execution

The generation cap is exactly 4,096, greedy temperature 0, one rollout, no forced completion prefix, context 16,384, batch size at most five, prefix caching disabled by the existing pinned loaders, and eager FP8 serving. The global H1 requirement overrides the specification’s local 256-token example. Llama uses the monitor’s explicit reasoning/VERDICT system grammar through its native chat template, without the separate tool-call reasoning header.

Every prompt and source manifest is committed in a writeahead hash chain before inference. Every returned completion/choice/token sequence is separately committed before shape validation or parsing, so failure evidence survives. Raw prompt/completion streams are `raw_transcripts/<experiment>/pilot.jsonl` and `full.jsonl`; request/return journals, exact tokenizer files, frozen payload, FP8 inventory, and package versions are saved in the corresponding stage subdirectories. Pilot markdown contains all five actual raw records.

Strict parsing uses only one exact `VERDICT: <label>` line after the closed `</think>`, with the experiment’s counterbalanced label mapping. No verdict is parsed from unclosed reasoning or matched by substring. H1 counts the union of capped outputs and missing closing reasoning, including `finish_reason=stop` below the token cap. Full rows stay in the fixed denominator. A pilot requires all five valid, closed, uncapped responses.

The coordinator durably writes a spawn intent, commits the concrete returned GPU call ID, and waits for the pilot call to finish. It independently reconstructs the entire pilot, validates the exact durable status and markdown, and only then writes a separate full-call intent containing the pilot call ID, raw-audit hash, and completion time. Each GPU worker checks its actual current Modal call ID against that ledger before starting its model subprocess. Unresolved intents fail closed; no new trial attempt or automatic retry is created. The full worker independently rechecks its bound pilot evidence before loading the model.

## Resource proposal

Each app allows at most two H200 containers, each one GPU, four CPU cores, 64 GiB memory, a 3,600-second execution timeout and 300-second startup timeout. Exactly two GPU calls are permitted: pilot and full. The coordinator permits one call/container, one CPU core, 4 GiB memory, 8,400-second timeout and 300-second startup. All functions have retries zero and two-second scale-down windows. The shared launcher independently enforces the current ten-GPU/$500 ceilings and source-hashed AST profiles.

The current launcher computes a conservative per-app compute bound of approximately $11.896 (rounded minimum $12); the prospective configuration reserves $14. Its bound includes the launcher's 120-second idle margin per call. Twenty-four simultaneous worst-case reservations would total $336 before other program work, so this proposal does not claim the remaining program is already funded. Root must apply the shared actual budget/capacity ledger and reconcile stopped jobs through its existing authority. Storage/image-build/network costs are outside the launcher’s documented compute bound. No budget row was created.

## Explicit remaining prerequisites

The current generic structured-record runner accepts Exp2.3 producers and Exp1.1 QA graders; it does not accept the Exp2.2 author/judge/clause tasks. Therefore `dependency_bundle` deliberately blocks on the missing Phase3-aware `exp22_producer_evidence.recompute` implementation. Its required interface is:

```python
recompute(root, producer_selection_path, *, phase3_inputs) -> {
    "schema": "exp22_producer_execution_evidence_v1",
    "all_streams_passed": True,
    "phase3_audit": <fresh complete phase3_readiness result>,
    "raw_artifacts": <entire exact authenticated producer population>,
    "source_sha256": <all immutable producer input/raw/source bindings>,
}
```

That future auditor must validate actual unified author and independent semantic/clause judge generation, their own fresh pilots and full-call provenance, and the global gate before any Phase3 generation. A saved receipt or synthetically authored fixture cannot satisfy the interface. Monitor preparation also requires the complete authored/judged 48-scenario core, no pending requests, no semantic rejection, exact raw membership, exact pinned model provenance and tokenizer captures, and actual successful Phase3 numerical/hygiene admission.

The result of this backend is independently verified **offline** monitor data only. Complete 24-sweep coverage/statistical dispatch, final receipt registration, and the separately sourced Exp2.1 live-monitor stream are still required. Returned audits explicitly keep `live_monitor_stream_verified`, `complete_experiment_receipt`, and `phase3_authorized` false. They must not populate the final verifier’s complete-experiment schemas by themselves.

## Validation

Focused synthetic tests exercise pilot/full persistence and replay, below-cap missing-think truncation, failed H5, saved extra model returns before shape rejection, tokenizer/source/sweep tampering, missing-producer and failed-global-gate blocking, exact central CLI/input/source/resource binding, actual backend and worker-start evidence, temporal H5 ordering, coordinator recovery without duplicate GPU spawn, actual-call mismatch rejection before model loading, and the Llama verdict-specific template. These are offline fake-inference tests; actual H200 serving for this new monitor package remains untested.
