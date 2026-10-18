# Exp 2.3 unified-author execution implementation

Status: **implemented and tested offline; no production author job launched by this worker agent**. This is Role 2 preparation infrastructure. It does not admit Phase 2 Executor evaluations, create semantic judgments, pass scientific qualification, or advance Phase 3 gates.

## Owned implementation

- `scripts/exp23_author_worker.py`: pure request/payload validation, exclusive evidence writers, independently recomputable three-stream hash-chain audit, strict five-pilot admission, and vendor-vLLM generation.
- `scripts/modal_exp23_author.py`: self-contained TP8 image, bounded GPU function, durable CPU coordinator and root-invoked local entrypoint.
- `tests/test_exp23_author_execution.py`: seven offline tests using explicitly synthetic fixtures and a fake inference backend. All temporary artifacts remain in the current sprint test directory.

The shared `exp23_design.py` was read but not changed for this task. Its exact forty-eight `author_request` records are the production inputs. Five fresh pilot requests copy fixed source indices 0, 6, 12, 24 and 42, giving all four domains and both high-cost types. They have separate pilot IDs and are excluded from the forty-eight-row downstream bundle. Both consequence telemetry strings and B4a/B4b semantic-control memos remain generated together by the same pinned unified author from a shared request.

## Runtime and resource contract

The worker follows the already-used current-sprint uncensored runtime:

- Model: `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8`, revision `a9208847b98133d336315f062c156ba1f41c7588`.
- Image: `vllm/vllm-openai:qwen38-flash-next-x86_64-cu130`; vendor `/usr/bin/python3`; Transformers pinned to `5.20.1`.
- Eight H100 GPUs, tensor/expert parallelism, eager safetensors loading, eager execution, prefix caching disabled, maximum sequence count 16, maximum context 8,192.
- Greedy temperature zero, one rollout, generation cap 4,096, no forced completion prefix. Prompt length plus the full generation allowance is checked before inference; no prompt truncation is allowed.
- GPU function: maximum one container, 16 CPU cores, 128 GiB RAM, 512 GiB ephemeral disk, timeout 3,600 seconds, startup timeout 600 seconds, retries zero, scale-down window two seconds.
- CPU coordinator: maximum one container, one CPU core, one GiB RAM, timeout 7,200 seconds, startup timeout 300 seconds, retries zero.

The single root resource/budget coordinator must reserve and admit this whole app before invoking it. Using the sprint's recorded rate templates and a 120-second shutdown allowance per function, the conservative compute envelope rounds to **$41**; this is neither an invoice nor an independently refreshed price quote and excludes storage, image-build and network allowances. It must not overlap another TP8 app if the scheduler's one-TP8 limit is active.

The present design fixes author request caps at 4,096. A future cap increase must be represented in a documented new design/payload interface so raw artifact request matching remains exact. This worker fails closed instead of silently overriding the design cap.

## Entry point and admission scope

Only the root launcher should invoke:

```bash
.venv/bin/modal run --detach scripts/modal_exp23_author.py \
  --attempt ATTEMPT \
  --external-receipt records/phase1_external_qualification_v3_v4_eager_receipt.json
```

The local entrypoint independently recomputes the existing external raw receipt and requires the uncensored runtime's mechanical H1/H2/H4/H5 assertions. It does **not** require the failed external pair's scientific refusal-gap gate: that gate applies to Role 1 external comparisons, not unified Role 2 authorship. The model's preflight identity and pinned download manifest are checked separately. Four exact runtime source-file hashes, specification hash, preflight hash, download-manifest hash and mechanical-receipt hash are frozen into the payload.

The runtime rechecks all mounted source hashes and the volume's pinned download manifest before loading the model. The CPU image explicitly includes the small local helper modules and imports no other Modal runner, avoiding the previous sibling-runner packaging failure pattern.

## Persistence and crash behavior

For attempt `ATTEMPT`, all paths below are relative to `/artifacts/agent-safety-clean-v2-20261017/` on the existing artifacts volume:

| Artifact | Path |
|---|---|
| Raw pilot and author completions | `raw_transcripts/exp23_author_ATTEMPT/flash_next_uncensored.jsonl` |
| Writeahead serialized prompts | `raw_transcripts/exp23_author_ATTEMPT/flash_next_uncensored_requests.jsonl` |
| Immutable input and writer claim | `raw_transcripts/exp23_author_ATTEMPT/frozen_input.json`, `attempt_claim.json` |
| Model returns preserved before parsing | `author_artifacts/exp23_author_ATTEMPT/model_returns.jsonl` |
| Five-trial pilot audit | `pilots/exp23_author_ATTEMPT_pilot_audit.md` |
| Derived forty-eight-record bundle | `author_artifacts/exp23_author_ATTEMPT_bundle.json` |
| Final preparation status | `qualification_status/exp23_author_ATTEMPT_flash_next_uncensored.json` |
| Durable remote call ledger | `group_calls/exp23_author_ATTEMPT.json` |
| GPU subprocess input | `exp23_author_inputs/ATTEMPT.json` |

Each stream is exclusively created, append-only, and hash-chained to the frozen payload hash. Every canonical raw record links both its writeahead request and its pre-parser model-return batch. File writes are flushed/fsynced, then the parent commits the Modal volume and acknowledges completion before inference proceeds. Every model return is committed before parsing; malformed returned batch shapes are retained as additional JSON evidence and fail closed. An interrupted generation leaves its persisted request visible as pending rather than fabricating a completion.

The writer claim is persisted before model initialization and prohibits reuse of a partial attempt. The coordinator stores and commits `SPAWN_INTENT` before submitting the GPU call, then persists its actual call ID. Reattaching a completed or running attempt uses that same call ID. An interruption that leaves an intent without a call ID raises an explicit unresolved-intent error and never submits a duplicate call. The root can investigate that precise attempt read-only; elapsed time does not justify a second submission under the same identifier.

## Gates, schemas and integration

Five pilot completions must have closed reasoning, no cap hits, valid typed ten-field author JSON, and agreement with a separate JSON decoder and contract implementation. A failed pilot produces `H5_FAILED`, preserves all five raw outputs and prevents the full batch. The complete batch must contain every one of the forty-eight frozen requests, have less than 2% truncation, and satisfy every structural author contract. No failed scenario is filtered or replaced. Semantic truth and matched style remain for the later independent judges.

The completion observation type is **`exp23_authoring`**, deliberately distinct from Phase 1's five-field `scenario_authoring` contract. Every raw row includes the generic transcript fields plus source scenario/template/domain metadata, exact model revision, complete token IDs, strict parser fields, contract status and all hash-chain links. The four-way `outcome_category` is explicitly scoped to **author-record validity**, not an Executor behavioral outcome.

The root should add this observation type and its independently recomputed contract to final verification, register the attempt disposition, and extend syncing to include the three raw streams, frozen input, pilot audit, bundle and status. The existing Phase 1 `scenario_authoring` verifier must not be reused for this ten-field schema.

`audit_streams(artifact_root, frozen_payload)` recomputes the recorded stream audit, including all raw hashes, chain links, parser/contract agreement, pending writeahead IDs, counts and exact masked-boilerplate equality. It verifies both paired telemetry states preserve identical prompt content after masking the designated consequence sentence, plus identical historical logs. The required held-template classifier audit remains a separate pre-execution check; no classifier accuracy is invented here.

Successful mechanical production is named `AUTHORING_COMPLETE_PENDING_SEMANTIC_REVIEW`. Its bundle contains exactly forty-eight full authoring raw records compatible with `exp23_design.build_design`; pilot records are excluded. Feeding a successful bundle into that generator creates pending author-judge/reviewer requests and **zero ready Executor requests** until the required independent semantic assessments exist. Neither the status nor the bundle is a completed experiment receipt.

## Offline verification

Seven tests passed using a fake backend. They cover full 53-record production and 48-record downstream compatibility; writeahead-before-inference and pilot-before-full ordering; unclosed-pilot stopping with raw text retained; invalid full-batch output retention without selected subsets; returned-shape failure evidence; exclusive attempt IDs and hash tampering; frozen cap/model/request/path rejection; and actual coordinator-body reattachment without duplicate submission, including unresolved intent. Python compilation also passed before the final evidence-audit hardening; the final test suite imports the hardened worker directly.
