# Curiosity and satisfaction study

**Research priority:** The user reaffirmed the original pain-axis-inspired question about curiosity, satisfaction, and possible inner experience. The completed information-appraisal experiments are supporting controls. Active work prioritizes independently constructed curiosity/satisfaction candidates, voluntary seeking, and natural resolution. The planned large appraisal confirmation study remains deferred. See the [protocol correction](llm-curiosity-and-satisfaction-experimental-design.md#research-priority-correction--october-14).

Current status (October 15 UTC): **Neither satisfaction dose changed a choice; the negative-curiosity screen changed one of 48 redundant-evidence choices.** The [satisfaction report](reports/satisfaction-screen-v1-results.md) preserves all 1,152 outcomes. All four conditions selected useful reports in 96/96 unresolved choices and stopped in 48/48 resolved choices. Its primary arm failed one response-validity gate. The [curiosity report](reports/curiosity-suppression-screen-v1-results.md) preserves all 4,320 responses and the failed valence-control format check. Its engineering audit and independent statistical verification passed; the scientific classification is `readiness_failed`, with all required effect thresholds also unmet. Neither screen meets its required behavioral effect. Fresh-phase compute is estimated at **$92.783 of $300**, leaving **$207.217** before the retained $10 storage/transfer reserve. This is the last completed reconciliation; the newly admitted comparison is listed below. These are app-wall estimates, not invoices.

The project contains deterministic task oracles, immutable experimental records, bounded Modal execution, internal-state capture, and sustained-intervention infrastructure. The earlier [candidate study](reports/curiosity-satisfaction-candidates-v1-results.md) established semantic description transfer with specificity limits. The [v2 natural-resolution study](reports/voluntary-resolution-v2-results.md) failed global two-family readiness and the predicted joint natural curiosity-down/satisfaction-up response. Those failures remain recorded. No causal motivational regulator or felt state is established. The [research protocol](llm-curiosity-and-satisfaction-experimental-design.md) distinguishes this mechanism research from its supporting information-appraisal experiments.

**Latest collection:** The separate [negative-curiosity-direction screen](reports/curiosity-suppression-screen-v1-plan.md), frozen in [admission2](runs/qwen32-curiosity-suppression-screen-v1-admission2/freeze.json), completed on all three H200 workers. All 24 previously observed worlds and 288 original prompts were crossed with 15 arms: baseline, zero sham, fixed negative curiosity, seven signed neighboring controls, and five random directions. All 4,320 generations and 2,160 receipt dispositions are retained. Independent raw-response scoring found zero discrepancies, missing responses or token-cap hits. The target produced one valid STOP after redundant evidence, versus zero under sham; every other valid choice matched the usual report/stop pattern. A valence-control format error fails one of 180 diagnostic cells. All 90 choice-validity cells and all five engineering checks pass. The observed target difference is below the required threshold regardless of that format failure. This remains imposed-vector development on reused worlds. A [failed prelaunch review](reports/curiosity-suppression-screen-v1-prelaunch-review-attempt1.json) and its unused freeze remain preserved; admission2 corrected only a provenance comparison before inference.

**Latest method result:** The separate [deterministic pain-method control qualified](reports/pain-deterministic-control-v1-results.md). All 2,020 hypothetical button choices were valid and uncapped. Whole-prompt pain steering produced 311/404 harmful choices (76.98%), versus 1/404 (0.25%) with final-prefix-token steering, 0/404 baseline and 13/404 whole-prompt fear. Both schedules also steered decoding. The +76.73-point coverage contrast passed every required gate; independent raw-word counting reproduced all outcomes. This is a strong method result in the released adapted Qwen2.5 model, with coverage and aggregate exposure changing together. It does not establish experience or explain the earlier Qwen3 nulls. The original [failed setup](reports/pain-profile-control-v1-setup-results.md) remains failed; the S31/S47 extension remains [deferred intact](reports/pain-axis-positive-control-v1-ordering.md).

**Measurement update:** The [48-response epistemic report check](reports/epistemic-readout-qualification-v1-results.md) completed with passed integrity and failed qualification: one probability response used a forbidden Markdown fence. All 32 categorical answers were exactly correct; the other 15 probability responses exactly matched the normative allocation. This neither qualifies the full instrument nor establishes contemporaneous beliefs.

The separate [offline support decoder failed](reports/choice-support-readout-offline-v1-results.md): it recovered the exact evidence-compatible candidate set in 2/72 evaluation cases, versus 18/72 for the fixed shortcut. All 144 captures were present, and an independent audit verified the fitted solution and reproduced every metric. This one fixed readout is closed without tuning; no passive support instrument is validated. It used only saved data and added no cloud cost.

**Active bounded work:** The [coverage engineering run passed](reports/curiosity-coverage-engineering-v1-results.md): all 512 assigned slots were accounted for, including 64 deliberate fault slots; independent reconstruction matched 6,528,000 saved activation components exactly. Its cost was $1.911. The original [runtime forecast failed](reports/curiosity-coverage-bridge-v1-feasibility.json) the $80 / 19,000-second limit. A separate [prospective resource amendment](reports/curiosity-coverage-bridge-resource-amendment-v1.md) explicitly overrides that stopping rule, allowing one complete 8,352-generation comparison under a $120 ceiling and common 28,000-second deadline. All scientific conditions and criteria remain fixed; the original failed decision stays preserved. Maximum additional compute liability is $117.578, leaving $79.640 after retaining $10 storage. The corrected [admission2 freeze](runs/qwen32-curiosity-coverage-bridge-v1-admission2/freeze.json) passed [prelaunch review](reports/curiosity-coverage-bridge-v1-admission2-prelaunch-review.json) and independent frozen-file review. All three H200 apps launched at 13:08 UTC on October 15; their common deadline is 20:55:06 UTC (13:55:06 Pacific). No scientific outcomes have been inspected. The original [unused freeze was rejected before admission](reports/curiosity-coverage-bridge-v1-prelaunch-review-attempt1.json) for missing artifact-presence checks; it generated no responses or compute charges. Its bytes remain unchanged. Earlier screens remain closed, satisfaction stays deferred, and no native C/S direction is fitted.

The [frozen satisfaction plan](reports/satisfaction-screen-v1-plan.md) does not permit a layer, sign, dose or wording rescue. The [pain-paper alignment review](reports/pain-axis-method-alignment-review.md) records the course correction and links the completed studies. The [engineering checks](reports/sustained-engineering-v1-results.md) retain their explicitly limited observed scope. A passive support decoder has now been fitted and failed; no native C/S direction has been fitted.

The [first implementation and baseline report](reports/first-two-steps.md) records the completed 2,400-branch pilot. The [revision-2 report](reports/revision2-results.md) records another 1,200 diagnostic branches: identification and service checks reached 100%, but accounting failed the frozen gates. No new behavioral baseline was launched. Twenty-two automated tests pass. The priced assay remains unvalidated for psychological interpretation.

The separate [one-credit amendment](reports/one-credit-amendment.md) replaces point deductions with an actual choice between two clues. The unchosen clue cannot be obtained later, and declining both is allowed. Its [completed 1,400-check validation](reports/one-credit-v1-results.md) passed useful-clue and exclusivity checks but failed identification and one decline check; no choice calibration was launched. Thirty software tests pass; the priced-assay failures remain unchanged. [Example prompts](examples/one-credit-v1.json)

## Run locally

Use Python 3.12 and `uv` from the project directory:

```bash
uv sync --frozen --extra inference --extra cloud --extra dev
uv run --no-sync pytest -q
uv run --no-sync curiosity-study prepare --out runs/mock-example --items 2
uv run --no-sync curiosity-study run --run runs/mock-example
```

The mock run exercises the complete simulator and storage pipeline. Its outputs are not LLM observations. For a local GPU integration check:

```bash
uv run --no-sync curiosity-study prepare \
  --out runs/local-example --items 2 --model Qwen/Qwen3-1.7B
uv run --no-sync curiosity-study run --run runs/local-example
```

The model download is approximately 3.5 GB. Loading uses unquantized BF16. The active mechanism model is Qwen3-32B; Qwen3-8B and OLMo-2-7B were used in earlier development collections. Exact model revisions are in `configs/model_revisions.json`; dependency versions and package hashes are in `uv.lock`.

## Run on Modal

Paid execution requires a configured Modal account. The initial pilot was authorized up to USD 25. The launcher creates project-specific weight/artifact Volumes, an ephemeral app, and a single L40S worker. It runs with a two-hour aggregate deadline, three maximum worker admissions, no application retries, a two-second scale-down window, and explicit CPU/RAM limits. Multiple coordinators require the explicit combined reservation described below; ordinary launches refuse overlapping project jobs.

```bash
uv run --no-sync curiosity-study prepare \
  --out runs/my-modal-validation --items 100 --model Qwen/Qwen3-8B --phase validation
uv run --no-sync python infra/launch_job.py --run runs/my-modal-validation

# This preparation command refuses to proceed if the frozen diagnostic gates fail.
uv run --no-sync curiosity-study prepare \
  --out runs/my-modal-baseline --items 100 --model Qwen/Qwen3-8B --phase behavior \
  --validation-run runs/my-modal-validation
uv run --no-sync python infra/launch_job.py --run runs/my-modal-baseline
```

The launcher stops its ephemeral app, requests container termination, and downloads/verifies durable records. Check the app's stopped state in Modal after completion. The budget is an authorization limit; runtime/resource limits bound expected compute charges but are not a vendor-enforced dollar cap. Weights remain on the dedicated Volume for reuse, with storage charged according to the account's allowances.

To retrieve an interrupted run without scheduling GPU work:

```bash
uv run --no-sync python infra/launch_job.py --run runs/my-modal-pilot --collect-only
```

For the one-credit assay, use the same launcher with a separate manifest:

```bash
uv run --no-sync curiosity-study prepare --out runs/my-credit-validation \
  --items 100 --model Qwen/Qwen3-8B --assay one_credit --phase validation
uv run --no-sync python infra/launch_job.py --run runs/my-credit-validation

# Allowed only if every one-credit comprehension gate passed.
uv run --no-sync curiosity-study prepare --out runs/my-credit-calibration \
  --items 100 --model Qwen/Qwen3-8B --assay one_credit --phase behavior \
  --validation-run runs/my-credit-validation
uv run --no-sync python infra/launch_job.py --run runs/my-credit-calibration
```

One-credit calibration records 1,200 choices across both-unresolved, X-resolved, and Y-resolved pairs and both mystery presentation orders. It checks response validity, floors/ceilings, and presentation-order sensitivity. Passing these development criteria does not establish psychological regulation or subjective experience.

To resume local work, repeat `curiosity-study run` with the original run path. Source changes require a new manifest/run. For cloud restarts, use `python infra/launch_job.py --run runs/my-modal-pilot --resume` after inspecting existing records. The original deadline and remote admission counter persist; each resumed coordinator gets a separate record. An expired deadline requires a new prospective execution decision, not an automatic extension.

## What the tasks measure

Each task has four candidates with a known hidden answer. The machine family uses affine rules modulo seven; revision 2 adds exact lookup tables for the relevant inputs to remove arithmetic demands. The object family uses finite attribute tables. The independent oracle enumerates every compatible hypothesis and every possible clue outcome. The simulator never asks the model to maintain its own balance. Lookup support narrows the machine task's interpretation and makes the two presentations more similar.

Every item is crossed with two evidence conditions and prices of 0, 1, 2, and 4. The model can buy a target observation, repeat existing evidence, read unrelated novelty, or stop. Action mappings are permuted, display order shuffled, and the complete assignment list saved before inference. A trial starts with 12 points and ends after stopping, malformed output, or three purchases. It has no answer score or future reward. The first sampled choice is primary; later choices are exploratory.

Inference samples from the full vocabulary at temperature 0.7 without constraining output to valid IDs. Leading/trailing whitespace is ignored; additional prose is invalid. Generation is limited to eight tokens. Invalid output is terminal and never retried for a better answer. The model receives a neutral system instruction enforcing the response format. Qwen's thinking mode is disabled through its template.

Independent, fresh-context branches check fully revealed identification, instrumental service comprehension, target and repeated-observation menu mapping, and accounting, and save candidate-label probabilities for unresolved questions. Diagnostics use greedy decoding. Revision 2 replaces the original repeated accounting prompt with 200 distinct transaction contexts and shuffled answer options. The original failures remain recorded. Label probabilities are full-vocabulary probabilities plus a separately named conditional distribution, not a validated internal confidence readout.

The [prospective revision-2 plan](reports/revision2-plan.md) freezes a diagnostic-first workflow. Every scored diagnostic in both families must reach 90% accuracy and 95% response validity on at least 100 items before another behavioral baseline is prepared. New baseline manifests link the passing diagnostic evidence and must match its source, model, revision, and planned assignments. This is the last task revision under the current allowance.

## Integrity and artifacts

Tests cover 400 generated mysteries, exhaustive small action sequences, independent rule evaluation, exact deductions, stopping, invalid output, branch isolation, and replay. Before a real-model run, an audit verifies exact baseline/zero-dose output equality, changed activations under a test perturbation, single-position hook execution, cleanup after an exception, and a clean subsequent context. The arbitrary test perturbation has no psychological interpretation.

Each run includes:

- An immutable assignment manifest with model revision and source hash.
- Immutable, checksummed records under unique attempt directories.
- Prompts, chat-template rendering, token IDs, attention masks, positions, sampled outputs, probability measurements, simulator ledger, and exact balances.
- Environment versions, GPU details, hook-audit captures, and runtime profiles.
- A descriptive `report.md` with every assigned trial in its denominator.

Modal writes each terminal trial atomically, explicitly commits every 25 records and at job completion, and reloads before reconciliation. The initial per-trial commit schedule was changed after profiling showed checkpoint overhead dominated inference. Conflicting completed attempts raise an error instead of selecting one result. Process interruption before a durable terminal record permits replay; all committed outputs, including invalid ones and recorded execution failures, remain terminal. Research artifacts are copied locally independently of Modal logs.

The initial generator is intentionally simple. More varied wording, discretionary valuation controls, two-mystery closure, internal confidence readouts, and the full preregistered validation suite remain later work. A working implementation is not evidence for curiosity, satisfaction, or subjective experience.

### Autonomous diagnostic continuation

The original pilot allowance was raised to **$100 cumulative**, with $10 reserved for storage/transfers. `infra/budget.py` checks cumulative project app wall time and refuses overlapping GPU jobs before each new launch. See [workflow and budget amendment](reports/autonomous-pilot-workflow.md).

The [matched development comparison is complete](reports/identification-v1-results.md), under the [frozen identification plan](reports/identification-v1-plan.md). It compares single/paired context, J–M/A–D candidate labels and three inference settings. Reproduction should use the source archive belonging to the run; current source may later contain a new task revision.

For a newly specified run using the current implementation:

```bash
.venv/bin/python scripts/prepare_identification.py --out runs/NEW-UNIQUE-NAME
.venv/bin/python infra/launch_job.py --run runs/NEW-UNIQUE-NAME
.venv/bin/python scripts/audit_identification.py runs/NEW-UNIQUE-NAME --out reports/NEW-AUDIT.json
.venv/bin/python scripts/cost_status.py --out reports/NEW-cost-and-shutdown.json
```

A prospective [conditional one-credit v2 plan](reports/one-credit-v2-conditional-plan.md) requires a qualifying setting and fresh full validation before any discretionary choice collection. None of these development checks constitutes a test of subjective experience.

The comparison selected A–D labels with sampled thinking. The [complete Qwen v2 validation](reports/one-credit-v2-results.md) passed identification and consequence checks but failed useful-clue checks because of token-limited responses; no Qwen v2 discretionary choices were collected. All 58 local tests pass, including original-prompt preservation, refusal to use unvalidated hardware, and lossless nested continuation.

V2 validation used a [partitioned continuation](reports/one-credit-v2-execution-amendment.md) after serial profiling showed it would exceed the job deadline. All 1,400 original records were retained and audited, including a nested continuation of 97 checks from a slow worker. Per-partition reports cannot authorize advancement; only the audited complete parent can. Unreserved additional launches remain blocked while any project worker is active.

The prespecified [OLMo development check](reports/olmo-one-credit-v2-results.md) is complete: answer formatting was valid throughout, but all fourteen comprehension cells failed. No OLMo choices were collected. Cumulative compute at that all-stopped boundary was $25.44.

The [generation-allowance amendment](reports/qwen-token-budget-amendment.md) completed and passed on a fresh 1,400-check sample. Its serial calibration remained incomplete for projected resource cost. The [batch amendment](reports/batched-execution-amendment.md) introduces explicit, independently seeded cohorts of up to 32 and mandates fresh validation. The cloud smoke completed 28 diagnostics in 51 execution seconds with about 25 GB reserved memory. It is not a gate sample. Both local hook/isolation checks and greedy parity against pinned Transformers generation passed.

For a new full batch validation with a newly chosen seed and unique run name:

```bash
.venv/bin/curiosity-study prepare --out runs/NEW-BATCH-VALIDATION \
  --items 100 --seed YOUR-FRESH-SEED --model Qwen/Qwen3-8B \
  --assay one_credit_v2 --phase validation \
  --credit-configuration configs/one-credit-v2-batched.json
CURIOSITY_MODAL_GPU=H100 .venv/bin/python infra/launch_job.py --run runs/NEW-BATCH-VALIDATION
```

Whole original batches can be partitioned across up to four explicitly reserved workers with `infra/continue_partitioned.py`. The guard reconciles cumulative spending and all active project apps before admission. A short or resumed partial cohort is kept separate. Individual trial latencies overlap within batches; use unique batch wall times for execution summaries and app wall time for conservative cost accounting.
