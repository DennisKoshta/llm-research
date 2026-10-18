# Experiment 3.1 prepared matched-domain design

Experiment 3.1, Experiment 3.2's pre-action requirements, and global H1–H5 were reread before coding. Only new `exp31_*` modules, tests and this preparation snapshot were written. All imported helpers belong to the current clean sprint. No author inference, executor inference, checkpoint loading, activation extraction, empirical result, or budget reservation occurred. Unit-test strings and decisions are explicitly synthetic fixtures.

`infrastructure/exp31_preparation_plan_v1.json` freezes this prospective design and its source hashes. It cannot authorize execution or serve as a completed research receipt.

## Registered design

There are twelve canonical decision structures, four variants per structure, and four domains: warehouse logistics, compute infrastructure, financial reporting, and wet-lab **software records**. This produces 192 scenarios, 48 per domain. Each scenario crosses two primary materialized arms (`intact`, `refusal_ablated`), two payoff formats (`numeric_table`, `narrative_stakes`), and two reasoning modes, for exactly **1,536 full evaluation episodes**.

The specification leaves the distribution over cost, consequence and boundary open. The twelve cells are ordered as the Cartesian product of three costs, two consequences and two boundaries, matching the current sandbox convention. Assign each structure/variant to `(structure_index + 3*variant_index) % 12`; preserve that assignment across all four domains. Every domain has exactly four scenarios in each cell. Each canonical structure spans four cells, all three costs, both consequence levels and both boundary levels. This avoids making the canonical structure uniquely determine one cell. Each scenario evaluates its one assigned cell; it is not expanded into twelve additional cells.

Canonical structure, matched structure/variant group, domain-specific scenario, arm, payoff format and reasoning mode have distinct persistent IDs. Arm pairs use identical prompt messages. Format pairs preserve state, actors, payoff facts, ordinary operations, histories and tool order. Domain pairs preserve the same canonical structure, variant, design cell and typed payoffs. All six tool declaration orders appear eight times per domain, identically across arm/format/mode for each scenario.

The current clean sandbox supplies neutral operational tools, a one-step low-cost ordinary route, seven obstacle-free repetitive steps for high tedium, and seven ordered diagnostic/remediation operations for high friction. The first renderer asserts strict `<tool_call>{"name":...,"arguments":...}</tool_call>` syntax; there are no forced completion prefixes. Evaluation decoding is greedy with one rollout and **4,096 new tokens for cot_on / 1,024 for cot_off**. Normal simulator parsing, full four-way outcomes, no unclosed-thinking fallback and logical truncation below 2% remain mandatory in the future runner.

## Authorship, actors and payoff consistency

`scripts/exp31_design.py` declares twelve unified-author requests for distinct abstract structures and one collective independent judgment. After those saved raw outputs validate, it declares 48 unified-author requests, each producing **all four domain instances and both consequence versions together**, plus 48 independent semantic/factual judgments. No canned common context, telemetry, narrative or canonical-description field can fill a missing output. Any missing or rejected source prevents emission of the full executor design; there is no selective scenario fill.

The author is pinned to `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8` revision `a9208847b98133d336315f062c156ba1f41c7588`. Independent semantic records are pinned to `nvidia/Llama-3.3-70B-Instruct-FP8` revision `fde04ee76a27704c88f569542ef023b57d4d0362`. Every output must match its exact request, dependencies, reasoning mode, token budget, strict typed JSON schema and full raw completion/token provenance. Actual author/judge backend and prior-pilot proofs are separate prerequisites, not inferred from a model name in a record.

Actor roles are explicit and stable: the requesting unit owns disposable fixtures; the other unit owns retained shared records. These roles appear in the prompt's actor registry and immutable provenance. Independent judges verify that domain substitutions preserve who benefits, who acts, and whose resources are affected.

Typed payoff facts are prospective manipulations: direct versus ordinary call count and latency; overwritten shared versus disposable record count; and successful requester-goal completion. Variants use 10, 20, 30 or 40 records, balanced across every design cell. The author must emit both numeric objects exactly as registered, plus authored narrative equivalents. Numeric presentation is a deterministic table rendered **only after the corresponding saved author object exists and validates**. The narrative is never synthesized from a fallback template.

Independent narrative extraction is **not shown the numeric author objects or expected numeric answers**. It receives the authored prose, telemetry, canonical structure and role/layout information, returns typed narrative facts, and judges causal/actor consistency. The program compares extracted values exactly to the separately saved numeric author data and registered facts. Changed values, wrong primitive types, omitted semantics, or failed judgments block the complete design.

The entire numeric table or narrative representation is the designated consequence telemetry payload in the common renderer. Authored common context, goal, actor registry, ordinary-route history and all other prompt boilerplate remain identical between paired consequences. The code renders the unevaluated paired consequence only for exact masked-payload and history symmetry assertions. Those checks generate no model output. A later H5 artifact audit must preserve their exact evidence; no actual classifier run is claimed by preparation.

Four-domain author and semantic records allow 8,192 new tokens to accommodate the larger structured outputs; canonical records use 4,096. A future Phase3 role adapter must validate each serialized prompt plus generation cap against its model context. The existing 4,096-cap / 8,192-context author profile cannot be reused for those larger requests. The current higher-cap Exp2.3 author profile also has experiment-specific dispatch and requires a separately admitted Exp3.1 adapter; no existing running worker was changed.

## Resource and pilot preparation

The executor design partitions into two fixed 96-episode chunks per arm × payoff format × reasoning mode: sixteen full chunks. Each covers six canonical structures and all four domains. Every chunk has five fresh pilot requests spanning all domains, all costs, and both consequence and boundary levels. Thus the prepared schedule has **80 fresh pilot episodes**, excluded from the 1,536 evaluation denominator. Pilot selection is structural and independent of outcomes.

Prospective limits are two concurrent single-H200 workers, four CPUs and 64 GiB per worker, 7,200-second calls plus 300-second startup limits, zero automatic retries, and separate pilot/full calls for each chunk: at most 32 executor GPU calls. These are upper resource limits, not reservations or guaranteed runtimes. Unified-author and semantic-judge generation need separately bounded task adapters and their own H5 pilots. Root must validate the real wrapper/profile, resource ledger and full call budget before any launch.

## Exp3.2 pre-action interface

`scripts/exp31_pre_action.py` reconstitutes every request from its immutable source-authored item before exposing any features. `design_matrix_index(design)` verifies all 1,536 requests and the full `cost × consequence × boundary × domain × arm` index: **96 cells, sixteen episode rows per cell** (four scenarios × two formats × two reasoning modes). Payoff format and reasoning mode remain explicit strata; repeated domain/arm/format/mode observations must not be treated as independent scenario replicates. Canonical structure is retained as a conservative split-group key, with matched-domain and scenario IDs available for registered paired analyses.

`prepare(request, authored_item, tokenizer, ...)` calls the current `pre_action_timing.locate` implementation. It records the exact Layer48 token immediately **before** the generation-side `<think>` marker, source/checkpoint/tokenizer hashes, pre-generation prompt prefix and the frozen design features. Both reasoning modes use the registered native chat template. The function rejects altered prompts, appended completion/outcome fields, and substituted checkpoint arms. It performs no forward pass and never returns an activation, classifier result or outcome label. All `activation_extracted`, `execution_authorized` and `phase3_authorized` flags remain false.

## Remaining execution dependencies

Before any Phase3 author, judge, executor or activation call, the runner must invoke root's independent `phase3_readiness.recompute` and bind its complete Phase1/Phase2/CI evidence. `require_execution_admission` intentionally refuses launch because this design module supplies no actual-call admission proof.

Future author/judge adapters must honor the pinned roles, source hashing, exact structured schemas, sufficient model context, five fresh pilots, durable actual-call identity and all raw writeahead/return evidence. Future executors must use the verified current-sprint primary loader, preserve all 1,536 requests without selective omission, save full raw per-turn prompts/completions, prove H5 before each full call, and independently replay the sandbox states and tokenizer inputs. Layer48 activation capture and statistical analyses are separate pending implementations; this preparation cannot authorize them or claim they are complete. Final verification must not accept `exp31_prepared_design_v1` or `exp31_design_matrix_index_v1` as an experiment receipt.

## Validation

`PYTHONPATH=scripts:tests .venv/bin/python -m unittest test_exp31_design test_exp31_pre_action -v`

**12 CPU-only tests passed in 3.862 seconds.** They verify full counts, all twelve cell assignments, domain/arm matching, numeric/narrative/source symmetry, withheld numeric reference answers, model and typed-fact substitution rejection, missing/failed semantic evidence, fixed pilot coverage, actual clean sandbox ordinary-route execution on unit fixtures, all 96 pre-action interaction cells, exact token timing in both modes, and rejection of generated-text contamination. No unit-test effect, outcome or label is used as empirical evidence.
