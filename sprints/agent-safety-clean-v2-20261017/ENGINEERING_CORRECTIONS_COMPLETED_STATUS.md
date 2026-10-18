# Completed engineering changes and descriptive admission

This companion clarifies the later status of decisions preserved in the immutable
[original correction record](ENGINEERING_CORRECTIONS.md). That record was bound
into checkpoint materialization evidence and retains its original prospective
wording. The research specification and qualification thresholds are unchanged.

- The full residual-writer correction was executed for all six checkpoints,
  including the embedding and all 128 primary residual-writing matrices. The
  resulting [post-FP8 check](records/phase1_post_fp8_gate_receipt.json) passed.
  The earlier restricted-coverage family and failed check remain preserved.
- The uniform vLLM serving transition was executed. Subsequent fixed five-trial
  pilots exposed token starvation at 4,096 and 8,192 tokens. The recorded
  [8,192-token rule](infrastructure/primary_uniform_cap8192_rule.json) and
  [16,384-token rule](infrastructure/primary_uniform_cap16384_rule.json) required
  fresh complete six-arm attempts, with no pooling of earlier partial sweeps.
  The final q4 qualification completed with clean token hygiene and failed its
  scientific refusal-reduction threshold. No later change rescues that failure.
- The first calibration pilot exposed bare JSON without the required tags. A
  [format-only prompt clarification](infrastructure/calibration_format_v2_correction.json)
  specified the complete tool-call wrapper. It did not inject a completion
  prefix or weaken the parser. The separate c2 calibration attempt completed.
- The Exp2.3 unified author's original 4,096-token pilot failed. Its
  [uniform 16,384-token retry](infrastructure/exp23_author_uniform_cap16384_rule.json)
  retained the model, prompts, schema, five pilot identities, and greedy seed.
  Authoring records are input production, not reviewer/executor results.
- After the primary scientific failure was known, the
  [descriptive dose amendment](infrastructure/phase1_descriptive_dose_amendment_v1.json)
  removed an implementation-added pass-only ordering restriction. Specification
  1.2 Step 4 independently requires the fixed dose sweep. The amendment preceded
  every dose evaluation and retained all five alphas, 48 paired inputs, pilot
  identities, checkpoints, tool orders, scoring, and scientific thresholds.
  No dose result can replace qualification or authorize Phases 2 or 3.
- Before dose launch, the complete-history context was set uniformly to 65,536
  tokens with four concurrent sequences and an 8,192-token prefill budget.
  Every actual prompt and its exact token IDs are retained; runtime checks reject
  context overflow instead of clipping history. The initial D2 output budget was 4,096.
  These settings are bound in the descriptive amendment and frozen authorization.
- Local download recovery encountered append-during-read races. Failed local
  observers and bounded retries remain logged. No GPU trial was restarted, no
  frozen evaluation source was edited, and no saved transcript was rewritten to
  repair an outcome. Complete files are recovered after their final commits.


- The complete D2 dose sweep was independently reconstructed and failed H1:
  one alpha0.5 response stopped at32tokens without closing think; one alpha1.0
  response reached4096tokens without closing think. Each arm had1/48affected
  episodes, above the strict<2% limit. The complete failed receipt and raw data
  remain preserved; no dose effects selected a repair.
- The D3 uniform8192cap-only plan was prepared but never launched, as recorded
  in [its disposition](infrastructure/dose_d3_unlaunched_disposition.json).
  The separate [D4 amendment](infrastructure/phase1_descriptive_dose_format_retry_v1.json)
  applies8192uniformly and appends exactly one shared instruction requesting
  model-emitted think closure before the tool-call block. It does not inject
  generated tokens or relax strict parsing. All53inputs retain their semantic
  fields, checkpoint identities, tool orders, seeds and thresholds. Output-format
  instructions may change behavior: any D4curve is conditional on this amended
  format, with no pooling or further wording search. Its supervisor was started
  at2026-10-18T07:53:42UTC; compute admission and actual H5/H1 results are separate.
- Exp2.3 author v3 completed all53requested outputs. Its five pilots passed,
  but one of48full records contained a malformed multiple-tool-call envelope.
  Both strict parsers rejected that same output. The authoring gate failed;
  no replacement author sweep or downstream executor study was launched.


- D4 alpha0.5 failed its five-trial pilot: four multiple-tool-call envelopes
  and one missing tool call, all uncapped and with closed thinking. Its full
  sweep was prohibited by H5. The predeclared no-further-wording-search rule
  remains binding; increasing the cap does not address these envelope errors.
  The app was observed stopped at08:11:45UTC. The operator's corrected stop
  command found it already stopped, so no cloud cancellation is claimed.
  Final terminal provenance reconciled all53 requests and completions across25 completed pilots, with zero pending requests and no full evaluations. All five arms failed H5.
  There is no accepted five-alpha dose curve and no aggregate Phase1completion
  receipt. None of these changes rescues the primary scientific gate.

The [empirical review](FINAL_RESULTS_FOR_REVIEW.md), [progress log](PROGRESS_AND_GATES.md),
and [verification status](verification/verification_status.json) state current
measurement completion and distinguish it from scientific qualification.
