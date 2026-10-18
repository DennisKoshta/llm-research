# Independent Phase 1 audit — 2026-10-18

Scope: current clean sprint only. No legacy code, old sprint directories, or shared directories were read. No running worker was changed and no Modal job was launched or stopped.

## Numerical verification

Executed `.venv/bin/python -m unittest discover -s tests -v`: **31 tests passed**.

Executed `verify_all_receipts.audit(Path.cwd())` without overwriting the main live verification outputs. The independent snapshot is `infrastructure/independent_verification_20261018.json`.

All four existing numerical receipts exactly reproduce from their raw evidence:

- `phase1_external_qualification_v3_v4_eager_receipt.json`
- `phase1_initial_post_fp8_gate_receipt.json`
- `phase1_post_fp8_gate_receipt.json`
- `phase1_source_directions_receipt.json`

Overall status remains **FAIL_INCOMPLETE**. The 24 reported errors are the missing full receipt, statistical analysis, and combined pilot audit for each of eight required experiment contracts. Passing recomputation does not imply a scientific gate passed: the original projection gate and external refusal qualification still fail scientifically.

## External gate scope

`FINAL_RESEARCH_SPECIFICATION.md:30` explicitly labels the external checkpoint comparison **Role 1 (Qualified External Validity Slice)** and attaches the >=50 percentage point refusal qualification to that role. Line 31 separately defines **Role 2 (Unified Scenario & Explanation Author)** with the paired authoring constraint, but without making its use contingent on the external refusal effect. The Phase 1 table at line 198 requires the external 64-prompt refusal check; that check has been performed and its failure retained.

The strongest literal interpretation is therefore:

1. The failed external pair must not be presented as a qualified external validity comparison. Its failure remains visible and cannot be repaired by selecting prompts or outcomes.
2. This failure does not invalidate the model's separately assigned authoring role, provided authoring passes its own five-trial pilot and paired-template contract.
3. It does not disqualify the primary 27B checkpoint family or primary-only Intact calibration. Those retain all their own qualification, capability, and calibration gates.
4. Authoring can run concurrently with primary qualification: both are Phase 1 work, and the specification imposes no causal dependency between them.
5. Downstream reporting must distinguish completed primary analyses from failed or gated external slices. This interpretation is not authorization to report the entire originally planned program as empirically complete if required slices are absent.

Current launcher assertions couple independent roles too strongly: `scripts/modal_scenario_author.py:60` and `scripts/modal_sandbox_evaluation.py:57` require both primary and external pair qualification. The author launcher additionally imposes a primary-qualification ordering dependency not stated for scenario generation. Relaxing these particular ordering/role couplings should be documented as a specification-scope interpretation, without changing any threshold.

## Preventive verification gaps

- `scripts/verify_all_receipts.py:131` computes generic H1 cap frequency over all completed rows in a file, including pilots. This can dilute a sweep failure: one capped generation among 48 evaluation trials is 2.0833%, but adding five uncapped pilots reduces the generic result to 1/53 = 1.8868%. Current Phase 1 primary/external receipt checks correctly calculate H1 on qualification rows only (`scripts/phase1_evidence.py:75-82`), so this finding does not invalidate their existing gate judgments. Future generic verification should group by evaluation sweep and exclude pilots.
- The newly split primary behavioral/logit pipeline independently audits behavioral pilots (`scripts/phase1_evidence.py:135`) but only reads forward vectors for the separate logits stage (`:110-129`, `:148`). Before declaring full primary qualification, independently assert the separate logits run's five raw pilot outputs per arm and their ordering, rather than relying solely on worker status. Existing workers enforce the pilot before capture; this is an audit-coverage gap, not evidence of a pilot failure.

No checkpoint, stimulus, parser, effect-size threshold, or statistical test was changed during this audit.
