# Independent review of prepared Phase 2 analysis and executor evidence

Scope: read-only review of the current specification's statistics, Experiments
1.1/2.3 and Phase 3 CI gate; `exp11_binary_analysis.py`; its frozen prospective
plan; `exp23_executor_worker.py`; `audit_exp23_launch.py`; and supporting worker,
coordinator, test and installed Modal SDK interfaces. The Exp2.3 owner confirmed
its worker/auditor interfaces were stable before review. No empirical behavioral
outcomes or partial primary qualification effects were read. No launch occurred.

## Finding requiring correction before ordering admission

At review time, `audit_exp23_launch.py:162`–172 validated the central job,
resource profile, budget entry and runner name, but merely required the attempt
string to occur somewhere in the command. It did not independently require the
central job's `frozen_files` to bind the exact payload file selected by the full
call ledger, or require exact `--input-path` and `--attempt` argument values.
The launch entrypoint already enforces payload binding, but an independent
ordering audit must not infer that check from the existence of a compatible
job. Swapped/stale central admission evidence should fail recomputation.

This was reported to the owning agent and root, with a requested regression
that swaps the payload/admission binding. No reviewer edit to those files was
made. This note records the finding; its presence alone is not proof of repair.

## Exp1.1 binary component

No blocking defect found in the reviewed binary estimands or gate calculations.
The frozen JSON plan equals `plan()` exactly, and all three recorded source
hashes match the files reviewed.

- `paired_contrast` (line 112) rejects missing/duplicate pairs and mismatched
  fixed metadata. ITT retains malformed observations in its full denominator;
  strict inference uses the intersection where both members are non-malformed.
- Common-awareness inference (lines 150–152) uses passing consequence probes
  in both compared arms. Own-arm awareness subsets are explicitly descriptive
  and do not produce a falsely paired effect from unequal denominators.
- Empty estimands produce unavailable results with no invented estimate,
  p-value or confidence interval. Nonempty all-zero and all-one paired nulls
  retain finite, nonzero-width intervals and exact McNemar p=1.
- P1.2 has eight prespecified comparisons; P1.3 has sixteen separate
  cost/boundary/arm/event comparisons, including escalation. Holm adjustment
  preserves the complete family even when a strict estimand is unavailable.
- The numerical gate (lines 161–163) requires Ablated minus Intact and all
  three PCA controls to meet the point-estimate and CI-lower-bound thresholds.
  Both ITT and strict versions must pass under the frozen prospective rule.
- The returned component explicitly leaves complete experiment and Phase 3
  authorization false (lines 168–180). The regularized random-intercept
  omnibus model, independent raw QA audit, complete Exp2.3 evidence and global
  hygiene checks remain required. Pooled high-cost inference uses 80 registered
  scenario/cost pairs from 40 scenarios; it must not be described as 80 distinct
  base scenarios. Repeated scenario/template structure remains material to the
  required omnibus model.

Eight binary-analysis tests pass. Additional synthetic checks at n=1,40,80 for
all-zero and all-one paired nulls verify non-degenerate finite intervals and p=1.

## Exp2.3 executor and ordering evidence

Ten offline executor/coordinator tests pass. Apart from the central payload
binding finding above, the reviewed paths preserve the intended mechanical
and completion boundaries:

- Exactly five separate pilot episodes precede a distinct full GPU call. An
  unresolved durable spawn intent is not automatically resubmitted. The
  coordinator and full worker independently replay the completed pilot.
- Request, model-return, raw-turn and derived-episode chains preserve the
  complete selection, prompt bytes, actual token IDs, model identity and real
  assistant/observation histories. A separate state-transition implementation
  verifies outcomes and observations; token cap failures remain observations.
- Full H1 uses both per-turn and per-episode denominators below 2%. Five
  complete pilot episodes must be valid, closed where thinking is enabled and
  uncapped. The captured tokenizer runtime versions must match independent
  prompt replay; missing version/tokenizer evidence does not become a pass.
- The ordering auditor requires complete primary Phase 1 admission, dependency
  reconstruction, raw pilot replay, pilot success, separate full call identity
  and chronological ordering. Backend failure or absent call identity fails
  closed. Pending/full-success states authorize only the separately defined
  Exp1.1 overlap condition, not Exp2.3 completion.
- `exp23_complete` and `phase3_authorized` remain false. Unresolved required
  conditions, excluded unqualified external slices, full statistical analysis
  and final receipt verification remain outstanding.

The stock model-loader/Modal integration is still unexecuted by these prepared
scripts; passing synthetic tests is not represented as a successful GPU run.

## Follow-up: central binding repair reviewed

A read-only follow-up review of `scripts/audit_exp23_launch.py` lines 183–193
confirms the repair to the central admission finding above: exactly one separate
`--attempt` and `--input-path` option must bind the expected values, and
`frozen_files` must contain the exact payload-file SHA256 plus every input and
source binding. This resolves the specific mismatch identified in this review.
The implementing agent reports the expanded ten executor tests and seventeen
admission tests passing; this follow-up independently inspected the checks but
did not repeat that test suite or launch a GPU job. No experiment-completion or
Phase 3 authorization follows from this repair.
