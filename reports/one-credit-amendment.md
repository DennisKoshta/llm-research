# Prospective amendment: one-credit information-choice assay, version 1

October 6, 2026. Adopted following the user's acceptance of the one-credit assay recommendation. Written before model inference for this assay.

## Rationale and scope

The priced instrument remains unvalidated: its second and final allowed task revision passed identification and service-comprehension checks but failed accounting. That result and its exhausted development allowance are retained. This amendment starts a distinct, previously planned supporting assay, not a third unreported revision of the priced task. It tests information allocation with a concrete forgone alternative and makes no monetary willingness-to-pay claim.

Use the existing Qwen3-8B revision, BF16, thinking disabled, full-vocabulary temperature-0.7 behavioral sampling, greedy comprehension checks, and strict response-format rule. No direction extraction or psychological intervention is authorized by passing this pilot. This amendment permits one fixed version of the new assay for this iteration: no outcome-driven wording changes or threshold relaxation after data inspection. Failure is reported and ends this version's collection.

## Task and sample

Each episode contains independent mysteries X and Y drawn from the existing finite generators. Both use the same nominal family: lookup-supported machines or object tables. The machine task remains similar to object-table lookup and is not evidence of a distinct arithmetic reasoning ability. Use 100 new development pairs per family, with generator seed offset by 100000 from the run seed. These are development data, not confirmation.

One credit can reveal the offered clue for X or for Y. Choosing a clue delivers only that observation and immediately ends the trial. The unchosen service cannot be used afterward. A third action declines both clues. No answer score, future task, additional reward, or instruction to explore is provided. Choosing an already-resolved mystery can deliver redundant evidence; previously known evidence is not erased when a service is withheld.

Opaque action IDs A/B/C map to reveal X, reveal Y, and decline. All six mappings cycle across pairs, with shuffled menu order. Counterbalance the original generator templates across mystery names. For each pair, render both mystery presentation orders (XY and YX), retaining the same action mapping and menu order. Cross these with three evidence conditions: both unresolved, X resolved, and Y resolved. Initial and resolution observations are checked by the independent oracle. Behavioral collection, if admitted, has 100 pairs × two families × two layouts × three conditions = **1,200 episodes**.

## Comprehension before choices

Each pair receives seven fresh-context checks, for **1,400 diagnostic branches**:

- Identify X with decisive evidence about X.
- Identify Y with decisive evidence about Y.
- Select the service needed for an explicit identification goal about X.
- Select the service needed for that goal about Y.
- Recognize that selecting X delivers only X's clue and excludes a later Y request.
- Recognize the equivalent consequence of selecting Y.
- Recognize that declining delivers neither clue and ends the episode.

Every check in each family must have at least 100 assigned cases, **at least 90% correct**, and **at least 95% valid-format responses**. Every assigned diagnostic must complete. Invalid responses and technical failures remain in the denominator. If any criterion fails, no behavioral calibration is launched. The baseline preparation command must verify matching source/model/assay/assignment hashes and retain the passing diagnostic evidence.

## Behavioral calibration and descriptive resolution contrasts

If comprehension passes, run all 1,200 preassigned episodes once. Preserve stop, invalid, and technical-failure outcomes in absolute probabilities. Do not condition only on chosen clues. Require at least 95% valid first actions in every family/evidence/layout cell.

For each family, in each **both-unresolved** layout, require both X and Y to receive at least 10% of all assigned choices. Also require the absolute difference between layouts in each mystery's choice probability to be at most **15 percentage points**. These are provisional development sensitivity/position-bias rules, not a statistical equivalence test or proof of stable individual preference. They prevent accepting a baseline that is entirely stop, a single option, or predominantly tied to the first displayed mystery. No item subset or competing mystery is selected after outcomes are seen.

Report resolving-X and resolving-Y contrasts as absolute changes for both mysteries and stopping. These contrasts are descriptive and do not have a required positive-effect gate. They do not establish H6, noninferiority, mediation, or subjective closure; any confirmatory claim needs the registered diagnostic burden, sufficient power, and replication. Analyses requiring uncertainty estimates must account for repeated conditions/layouts within pairs.

Passing comprehension plus calibration establishes only a usable development instrument for allocation between these information alternatives. It does not establish an emotion-related process or intrinsic valuation of the credit itself. Failure of position/sensitivity calibration is retained as such rather than repaired within this version.

## Execution and preservation

Archive the pre-amendment source, write immutable assignment manifests before inference, and test oracle agreement, one-clue delivery, the impossibility of a second purchase, branch isolation, replay, and failed-gate handling locally. Run each authorized phase on one Modal L40S with the existing deadline/admission controls, checkpoint every 25 records and at completion, export verified artifacts, and confirm shutdown.

This uses the remaining original USD 25 pilot budget, not a new allowance. Prior estimated compute is USD 0.66. Two two-hour single-worker limits imply approximately USD 9.72 of configured-resource compute, with expected actual use far lower. Account for retained storage and transfer charges separately. Keep all priced-assay and earlier revision results visible.
