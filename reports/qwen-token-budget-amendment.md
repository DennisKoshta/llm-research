# Bounded generation-allowance amendment for Qwen one-credit v2

October 7, 2026, after the complete 768-token Qwen validation failed and while the prespecified OLMo check is running. This is an adaptive development decision motivated by recorded truncation. It is not preregistered confirmation, a repair of earlier scores, or an additional search over prompt wording.

## Rationale and fixed change

The complete v2 audit found 91 invalid responses, all at the 768-token limit without EOS. Eighty-nine lacked thinking closure. The useful-clue checks failed, while identification and consequence checks passed. This supports checking the response allowance as an operational constraint; it does not establish how unfinished responses would end.

Make one bounded allowance increase to **2,048 generated tokens** for both diagnostics and any subsequent choice calibration. Keep Qwen3-8B's exact revision, BF16, eager attention, H100, full-vocabulary temperature 0.7, A–D candidates, A–C actions, 1–4 consequences, thinking template, prompts, strict parsing, and all scientific gates fixed. Do not force an answer, alter the thinking trace, or resume only failed responses. Preserve the complete 768-token result as a failed setting.

This remains the same semantic one-credit-v2 interface. The new allowance is explicit in every assignment and in the run configuration; manifests, source hashes, and output directories distinguish it from the original setting. Existing settings retain their previous defaults. Finish any eligible OLMo calibration before installing the implementation change so its source-matching requirement is preserved.

## Fresh collection and stop rule

Use seed 20261010 and a fresh 100 pairs per family: 1,400 diagnostic branches. The unchanged gate requires at least 90% strict correctness and 95% response validity in all fourteen family/check cells, with all assignments terminal. No additional token-limit search follows this setting within the current pilot. Do not select an allowance based on discretionary choice effects.

Only if all checks pass, prepare 1,200 choices on the same new development pairs with the same model, source, hardware, allowance, and task. Apply the existing calibration criteria: at least 95% valid per family/condition/layout cell; at least 10% absolute probability for each mystery in both unresolved layouts; at most 15 percentage points of layout sensitivity for either mystery. No positive resolution-effect gate. All assigned trials, stops, and invalid responses remain in denominators.

Use the existing independent audit and descriptive response diagnostics. Report any residual truncation and wrong answers. Conditional descriptive intervals resample complete pair clusters using the already specified procedure. Neither successful comprehension nor calibrated choices establish a motivational mechanism or subjective experience.

### Descriptive sensitivity check before intervention development

Specified before any choice collection: for each family and layout in both-unresolved trials, report the probability of selecting either clue and of stopping. The maximum possible upward change in information choice is one minus its baseline probability; if invalid and execution-failure rates stay fixed, only the stop probability is available for conversion to information choice. The maximum downward change is the baseline information-choice probability. These are sample-based algebraic limits, not confidence bounds, power estimates, or new advancement thresholds.

Keep the original calibration gates unchanged. A calibrated task can still have too little room for a planned amplification or suppression contrast. Also, this one-credit instrument has no separate non-information optional activity control, so decreased information choice alone cannot establish motivation-specific satisfaction. Use these limitations to assess what subsequent experiment the task can support; do not use them to hide scientific nulls or select a desired resolution effect.

## Execution and budget

Enter after OLMo's currently eligible collection is completed and all apps have stopped, if the cumulative budget guard can admit the run. Use up to four exactly reserved H100 workers, a new parent deadline of at most two hours, and the existing scheduling-only continuation rules if needed. The combined reservation, including startup headroom, must fit the remaining $100 total after the $10 reserve. A larger token cap permits longer responses but does not extend the collection deadline or budget. Subsequent calibration requires its own admission within the remaining allowance. Preserve incomplete collection if the resource limits prevent completion.

### Exact GPU allocation and stopped-slot recovery

During startup, one worker received an H200 and was rejected by the existing hardware check before any study response or integrity trial. Its failed app and empty export are preserved. [Modal documents](https://modal.com/docs/guide/gpu) automatic H100-to-H200 upgrades at the requested price and an `H100!` request that disables them. Future H100 launches now use that exact request; the scientific hardware field and runtime check remain H100.

A stopped reserved worker may be resumed in the same reserved slot only after all its prior apps are verified stopped with zero tasks. The guard retains the original deadline and exact manifest, rejects unrelated active project apps and overlapping resumptions, rechecks the combined remaining liability against the current allowance, and allows at most two local resumes in addition to the original admission. Remote admission remains capped at three. This permits the rejected worker to resume while its three correctly allocated peers finish, without expanding the reservation or repeating a completed answer. It changes execution only; original coordinator records and additional attempt records remain separate.

At 08:22 UTC, timing-only summaries showed worker 3 averaging 18.60 seconds per response over 61 saved responses, projecting approximately 108.5 minutes for its 350 branches, beyond the 90-minute function timeout. It will be explicitly committed, stopped, exported, and resumed once in its existing reserved slot under the same absolute deadline. The projection and scheduling decision use only completion counts, inference durations, and token counts; no correctness fields were inspected. Reallocation may or may not improve throughput. Existing responses remain terminal, and the original admission limits remain in force.
