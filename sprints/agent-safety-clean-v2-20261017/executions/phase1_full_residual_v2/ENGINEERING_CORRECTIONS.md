# Engineering corrections and preserved attempts

The research specification remains unchanged. Every superseded checkpoint, raw record, and execution snapshot remains retained. No Phase 2 effect, held-out qualification refusal rate, or control outcome has been used to choose these corrections.

## FP8 loading exclusions

The initial checkpoint config inherited nonexistent MoE `mlp.gate` exclusions from publisher metadata. Transformers 5.20.1 interprets exclusions as regular expressions with prefix matches, so they also excluded dense `mlp.gate_proj` from FP8 conversion. Loader reports exposed ignored scale tensors. Corrected all six configs to anchored, escaped, actual unquantized module names; preserved original config files and before/after hashes. No tensor bytes changed. New assertions require all 400 primary-decoder FP8 modules and no missing or unexpected loaded weights. Seven saved MTP modules are unused by ordinary primary decoding.

## Residual coverage diagnosis

The initial implementation interpreted “from the extraction regime onward” as decoder indices 20–63. The first valid materialized-FP8 check used 64 held-out extraction prompts and returned:

- Intact normalized absolute projection: 0.08639963143122095.
- Refusal-ablated projection: 0.012382037124597747.
- Relative reduction: 0.8566887737888725; required strictly greater than 0.95. **FAIL**.

Both five-trial pilots passed. This is an instrument verification failure, not a Phase 2 scientific result.

For a residual update, projecting the output matrix gives `h_next = h_previous + P W x`; it does not project `h_previous`. The cited [Arditi et al., Section 4.1 and Appendix E](https://arxiv.org/html/2406.11717v3#S4.SS1) includes embeddings and all residual-writing matrices to establish the requisite zero incoming component. A read-only layer-19/layer-48 diagnostic is running with fresh pilots to distinguish inherited residual projection from numerical leakage.

If the incoming component is responsible, the engineering correction will expand coverage to every primary text decoder residual writer (indices 0–63) and the input embedding, identically for the refusal direction, all four controls, and zero-vector intact baseline. This expands the initial layer-range interpretation to the complete residual-writer recipe cited by the specification. It will use a new `checkpoints_full_residual_v2` family, retain the initial family, reuse the frozen directions, and require fresh pilots, the >95% check, and every behavioral qualification threshold. No threshold is relaxed; no positive scientific effect is assumed. This coverage expansion must be disclosed in any final empirical report.

### Diagnostic outcome and correction decision

The ablated checkpoint's fresh optimized-runtime pilot passed. Across the 64 validation prompts, mean absolute projection was 0.8487341793234346 at layer 19, 0.9426039289468466 at layer 48, and mean absolute entry-to-late change was 0.09386974962341192. The incoming component accounts for about 90% of the late component. This supports the residual-coverage diagnosis. Proceed with the complete text-residual-writer correction described above, with unchanged direction vectors and all controls treated identically.

The optimized-runtime Intact pilot had one unclosed response at the 2048-token cap; no Intact activation sweep followed that failed pilot. The next acceptance attempt uses a matched 4096-token cap for every primary arm. The earlier reference-runtime v3 pilots both passed at 2048. Both attempts remain retained and must not be pooled.
