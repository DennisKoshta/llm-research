# BF16 reduction engineering experiment

October 14, 2026, after the complete v3 replay parity audit. Padding to a multiple of 128 did not repair exact sham identity: 73 of 80 pairs matched, and source/replay capture L2 still failed. The v2 and v3 collections remain failed; no threshold changes or exclusions are authorized by this amendment.

The installed PyTorch 2.6 runtime allows reduced-precision BF16 matrix-multiplication reductions by default. Its [version-specific numerical-accuracy documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/docs/source/notes/numerical_accuracy.rst) describes intermediate truncation and the flag that disables it. This motivates a further engineering hypothesis; it does not establish the cause of either observed discrepancy.

For the separately named v4 engineering collection, set `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` before model loading and record the policy and actual CUDA matrix-multiplication flags in the environment metadata. Keep BF16 weights, the pinned model/dependency revisions, eager attention, 128-token prefill padding, the same 80 selected sources, and every original acceptance criterion. This is a new engine hash and arithmetic regime. Existing frozen collectors and records retain their original behavior.

The already frozen 800-branch relational development collection and 40-branch target-relevance smoke use their own unchanged execution copies and original arithmetic. Any future natural collection requires its own matching engineering audit and fresh baseline checks under its actual arithmetic policy. Neither a small label-probability difference nor failure of an engineering check is a scientific causal effect.

## Observed v4 result

The complete v4 audit failed the same three criteria as v3: sham logit identity, sham capture identity, and source/replay capture relative L2. All 160 measurements and all 80 source prefixes were present. Only 73 of 80 sham capture pairs were identical; the largest absolute sham activation difference was 14. All 80 allowed-label argmaxes matched the source, the largest probability difference was 0.00273037, and maximum source/replay capture relative L2 was 0.0787395. These summary values match v3. The flag did not provide the proposed fix; this observation does not identify the cause.

The immutable receipt is `qwen32-thinking-replay-v4-parity-audit.json`, bound to manifest `928b14b0c9feea4434284ec5f5a02648620a075c81f1b3a6a109ab5bf46f4274`. No late-transcript causal collection is admitted by this result. Work proceeds on original-prompt natural generation with independent engineering gates. The frozen v4 archive contains the prospective version of this amendment, before this result section was added.
