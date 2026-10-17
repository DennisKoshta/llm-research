# Matched identification diagnostic v1

Frozen before inference, October 6, 2026. The user authorized autonomous continuation within the original cumulative $25 pilot budget; routine development decisions do not require renewed approval.

## Question and design

Why did identification deteriorate in the paired-mystery assay? Cross single versus paired context, J–M versus A–D candidate labels, and three inference settings: thinking disabled/greedy, thinking disabled/sampled, thinking enabled/sampled. This gives a direct matched context and label comparison, a decoding comparison with thinking disabled, and a thinking-mode comparison holding sampling temperature fixed. Thinking includes a different generation allowance, so it is a mode-and-compute intervention, not a pure template effect.

Use 25 fresh development pairs per family, both queried sides, all 12 combinations: 1,200 branches, 50 identification answers per family/cell. Seed 20261007; reuse each exact target and distractor across its comparisons. Queried mystery resolved, other unresolved; presentation order balanced by pair parity. A–D consistently relabels both candidate tables and the correct answer, preserving all evidence. One draw per sampled cell; no repeated-draw precision claim.

Qwen3-8B pinned revision, BF16, existing PyTorch inference loop. Sampled modes use temperature 0.7, full vocabulary, top-p 1. No-thinking limit 8 tokens, thinking limit 768 tokens. Greedy is a separate baseline; no greedy-thinking condition. Thinking requires a closing `</think>` and then exactly one allowed final ID. Missing closure, prose, multiple IDs, execution failures and truncations without a final answer count against accuracy and validity. Record complete raw output and tokens. Never rescore prior runs.

Qwen's [official model card](https://huggingface.co/Qwen/Qwen3-8B) documents the template switch and final-answer delimiter; it recommends sampling for thinking. Our common sampled settings serve this controlled comparison rather than reproducing the vendor's benchmark configuration.

## Decision rule

Report every family/context/label/mode cell with all assigned branches in its denominator. This is diagnosis and development selection, not a confirmatory statistical test. Candidate settings must reach at least 90% strict accuracy and 95% format validity in paired context in each family. Prefer no-thinking greedy A–D, then no-thinking sampled A–D, then thinking sampled A–D. Consider J–M only if all A–D settings fail, in the same inference order. A selected setting must undergo a fresh full assay validation (100 pairs per family/kind); this diagnostic cannot authorize behavior on its own. If none qualifies, evaluate whether simplifying the response interface or moving to the planned replication model is scientifically justified, without claiming that any preference result has been observed.

The old one-credit-v1 failure and priced-assay failures remain in the record. Any new assay revision receives a new identifier and a prospective amendment before inference. Distinguish selection from validation and report every attempted revision.

## Resources

One L40S worker, no retries, maximum 90-minute function duration and two-hour aggregate job deadline. Maximum configured GPU+CPU+RAM rate $0.00067476/second based on [Modal pricing](https://modal.com/pricing), checked October 6. Prior project compute estimate $0.87178992. Even the full two-hour bound adds at most approximately $4.86 in configured compute, leaving substantial headroom under $25. Retained volume storage and transfer costs are separate; reserve at least $3 rather than spending all $25 on compute. Reconcile app wall times after each run; never launch parallel GPU jobs. Stop and export each run before admitting another.
