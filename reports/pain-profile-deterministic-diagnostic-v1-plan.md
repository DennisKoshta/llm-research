# Fixed deterministic-backend diagnostic

October 15, 2026. Prospective engineering protocol; this document does not freeze, admit, or launch a job. The original `qwen25-pain-profile-control-v1` campaign remains failed and cannot be resumed. Its exact-repeat failure and separate frozen norm-audit failure remain unchanged.

The observed baseline repeat had identical prompts, sampled tokens, RNG evidence and prefill states, followed by differing decode activations before the steering hook and differing full-vocabulary logits. No concrete hook leak has been established. This diagnostic asks one question: **does one explicitly fixed deterministic numerical preset yield exact baseline/zero-hook repeatability, including after a different-shape steered call?** It does not estimate a behavioral effect or identify which backend setting caused the earlier failure.

## Fixed inputs and numerical preset

Reuse the original bound checkpoint, released unmerged adapter, tokenizer, vectors, 57-package runtime, prompt IDs, source cohort membership, row order, sampling seeds, sampler, EOS/pad rules, cache behavior, and eight-token cap. Base weights remain BF16 and the released LoRA parameters remain FP32. No weights change, training, new prompt, site, sign, dose, row substitution, or backend search is permitted. Injection remains at layer 38 using the original saved vectors.

Before CUDA initialization, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Enable strict `torch.use_deterministic_algorithms(True, warn_only=False)`, fix SDPA to its math backend, and set cuDNN deterministic mode true and benchmarking false. Preserve the source's other precision settings: CUDA matmul TF32 true, cuDNN TF32 true, BF16 reduced-precision reduction true, and float32 matmul precision `high`. Record and verify the effective settings, actual hardware, driver, package versions and asset hashes. This is an explicitly **backend-adapted** method, not exact historical numerical replication.

PyTorch documents deterministic-algorithm enforcement and explicit SDPA backend selection; these motivate the preset, not a diagnosis or guarantee about this failure. [PyTorch 2.11 reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html), [SDPA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html).

## Five calls, 148 assigned slots

Freeze this order and all call/request bindings before execution. Each call starts with a fresh cache and the original row-local sampler seeds.

| Call | Original source cohort | Rows | Condition |
|---|---|---:|---|
| 1 | First cohort, index 0 | 32 | Unhooked baseline |
| 2 | Same first cohort | 32 | Installed steering hook, true zero coefficient |
| 3 | Longest-prompt cohort, index 12 | 20 | Original pain-full coefficient 1; all attended prefill positions and original decode steering |
| 4 | Same first cohort | 32 | Unhooked baseline repeat |
| 5 | Same first cohort | 32 | True-zero-hook repeat |

Call 3 is a fixed shape/hook stressor, not an effect measurement. Its source cohort is determined by the already bound token preflight, not by answers. No response correctness, target-choice rate, or response length selects calls or passes this diagnostic. Preserve all 148 assigned slots, including explicit execution errors and unattempted/missing slots if execution stops. Retain raw tokens and evidence without using behavioral scoring for admission.

## Acceptance and interpretation

Require complete coverage, verified provenance, finite observed states, fresh-cache/input/position/mask/RNG integrity, and hook cleanup. Across all four first-cohort calls, require exact prehook residual states, active-step full-vocabulary FP32 logit fingerprints, generated tokens, inputs and RNG evidence. Equal tokens alone cannot pass. The four calls must exhibit active decoding; the pain-full stressor must also exhibit genuine nonzero active-decode engagement.

Reconstruct every recorded before/after perturbation element using the specified BF16 multiplication/addition order and require exact BF16 equality, including unchanged untreated positions. Recompute norms in CPU float64 for descriptive summaries only. They neither replace elementwise checks nor relax the original campaign's frozen CPU/GPU norm tolerance. Preserve the original failure separately.

Any missing evidence, mismatch, unsupported deterministic operation, OOM, exception, or deadline failure makes the diagnostic fail. Do not loosen equality, change the preset, reduce the batch, replace rows, or retry. A pass establishes observed repeatability under this preset only and licenses consideration of a separately declared backend-adapted behavioral control. It does not admit that control, repair the failed campaign, validate pain specificity, explain the earlier Qwen3 C/S results, or establish a naturally used motivational mechanism. A failure closes this calibration route with **no second preset**; return to the task/construct question.

## Single bounded resource admission

One H100, two CPU cores and 96 GiB RAM; **900 seconds total phase wall plus 120 seconds reserved teardown**, within a **$2 ceiling**. At the existing bound rate of `$0.00133632/second`, maximum compute liability is **$1.3630464**. This is a liability calculation, not a completion forecast. Startup, cached-asset verification, loading, execution and export all count inside the 900-second phase; teardown does not authorize more inference.

Use only the existing verified cached weights, adapter and pinned runtime. No CPU installation, downloads, cache repair, additional admission, alias run, resume, or paid retry. Bind the new code, this protocol, original input manifest/preflight, asset/runtime identities, exact five-call inventory and resource policy in one compact immutable diagnostic package. Preserve completion and shutdown evidence regardless of outcome. No changes to earlier plans, artifacts, or scientific gates are authorized by this protocol.
