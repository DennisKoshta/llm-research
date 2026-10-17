# Relational development: early transfer is weak, while final decision states separate information gain

October 14, 2026. **Development evidence only.** The complete 800-branch relational instrument passes its original behavioral gates: 799 correct and valid responses, no cap hits. Yet the original pooled balanced information axis at the prespecified **layer 47 / record** site has AUC 0.4888–0.5056 across the two families and opposite yes/no questions. At **layer 57 / decision**, the same site-specific frozen-axis family gives AUC 1.0000 in all four family/polarity groups, with substantially less stable accuracy at the original extraction threshold. This is evidence of late representational transfer, not a demonstrated causal mechanism or a replacement for the prespecified early candidate.

All 3,360 stored deterministic metric summaries were independently recomputed from the saved scores and original thresholds. AUC, threshold accuracy, complete-world count and within-world concordance match exactly; mean factorial gaps differ by at most 2.28×10⁻¹³ from arithmetic ordering. Saved bootstrap intervals were not independently recomputed and are not used here to declare significance. No target-relevance outcomes were inspected for this report.

Companion figures show the complete pooled balanced site matrix for new, repeat and instrumental action: [AUC figure](/home/dennis/projects/llm-curiosity-and-satisfaction/reports/figures/relational-information-v1-development/pooled-balanced-transfer-auc.png), [fixed-threshold accuracy figure](/home/dennis/projects/llm-curiosity-and-satisfaction/reports/figures/relational-information-v1-development/pooled-balanced-transfer-accuracy.png). Publication exports are available as [AUC PDF](/home/dennis/projects/llm-curiosity-and-satisfaction/reports/figures/relational-information-v1-development/pooled-balanced-transfer-auc.pdf) and [accuracy PDF](/home/dennis/projects/llm-curiosity-and-satisfaction/reports/figures/relational-information-v1-development/pooled-balanced-transfer-accuracy.pdf).

## What was measured

The model is Qwen/Qwen3-32B at revision `9216db5781bf21249d130ec9da846c4624c16137`. The development collection has 20 assigned base-world indices per family, four correlated-property/query cells per world, five question kinds, table rendering, and a 4,096-token generation allowance. The complete four-cell world is the repeated-measurement block. Each family/kind summary contains 20 assigned worlds and 80 branches; the 800 branches are not 800 independent experimental units. These are raw assigned-index counts, not a claim of 20 semantically unique worlds per family.

The relational task requires inferring whether an offered report is informative from the candidate table and an observed fact. The known and queried properties have different names. Its binary gain pattern is `[0,1,1,0]` across cells `(correlated,query) = (0,0),(0,1),(1,0),(1,1)`. This is more demanding than identifying whether the report repeats the same named property. Full-world information gain still equals accurate-report entropy within the compatible candidate set; this assay does not separate those quantities.

The 12 source axes—pooled, objects and machines, each with new, repeat, balanced and answer directions—were frozen in the original information study. Each layer/position has its own fitted direction and extraction threshold. Applying “the balanced axis” at different sites does not mean one common vector has been moved through the network. No transfer direction, sign, intercept or threshold was refitted. All 12 × 2 families × 5 kinds × 4 layers × 7 positions = **3,360** summaries are retained in [the original analysis](/home/dennis/projects/llm-curiosity-and-satisfaction/analyses/relational-information-v1-development/analysis.json).

Behavioral validity and classifier accuracy are different quantities. The former tests the model's generated answer; the latter asks whether a frozen projection exceeds its old extraction threshold. A model can answer correctly while that threshold classifies its hidden state poorly.

## Behavioral qualification passed at the declared development scale

All 40 family/kind/cell groups contain their 20 assigned completions and satisfy the original gates: accuracy at least 90%, validity at least 95%, and at least 20 assigned indices. Objects have 400/400 correct and valid responses; machines have 399/400. The single invalid/incorrect response is in machines/repeat, correlated=0, query=1: 19/20 valid and correct, exactly the 95% validity boundary. No capped or invalid response was removed from the representation analysis. The collection generated 315,544 tokens and had no cap hits.

This establishes that the small development instrument is usable for this readout. It does not convert a 20-world-per-family study into the larger planned confirmation, nor repair the separate original study's failed global count gate.

## The fixed early candidate did not transfer in this assay

The exact layer-47 record results for the pooled balanced axis are:

| Family | Question | AUC | Old-threshold accuracy | Mean gain contrast | Worlds with positive gain contrast |
|---|---|---:|---:|---:|---:|
| Objects | New | 0.505625 | 0.5000 | 0.087373 | 13/20 |
| Objects | Repeat | 0.497500 | 0.5000 | 0.008353 | 13/20 |
| Machines | New | 0.488750 | 0.5000 | −0.064031 | 8/20 |
| Machines | Repeat | 0.499375 | 0.5000 | −0.007052 | 9/20 |

These near-chance observed rankings do not support robust early transfer of this frozen direction to the relational problem. They do not prove that the model has no early representation, that another subspace cannot encode the quantity, or that perturbing this component can have no downstream effect. They also do not license changing the causal candidate to a favorable later site.

The record boundary precedes the question. For a fixed world and cell, its factual prefix is identical across all five kinds. Consequently, five record-kind summaries are repeated measurements of the same conceptual state, not five independent task replications. Small cross-kind numerical differences are not evidence that a causal transformer at this boundary has read the later question.

## All planned sites for the pooled balanced axis

Each entry below is **AUC / accuracy at the original extraction threshold**. Every available new/repeat entry uses all 20 complete four-cell worlds. AUC 0.5 is chance ordering; AUC below 0.5 is retained with its original sign. No sign reversal or threshold recentering is applied.

| Layer | Position | Objects: new | Objects: repeat | Machines: new | Machines: repeat |
|---:|---|---:|---:|---:|---:|
| 15 | record | 0.5212 / 0.5000 | 0.5162 / 0.5000 | 0.4975 / 0.5000 | 0.4519 / 0.5000 |
| 15 | prefill | 0.4838 / 0.5000 | 0.5100 / 0.5000 | 0.5431 / 0.5125 | 0.4838 / 0.4750 |
| 15 | token_32 | 0.5494 / 0.5000 | 0.4600 / 0.4625 | 0.5531 / 0.5250 | 0.5231 / 0.5125 |
| 15 | token_128 | 0.5181 / 0.5125 | 0.4919 / 0.5250 | 0.4781 / 0.5000 | 0.4806 / 0.4750 |
| 15 | token_512 | Unavailable | Unavailable | Unavailable | Unavailable |
| 15 | think_end | 0.4406 / 0.4125 | 0.4550 / 0.4625 | 0.2687 / 0.3500 | 0.6556 / 0.5250 |
| 15 | decision | 0.3969 / 0.4375 | 0.4719 / 0.4875 | 0.2838 / 0.3875 | 0.6419 / 0.5625 |
| 31 | record | 0.5175 / 0.5000 | 0.5106 / 0.5000 | 0.4944 / 0.5000 | 0.5069 / 0.5000 |
| 31 | prefill | 0.5125 / 0.5000 | 0.5025 / 0.5000 | 0.5300 / 0.5000 | 0.5256 / 0.5000 |
| 31 | token_32 | 0.5387 / 0.5375 | 0.4475 / 0.4625 | 0.5513 / 0.5250 | 0.4750 / 0.5000 |
| 31 | token_128 | 0.4662 / 0.5000 | 0.5138 / 0.5000 | 0.4956 / 0.5000 | 0.4669 / 0.4750 |
| 31 | token_512 | Unavailable | Unavailable | Unavailable | Unavailable |
| 31 | think_end | 0.3656 / 0.4625 | 0.6112 / 0.6375 | 0.1881 / 0.4250 | 0.7475 / 0.5625 |
| 31 | decision | 0.3450 / 0.4000 | 0.6844 / 0.6250 | 0.2581 / 0.5125 | 0.8325 / 0.5125 |
| 47 | record | 0.5056 / 0.5000 | 0.4975 / 0.5000 | 0.4888 / 0.5000 | 0.4994 / 0.5000 |
| 47 | prefill | 0.4981 / 0.5000 | 0.5175 / 0.5000 | 0.5044 / 0.4875 | 0.4919 / 0.5000 |
| 47 | token_32 | 0.5750 / 0.5625 | 0.4056 / 0.4375 | 0.5600 / 0.5250 | 0.5012 / 0.5250 |
| 47 | token_128 | 0.4575 / 0.4875 | 0.4919 / 0.4875 | 0.4781 / 0.4625 | 0.4044 / 0.5000 |
| 47 | token_512 | Unavailable | Unavailable | Unavailable | Unavailable |
| 47 | think_end | 0.3762 / 0.4625 | 0.4988 / 0.5125 | 0.1881 / 0.2750 | 0.7300 / 0.5500 |
| 47 | decision | 0.4338 / 0.4625 | 0.6306 / 0.5750 | 0.3381 / 0.2875 | 0.8137 / 0.7625 |
| 57 | record | 0.5212 / 0.5000 | 0.4881 / 0.5000 | 0.4925 / 0.5000 | 0.4631 / 0.5000 |
| 57 | prefill | 0.5194 / 0.5250 | 0.5081 / 0.5250 | 0.5519 / 0.5000 | 0.5006 / 0.5000 |
| 57 | token_32 | 0.6075 / 0.5750 | 0.4863 / 0.5375 | 0.5325 / 0.5750 | 0.4881 / 0.5000 |
| 57 | token_128 | 0.4369 / 0.4500 | 0.4300 / 0.4375 | 0.5256 / 0.5000 | 0.4444 / 0.4875 |
| 57 | token_512 | Unavailable | Unavailable | Unavailable | Unavailable |
| 57 | think_end | 0.3950 / 0.5125 | 0.8281 / 0.5625 | 0.2537 / 0.2625 | 0.9363 / 0.8250 |
| 57 | decision | 1.0000 / 1.0000 | 1.0000 / 0.9875 | 1.0000 / 0.9750 | 1.0000 / 0.7375 |

The overall pattern is broad: record and prefill rankings remain near chance across all four layers; token 32 and token 128 show small, inconsistent deviations rather than stable cross-family/polarity separation. The late positions diverge. Think-end and decision scores in layers 15, 31 and 47 often differ in orientation across question polarity. Layer 57 at the decision position is the distinctive strong common-polarity result. This is an all-site descriptive observation after examining the planned matrix, not a multiplicity-adjusted discovery claim.

“Think end” is the last processed closing-think delimiter selected by the final-answer parser. “Decision” is the activation immediately before the first non-whitespace final token after that delimiter, including EOS. Thus the decision capture precedes the scored final token; it is not an activation after ingesting that answer token. It nevertheless follows generated reasoning and can encode a completed judgment or answer policy. The two late captures should not be conflated with pre-reasoning appraisal.

## The late result is polarity-balanced, but action transfer is weaker

At layer 57/decision, the pooled balanced direction ranks gain perfectly in both families and both question polarities. Its original-threshold accuracy is 80/80 for objects/new, 79/80 for objects/repeat, 78/80 for machines/new, and 59/80 for machines/repeat. Perfect ranking with only 73.75% accuracy in the last group indicates an extraction-threshold calibration mismatch; the present report does not fit a replacement threshold.

These four AUCs describe ordering within each family/question group. They do not by themselves establish a calibrated, universal decision boundary shared across the groups.

At layer 57/think_end, balanced AUC is only 0.3950 for objects/new and 0.25375 for machines/new, but 0.828125 for objects/repeat and 0.93625 for machines/repeat. A result summarized solely as “late AUC is high” would hide this polarity dependence.

The frozen axis controls make the interpretation more specific. At layer 57/decision, pooled:new and pooled:answer each have gain AUC 1 on both new-question families and 0 on both repeat-question families. Pooled:repeat has the converse pattern. Pooled:balanced has AUC 1 on all four groups. The balanced result therefore is not explained by merely reusing the same fixed Yes-versus-No projection: that projection changes its relationship to gain when the question is reversed. It remains compatible with question-conditioned answer computations or a late representation of an already completed inference. The result alone does not establish where or whether that component causally computes information value.

The non-yes/no readouts constrain generalization. For the pooled balanced direction at layer 57/decision:

| Readout | Objects AUC | Machines AUC | Objects threshold accuracy | Machines threshold accuracy |
|---|---:|---:|---:|---:|
| Instrumental action | 0.677500 | 0.888750 | 0.5000 | 0.5000 |
| Memory | 0.441250 | 0.426250 | 0.3750 | 0.4125 |
| Count | 0.525000 | 0.416875 | 0.5000 | 0.5000 |

Instrumental action has partial ranking transfer, especially for machines, but the original yes/no-derived threshold does not classify it usefully. This is weaker evidence for a common instructed-action appraisal signal than the perfect yes/no result. Memory and count outcomes are controls whose correct answers do not vary with the gain label in the same way; the near-chance gain rankings do not measure their behavioral competence. In particular, the relational count answer is always two, so this readout cannot substitute for the planned balanced 1/2/4 competence test.

For completeness, this second site matrix gives **objects AUC / machines AUC** for all remaining kinds. Their full threshold accuracies remain in the linked 3,360-row source analysis.

| Layer | Position | Instrument: objects / machines | Memory: objects / machines | Count: objects / machines |
|---:|---|---:|---:|---:|
| 15 | record | 0.5075 / 0.4369 | 0.5081 / 0.4581 | 0.5162 / 0.4519 |
| 15 | prefill | 0.5100 / 0.4356 | 0.5100 / 0.5619 | 0.5269 / 0.4431 |
| 15 | token_32 | 0.5525 / 0.4625 | 0.5112 / 0.5162 | 0.5000 / 0.5012 |
| 15 | token_128 | 0.3956 / 0.5631 | 0.4630 / 0.6169 | 0.5052 / 0.5125 |
| 15 | token_512 | Unavailable | Unavailable | Unavailable |
| 15 | think_end | 0.2306 / 0.4294 | 0.4394 / 0.5144 | 0.4825 / 0.4688 |
| 15 | decision | 0.2662 / 0.4319 | 0.4913 / 0.5288 | 0.4869 / 0.5162 |
| 31 | record | 0.5019 / 0.5138 | 0.5044 / 0.5350 | 0.5106 / 0.5069 |
| 31 | prefill | 0.4944 / 0.4969 | 0.4975 / 0.4919 | 0.5094 / 0.4956 |
| 31 | token_32 | 0.5737 / 0.4681 | 0.5112 / 0.5144 | 0.5088 / 0.4956 |
| 31 | token_128 | 0.4275 / 0.5219 | 0.4236 / 0.5144 | 0.5069 / 0.4994 |
| 31 | token_512 | Unavailable | Unavailable | Unavailable |
| 31 | think_end | 0.2925 / 0.4875 | 0.4556 / 0.5544 | 0.5119 / 0.4906 |
| 31 | decision | 0.2537 / 0.4225 | 0.4875 / 0.5381 | 0.4487 / 0.4356 |
| 47 | record | 0.4988 / 0.4888 | 0.4825 / 0.5069 | 0.4975 / 0.4994 |
| 47 | prefill | 0.5487 / 0.5319 | 0.5244 / 0.4838 | 0.4850 / 0.4606 |
| 47 | token_32 | 0.5637 / 0.4431 | 0.5481 / 0.5019 | 0.5337 / 0.4544 |
| 47 | token_128 | 0.4419 / 0.4688 | 0.3920 / 0.5413 | 0.4427 / 0.4825 |
| 47 | token_512 | Unavailable | Unavailable | Unavailable |
| 47 | think_end | 0.2419 / 0.4562 | 0.4875 / 0.5312 | 0.5038 / 0.4688 |
| 47 | decision | 0.3769 / 0.4025 | 0.4462 / 0.5606 | 0.4800 / 0.4813 |
| 57 | record | 0.4775 / 0.5050 | 0.4969 / 0.4894 | 0.4881 / 0.4631 |
| 57 | prefill | 0.4963 / 0.5162 | 0.5106 / 0.4800 | 0.4406 / 0.4394 |
| 57 | token_32 | 0.5381 / 0.4562 | 0.5225 / 0.5188 | 0.5231 / 0.5219 |
| 57 | token_128 | 0.5019 / 0.5581 | 0.3750 / 0.5919 | 0.4340 / 0.5744 |
| 57 | token_512 | Unavailable | Unavailable | Unavailable |
| 57 | think_end | 0.3500 / 0.6687 | 0.4981 / 0.5269 | 0.4625 / 0.4363 |
| 57 | decision | 0.6775 / 0.8888 | 0.4412 / 0.4263 | 0.5250 / 0.4169 |

## Other frozen source axes and missing positions remain visible

The following ranges include both target families, both yes/no polarities, and all four layers within each position. They summarize all 12 frozen source axes rather than selecting a favorable source. A range spanning 0 to 1 at a late position can reflect polarity reversal, not uniformly successful transfer. Token 512 is unavailable for every source axis and is omitted from this range table rather than represented as zero.

| Frozen source axis | Record, across four layers | Prefill | Token 32 | Token 128 | Think end | Decision |
|---|---:|---:|---:|---:|---:|---:|
| pooled:new | 0.4637–0.5225 | 0.4850–0.5531 | 0.4119–0.5787 | 0.4300–0.5256 | 0.0094–0.9994 | 0.0000–1.0000 |
| pooled:repeat | 0.4569–0.5194 | 0.4706–0.5594 | 0.4344–0.6175 | 0.3831–0.5594 | 0.0000–1.0000 | 0.0000–1.0000 |
| pooled:balanced | 0.4519–0.5212 | 0.4838–0.5519 | 0.4056–0.6075 | 0.4044–0.5256 | 0.1881–0.9363 | 0.2581–1.0000 |
| pooled:answer | 0.4763–0.5744 | 0.4462–0.5737 | 0.4000–0.5475 | 0.4550–0.5687 | 0.0000–1.0000 | 0.0000–1.0000 |
| objects:new | 0.4594–0.5325 | 0.4450–0.5637 | 0.3606–0.6262 | 0.3781–0.5731 | 0.0031–1.0000 | 0.0000–1.0000 |
| objects:repeat | 0.4387–0.5294 | 0.4206–0.5525 | 0.4406–0.5938 | 0.3900–0.5269 | 0.0000–1.0000 | 0.0000–1.0000 |
| objects:balanced | 0.4387–0.5325 | 0.4394–0.5600 | 0.3894–0.6125 | 0.3956–0.5613 | 0.2456–0.9956 | 0.3688–1.0000 |
| objects:answer | 0.4662–0.5813 | 0.4462–0.6162 | 0.3781–0.5956 | 0.4425–0.5931 | 0.0000–1.0000 | 0.0000–1.0000 |
| machines:new | 0.4894–0.5225 | 0.4825–0.5513 | 0.4350–0.5575 | 0.4562–0.5188 | 0.0306–0.9531 | 0.0000–1.0000 |
| machines:repeat | 0.4781–0.5138 | 0.4744–0.5513 | 0.4363–0.6150 | 0.4200–0.5625 | 0.0138–1.0000 | 0.0000–1.0000 |
| machines:balanced | 0.4844–0.5144 | 0.4781–0.5544 | 0.4675–0.5706 | 0.4519–0.5281 | 0.1787–0.8125 | 0.2269–1.0000 |
| machines:answer | 0.4800–0.5544 | 0.4569–0.5406 | 0.3794–0.5719 | 0.4375–0.5469 | 0.0000–1.0000 | 0.0000–1.0000 |

The complete-world denominators are 20 for every available site except token 128 in object memory (18) and object count (12), consistently across layers and source axes. The four-cell complete-world rule discards a world from a position's metric if any cell lacks that position; it does not discard the world from other positions. These are position-dependent subsets, so a difference across positions can include a change in sampled worlds.

Every token-512 score is unavailable because the corresponding frozen extraction axes are unavailable. That does not show that no relational trajectory reached token 512, and a transfer-only axis is not fitted to fill the gap. Missing scores remain missing. Every new/repeat entry in the main matrix has 20 complete worlds at all the other positions, including the world containing the single invalid answer.

## What follows, and what does not

The narrow supported finding is a dissociation: successful relational behavior coexists with poor early transfer of the original information direction, while a frozen late balanced direction separates inferred gain across opposite yes/no question polarities. This suggests that the earlier task's strong record decoding was not sufficient evidence for an early representation that generalizes to this relational computation. It leaves late integration or completed-decision representations as hypotheses.

This study is development-only. Its `confirmation_eligible` provenance flag records the overlap checks that were performed against the bound extraction/development assignments; it is not a statistical confirmation designation. The current legacy semantic key can generate two table identities for one counterfactual base world, so the 80 listed semantic hashes are not 80 independent experimental units. There was no explicit all-inspected-pilot registry in this descriptive transfer call. Raw index-based summaries should not be presented as semantically deduplicated confirmation. All inspected relational assignments now belong in the future exclusion registry.

No causal intervention, acquisition preference or subjective experience has been established by these results. Report entropy remains confounded with full-world gain here. The separately planned target-relevance task addresses that specific construct confound and has its own frozen eligibility rules. This relational result neither predicts its unseen outcomes nor changes those rules. **Layer 47/record remains the prespecified early candidate; layer 57/decision is not promoted into that slot.**

Condition-major packing may resolve a row-layout numerical artifact, but does not itself repair the separate failed source-capture L2 parity of fixed-transcript replay. A possible later study could intervene directly during native autoregressive generation near answer production, using newly captured same-engine donors and a prospective temporal-alignment and control plan. That would be a new late-stage hypothesis, potentially affecting answer readout rather than information appraisal, and would require its own engineering proof and fresh confirmation. This report implements or authorizes none of it.

## Reproducibility and independent verification

The independent check joined saved trial IDs to the unchanged source manifest, reconstructed the four factorial cells for every world, compared every gain-one score to every gain-zero score to calculate tie-aware AUC, and applied the original extraction thresholds directly. It independently computed mean within-world gain contrasts and concordance. It did not call the production `projection_metrics` function, inspect only correct answers, refit axes, or select a subset of favorable metrics. Scores and frozen analysis artifacts were read only.

The manifest and audit canonical content hashes were checked against the analysis bindings, and the direction file hash matched the recorded extraction artifact. Below are SHA-256 hashes of the file bytes; these differ from canonical JSON content digests used inside manifests.

| Source | File-byte SHA-256 |
|---|---|
| [Transfer analysis](/home/dennis/projects/llm-curiosity-and-satisfaction/analyses/relational-information-v1-development/analysis.json) | `502e8f753ac4463079397cedffad11c61e984698e4eb245e3867fbd8488775a0` |
| [Saved transfer scores](/home/dennis/projects/llm-curiosity-and-satisfaction/analyses/relational-information-v1-development/scores.npz) | `457b21fb1771531e8ddc305880beb4dd5832c4e378b6b7822864fb1eb62384b5` |
| [Behavioral audit](/home/dennis/projects/llm-curiosity-and-satisfaction/reports/qwen32-relational-information-v1-development-audit.json) | `fcfce7384624e0145af2d7f4d01bfe852b6fdad4a714f729c28e2f42ed9fa0c6` |
| [Source manifest](/home/dennis/projects/llm-curiosity-and-satisfaction/runs/qwen32-relational-information-v1-development/manifest.json) | `4c251a1d0c39dd389251d84c6cff8e2ca4bcabc83101ec3256847bc81b61fa11` |
| [Frozen direction array](/home/dennis/projects/llm-curiosity-and-satisfaction/analyses/thinking-state-v1-development/directions.npz) | `4f6541e8967e4a517be0b42b8b2d028e95eb7ba605e7388e3a2b241eca989b60` |

Canonical source-manifest digest: `27a380aac07aa4b9793f6fc6413651c4f32010a95949b47768a69df9ab9f30e2`. The source was collected and analyzed as `development_validation` / `development`. No new inferential threshold is applied retrospectively in this report.
