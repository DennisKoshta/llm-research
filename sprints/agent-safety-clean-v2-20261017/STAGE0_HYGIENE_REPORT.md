# Stage 0 hygiene report

Overall status: **FAIL_INCOMPLETE**

Observed raw trial records: 4278.

Software unit tests are not substitutes for empirical H1–H5 gates. Missing experiments, checkpoints, pilots, adapters, or analyses cannot pass.

## Observed raw files

- `raw_transcripts/exp23_author_v2/flash_next_uncensored.jsonl` [archived_engineering]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 5 truncated; 5 invalid parses.
- `raw_transcripts/exp23_author_v2/flash_next_uncensored_requests.jsonl` [archived_engineering]: 0 completed generation trials; 5 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/exp23_author_v3_cap16384/flash_next_uncensored.jsonl` [archived_engineering]: 53 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 1 invalid parses.
- `raw_transcripts/exp23_author_v3_cap16384/flash_next_uncensored_requests.jsonl` [archived_engineering]: 0 completed generation trials; 53 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_activation_full_v2/intact.jsonl` [accepted]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_activation_full_v2/refusal_ablated.jsonl` [accepted]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_activation_v3/intact.jsonl` [archived_engineering]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_activation_v3/refusal_ablated.jsonl` [archived_engineering]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_activation_v4_diagnostic/intact.jsonl` [archived_engineering]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 1 truncated; 1 invalid parses.
- `raw_transcripts/phase1_activation_v4_diagnostic/refusal_ablated.jsonl` [archived_engineering]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_activation_vectors_full_v2/intact.jsonl` [accepted]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_activation_vectors_full_v2/refusal_ablated.jsonl` [accepted]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_activation_vectors_v3/intact.jsonl` [archived_engineering]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_activation_vectors_v3/refusal_ablated.jsonl` [archived_engineering]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_activation_vectors_v4_diagnostic/refusal_ablated.jsonl` [archived_engineering]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_author_v1/flash_next_uncensored.jsonl` [candidate]: 45 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_author_v1/flash_next_uncensored_requests.jsonl` [candidate]: 0 completed generation trials; 45 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_calibration_author_v1_c1/intact_cot_on.jsonl` [archived_engineering]: 11 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 3 invalid parses.
- `raw_transcripts/phase1_calibration_author_v1_c1/intact_cot_on_requests.jsonl` [archived_engineering]: 0 completed generation trials; 11 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_calibration_author_v1_c2/intact_cot_on.jsonl` [candidate]: 691 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 1 truncated; 1 invalid parses.
- `raw_transcripts/phase1_calibration_author_v1_c2/intact_cot_on_requests.jsonl` [candidate]: 0 completed generation trials; 691 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_0_0_cot_on.jsonl` [archived_engineering]: 167 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_0_0_cot_on_requests.jsonl` [archived_engineering]: 0 completed generation trials; 167 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_1_0_cot_on.jsonl` [archived_engineering]: 167 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 1 truncated; 1 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_1_0_cot_on_requests.jsonl` [archived_engineering]: 0 completed generation trials; 167 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_m0_5_cot_on.jsonl` [archived_engineering]: 173 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_m0_5_cot_on_requests.jsonl` [archived_engineering]: 0 completed generation trials; 173 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_p0_5_cot_on.jsonl` [archived_engineering]: 161 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 1 truncated; 3 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_p0_5_cot_on_requests.jsonl` [archived_engineering]: 0 completed generation trials; 161 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_p1_5_cot_on.jsonl` [archived_engineering]: 173 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 2 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_p1_5_cot_on_requests.jsonl` [archived_engineering]: 0 completed generation trials; 173 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_0_0_cot_on.jsonl` [archived_engineering]: 11 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 3 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_0_0_cot_on_requests.jsonl` [archived_engineering]: 0 completed generation trials; 11 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_1_0_cot_on.jsonl` [archived_engineering]: 11 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 3 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_1_0_cot_on_requests.jsonl` [archived_engineering]: 0 completed generation trials; 11 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_m0_5_cot_on.jsonl` [archived_engineering]: 11 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 3 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_m0_5_cot_on_requests.jsonl` [archived_engineering]: 0 completed generation trials; 11 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_p0_5_cot_on.jsonl` [archived_engineering]: 9 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 5 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_p0_5_cot_on_requests.jsonl` [archived_engineering]: 0 completed generation trials; 9 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_p1_5_cot_on.jsonl` [archived_engineering]: 11 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 2 invalid parses.
- `raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_p1_5_cot_on_requests.jsonl` [archived_engineering]: 0 completed generation trials; 11 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_external_v1/flash_next_intact.jsonl` [interrupted]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_external_v3/flash_next_intact.jsonl` [accepted]: 69 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_external_v3/flash_next_uncensored.jsonl` [accepted]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_external_v4_eager/flash_next_uncensored.jsonl` [accepted]: 69 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 1 truncated; 5 invalid parses.
- `raw_transcripts/phase1_external_v4_eager/flash_next_uncensored_requests.jsonl` [accepted]: 0 completed generation trials; 69 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_llama_serving_v1/llama_intact.jsonl` [engineering_pilot]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_llama_serving_v1/llama_intact_requests.jsonl` [engineering_pilot]: 0 completed generation trials; 5 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_logits_full_v2_logits_q1/intact.jsonl` [candidate]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_logits_full_v2_logits_q1/pca_101.jsonl` [candidate]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_logits_full_v2_logits_q1/pca_102.jsonl` [candidate]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_logits_full_v2_logits_q1/pca_103.jsonl` [candidate]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_logits_full_v2_logits_q1/refusal_ablated.jsonl` [archived_engineering]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 1 truncated; 1 invalid parses.
- `raw_transcripts/phase1_logits_full_v2_logits_q1/unrelated.jsonl` [candidate]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_logits_full_v2_logits_q2_cap8192/refusal_ablated.jsonl` [candidate]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_logits_vectors_full_v2_logits_q1/intact.jsonl` [candidate]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_logits_vectors_full_v2_logits_q1/pca_101.jsonl` [candidate]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_logits_vectors_full_v2_logits_q1/pca_102.jsonl` [candidate]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_logits_vectors_full_v2_logits_q1/pca_103.jsonl` [candidate]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_logits_vectors_full_v2_logits_q1/unrelated.jsonl` [candidate]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_logits_vectors_full_v2_logits_q2_cap8192/refusal_ablated.jsonl` [candidate]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_full_v2_q1/intact.jsonl` [archived_engineering]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_full_v2_q1/pca_101.jsonl` [archived_engineering]: 53 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_full_v2_q1/pca_102.jsonl` [archived_engineering]: 17 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_full_v2_q1/refusal_ablated.jsonl` [archived_engineering]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vectors_full_v2_q1/intact.jsonl` [archived_engineering]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vectors_full_v2_q1/pca_101.jsonl` [archived_engineering]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vectors_full_v2_q1/pca_102.jsonl` [archived_engineering]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vectors_full_v2_q1/refusal_ablated.jsonl` [archived_engineering]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 64 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q2/intact.jsonl` [archived_engineering]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 1 truncated; 1 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q2/intact_requests.jsonl` [archived_engineering]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q2/pca_101.jsonl` [archived_engineering]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q2/pca_101_requests.jsonl` [archived_engineering]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q2/pca_102.jsonl` [archived_engineering]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 1 truncated; 1 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q2/pca_102_requests.jsonl` [archived_engineering]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q2/pca_103.jsonl` [archived_engineering]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 1 truncated; 1 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q2/pca_103_requests.jsonl` [archived_engineering]: 0 completed generation trials; 5 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q2/refusal_ablated.jsonl` [archived_engineering]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 2 truncated; 5 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q2/refusal_ablated_requests.jsonl` [archived_engineering]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q2/unrelated.jsonl` [archived_engineering]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 1 truncated; 1 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q2/unrelated_requests.jsonl` [archived_engineering]: 0 completed generation trials; 5 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q3_cap8192/intact.jsonl` [archived_engineering]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q3_cap8192/intact_requests.jsonl` [archived_engineering]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q3_cap8192/pca_101.jsonl` [archived_engineering]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q3_cap8192/pca_101_requests.jsonl` [archived_engineering]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q3_cap8192/pca_102.jsonl` [archived_engineering]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q3_cap8192/pca_102_requests.jsonl` [archived_engineering]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q3_cap8192/pca_103.jsonl` [archived_engineering]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 2 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q3_cap8192/pca_103_requests.jsonl` [archived_engineering]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q3_cap8192/refusal_ablated.jsonl` [archived_engineering]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 3 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q3_cap8192/refusal_ablated_requests.jsonl` [archived_engineering]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q3_cap8192/unrelated.jsonl` [archived_engineering]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 1 truncated; 1 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q3_cap8192/unrelated_requests.jsonl` [archived_engineering]: 0 completed generation trials; 5 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q4_cap16384/intact.jsonl` [accepted]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q4_cap16384/intact_requests.jsonl` [accepted]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q4_cap16384/pca_101.jsonl` [accepted]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q4_cap16384/pca_101_requests.jsonl` [accepted]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q4_cap16384/pca_102.jsonl` [accepted]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q4_cap16384/pca_102_requests.jsonl` [accepted]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q4_cap16384/pca_103.jsonl` [accepted]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 2 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q4_cap16384/pca_103_requests.jsonl` [accepted]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q4_cap16384/refusal_ablated.jsonl` [accepted]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 3 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q4_cap16384/refusal_ablated_requests.jsonl` [accepted]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q4_cap16384/unrelated.jsonl` [accepted]: 133 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_qualification_vllm_q4_cap16384/unrelated_requests.jsonl` [accepted]: 0 completed generation trials; 133 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_serving_unrelated_cap16384_v1/unrelated.jsonl` [engineering_pilot]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_serving_unrelated_cap16384_v1/unrelated_requests.jsonl` [engineering_pilot]: 0 completed generation trials; 5 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_serving_v4/intact.jsonl` [engineering_pilot]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_serving_v4/intact_requests.jsonl` [engineering_pilot]: 0 completed generation trials; 5 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_source_pilot/source_bf16.jsonl` [accepted]: 5 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 0 forward-only observations; 0 truncated; 0 invalid parses.
- `raw_transcripts/phase1_source_vectors/intact_bf16.jsonl` [accepted]: 0 completed generation trials; 0 prompt-journal entries; 0 preserved return batches; 448 forward-only observations; 0 truncated; 0 invalid parses.

## Preserved hygiene findings

- raw_transcripts/exp23_author_v2/flash_next_uncensored.jsonl: H5 pilot requires exactly five episodes with uncapped, closed, valid envelopes in pilot/flash_next_uncensored/cot_on/None [archived_engineering]
- raw_transcripts/phase1_activation_v4_diagnostic/intact.jsonl: H5 pilot requires exactly five episodes with uncapped, closed, valid envelopes in pilot/intact/cot_on/None [archived_engineering]
- raw_transcripts/phase1_calibration_author_v1_c1/intact_cot_on.jsonl: H5 pilot requires exactly five episodes with uncapped, closed, valid envelopes in pilot/intact/cot_on/None [archived_engineering]
- raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_1_0_cot_on.jsonl: H1 truncation rate >=2% in sweep evaluation/dose_1_0/cot_on/None (pilots excluded) [archived_engineering]
- raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_p0_5_cot_on.jsonl: H1 truncation rate >=2% in sweep evaluation/dose_p0_5/cot_on/None (pilots excluded) [archived_engineering]
- raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_0_0_cot_on.jsonl: H5 pilot requires exactly five episodes with uncapped, closed, valid envelopes in pilot/dose_0_0/cot_on/None [archived_engineering]
- raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_1_0_cot_on.jsonl: H5 pilot requires exactly five episodes with uncapped, closed, valid envelopes in pilot/dose_1_0/cot_on/None [archived_engineering]
- raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_m0_5_cot_on.jsonl: H5 pilot requires exactly five episodes with uncapped, closed, valid envelopes in pilot/dose_m0_5/cot_on/None [archived_engineering]
- raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_p0_5_cot_on.jsonl: H5 pilot requires exactly five episodes with uncapped, closed, valid envelopes in pilot/dose_p0_5/cot_on/None [archived_engineering]
- raw_transcripts/phase1_dose_response_author_v1_d4_cap8192_format_v1/dose_p1_5_cot_on.jsonl: H5 pilot requires exactly five episodes with uncapped, closed, valid envelopes in pilot/dose_p1_5/cot_on/None [archived_engineering]
- raw_transcripts/phase1_logits_full_v2_logits_q1/refusal_ablated.jsonl: H5 pilot requires exactly five episodes with uncapped, closed, valid envelopes in pilot/refusal_ablated/cot_on/None [archived_engineering]
- raw_transcripts/phase1_qualification_vllm_q2/pca_103.jsonl: H5 pilot requires exactly five episodes with uncapped, closed, valid envelopes in pilot/pca_103/cot_on/None [archived_engineering]
- raw_transcripts/phase1_qualification_vllm_q2/unrelated.jsonl: H5 pilot requires exactly five episodes with uncapped, closed, valid envelopes in pilot/unrelated/cot_on/None [archived_engineering]
- raw_transcripts/phase1_qualification_vllm_q3_cap8192/unrelated.jsonl: H5 pilot requires exactly five episodes with uncapped, closed, valid envelopes in pilot/unrelated/cot_on/None [archived_engineering]

## Outstanding verification failures

- Missing required records/phase1_receipt.json
- Missing required records/phase1_statistical_analysis.json
- Missing required pilot audit for phase1
- Missing required records/exp_2_3_receipt.json
- Missing required records/exp_2_3_statistical_analysis.json
- Missing required pilot audit for exp_2_3
- Missing required records/exp_1_1_receipt.json
- Missing required records/exp_1_1_statistical_analysis.json
- Missing required pilot audit for exp_1_1
- Missing required records/exp_1_2_receipt.json
- Missing required records/exp_1_2_statistical_analysis.json
- Missing required pilot audit for exp_1_2
- Missing required records/exp_2_2_receipt.json
- Missing required records/exp_2_2_statistical_analysis.json
- Missing required pilot audit for exp_2_2
- Missing required records/exp_2_1_receipt.json
- Missing required records/exp_2_1_statistical_analysis.json
- Missing required pilot audit for exp_2_1
- Missing required records/exp_3_1_receipt.json
- Missing required records/exp_3_1_statistical_analysis.json
- Missing required pilot audit for exp_3_1
- Missing required records/exp_3_2_receipt.json
- Missing required records/exp_3_2_statistical_analysis.json
- Missing required pilot audit for exp_3_2

## Independent numerical receipt checks

- `records/phase1_author_v1_receipt.json`: PASS
- `records/phase1_calibration_author_v1_c2_receipt.json`: PASS
- `records/phase1_dose_response_author_v1_d2_descriptive_receipt.json`: PASS
- `records/phase1_external_qualification_v3_v4_eager_receipt.json`: PASS
- `records/phase1_initial_post_fp8_gate_receipt.json`: PASS
- `records/phase1_post_fp8_gate_receipt.json`: PASS
- `records/phase1_primary_qualification_vllm_q4_cap16384_receipt.json`: PASS
- `records/phase1_source_directions_receipt.json`: PASS

All present numerical receipts recomputed: **True**.
Full research verification: **FAIL_INCOMPLETE**.
Partial receipt verification is not a claim that all phases or gates are complete.
