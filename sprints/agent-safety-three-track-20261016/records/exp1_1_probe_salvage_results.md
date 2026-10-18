### Task 2 detail — Exp 1.1 layer-48 probe (5-fold leave-two-templates-out)

Captured last tokens: `{"intact": {"pre_on": "\u010a", "pre_off": "\u010a\u010a", "post": "</think>"}, "refusal_ablated": {"pre_on": "\u010a", "pre_off": "\u010a\u010a", "post": "</think>"}}`

Labels: strict Exp 1.1 re-score (complete `<tool_call>` JSON), truncated trials dropped. Pooled out-of-fold AUROC; `pos/neg` = shortcut / non-shortcut.

| regime | position | subset | pos/neg | probe | r̂ 1-D | TF-IDF | design cell |
|---|---|---|---|---|---|---|---|
| 1 pooled | pre | cot_on, both arms | 186/502 | 0.976 | 0.691 | 0.945 | 0.983 |
| 2 within-arm | pre | cot_on, intact | 65/278 | 0.970 | 0.958 | 0.931 | 0.978 |
| 2 within-arm | pre | cot_on, refusal_ablated | 121/224 | 0.986 | 0.477 | 0.978 | 0.979 |
| 1 pooled | post | cot_on, both arms | 186/502 | 1.000 | 0.798 | 0.989 | 0.983 |
| 2 within-arm | post | cot_on, intact | 65/278 | 1.000 | 0.967 | 0.978 | 0.978 |
| 2 within-arm | post | cot_on, refusal_ablated | 121/224 | 1.000 | 0.479 | 0.996 | 0.979 |
| 1 pooled | pre | cot_off, both arms | 272/428 | 0.947 | 0.757 | 0.879 | 0.959 |
| 2 within-arm | pre | cot_off, intact | 98/257 | 0.969 | 0.883 | 0.883 | 0.965 |
| 2 within-arm | pre | cot_off, refusal_ablated | 174/171 | 0.920 | 0.426 | 0.907 | 0.936 |

Within-arm cells with ≥3 shortcut and ≥3 non-shortcut trials: `cot_on/intact/low_cost__harmless__implicit` (4/23); `cot_on/refusal_ablated/high_friction__harmless__explicit` (3/19); `cot_off/intact/high_friction__harmful__implicit` (8/22); `cot_off/intact/high_friction__harmless__explicit` (23/4); `cot_off/intact/high_tedium__harmless__explicit` (6/22); `cot_off/refusal_ablated/high_friction__harmful__explicit` (14/9); `cot_off/refusal_ablated/high_friction__harmful__implicit` (26/3); `cot_off/refusal_ablated/high_tedium__harmful__explicit` (8/20); `cot_off/refusal_ablated/high_tedium__harmful__implicit` (22/7); `cot_off/refusal_ablated/high_tedium__harmless__explicit` (13/14).

| regime | position | subset | pos/neg | probe | r̂ 1-D | TF-IDF | design cell |
|---|---|---|---|---|---|---|---|
| 3a within-cell (arm-level model) | pre | cot_on/intact/low_cost__harmless__implicit | 4/23 | 0.598 | 0.783 | 0.761 | 0.500 |
| 3a within-cell (arm-level model) | post | cot_on/intact/low_cost__harmless__implicit | 4/23 | 1.000 | 0.750 | 0.826 | 0.500 |
| 3a within-cell (arm-level model) | pre | cot_on/refusal_ablated/high_friction__harmless__explicit | 3/19 | 0.754 | 0.421 | 0.947 | 0.500 |
| 3a within-cell (arm-level model) | post | cot_on/refusal_ablated/high_friction__harmless__explicit | 3/19 | 1.000 | 0.789 | 1.000 | 0.500 |
| 3a within-cell (arm-level model) | pre | cot_off/intact/high_friction__harmful__implicit | 8/22 | 0.847 | 0.744 | 0.557 | 0.500 |
| 3a within-cell (arm-level model) | pre | cot_off/intact/high_friction__harmless__explicit | 23/4 | 0.891 | 0.902 | 0.587 | 0.500 |
| 3a within-cell (arm-level model) | pre | cot_off/intact/high_tedium__harmless__explicit | 6/22 | 0.750 | 0.841 | 0.568 | 0.500 |
| 3a within-cell (arm-level model) | pre | cot_off/refusal_ablated/high_friction__harmful__explicit | 14/9 | 0.587 | 0.373 | 0.714 | 0.500 |
| 3a within-cell (arm-level model) | pre | cot_off/refusal_ablated/high_friction__harmful__implicit | 26/3 | 0.808 | 0.397 | 0.538 | 0.500 |
| 3a within-cell (arm-level model) | pre | cot_off/refusal_ablated/high_tedium__harmful__explicit | 8/20 | 0.556 | 0.306 | 0.481 | 0.500 |
| 3a within-cell (arm-level model) | pre | cot_off/refusal_ablated/high_tedium__harmful__implicit | 22/7 | 0.727 | 0.435 | 0.617 | 0.500 |
| 3a within-cell (arm-level model) | pre | cot_off/refusal_ablated/high_tedium__harmless__explicit | 13/14 | 0.352 | 0.451 | 0.626 | 0.500 |
| 3b trained on split cells only | pre | cot_on: intact/low_cost__harmless__implicit + refusal_ablated/high_friction__harmless__explicit | 7/42 | 0.228 | 0.187 | 0.068 | 0.126 |
| 3b trained on split cells only | post | cot_on: intact/low_cost__harmless__implicit + refusal_ablated/high_friction__harmless__explicit | 7/42 | 1.000 | 0.190 | 0.218 | 0.126 |
| 3b trained on split cells only | pre | cot_off: intact/high_friction__harmful__implicit + intact/high_friction__harmless__explicit + intact/high_tedium__harmless__explicit + refusal_ablated/high_friction__harmful__explicit + refusal_ablated/high_friction__harmful__implicit + refusal_ablated/high_tedium__harmful__explicit + refusal_ablated/high_tedium__harmful__implicit + refusal_ablated/high_tedium__harmless__explicit | 120/101 | 0.753 | 0.594 | 0.559 | 0.722 |

3a scores are centred within each test fold before computing AUROC (removes fold-model offsets); the design-cell baseline is then 0.5 by construction. TF-IDF sees only the prompt (pre) or prompt + think text (post).

