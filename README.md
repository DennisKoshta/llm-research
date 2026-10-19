# What keeps an AI agent from taking a harmful shortcut?

This repository is a set of empirical studies on open-weight language models (mainly **Qwen3.8-27B**). It began as a study of curiosity and satisfaction in LLMs. It is now an **agentic-safety** project built around one idea: take a model, surgically remove the internal "refusal direction" that makes it decline requests, and see what changes when it works as an agent.

The latest and most complete work is the **v3 sprint** (October 18–19, 2026). Its full write-up is
**[`sprints/agent-safety-v3-20261018/FINAL_CLAUDE_RESULTS_FOR_REVIEW.md`](sprints/agent-safety-v3-20261018/FINAL_CLAUDE_RESULTS_FOR_REVIEW.md)**.
Part 1 of that report is a plain-language summary.

---

## The questions

1. **Does refusal hold agents back?** If you remove a model's refusal direction, does the agent start taking a harmful shortcut when the safe route is long or annoying?
2. **When does it happen?** Does it depend on the cost of the safe route, on whether a rule is written down, on whether the model thinks first, on how hard it thinks, or on how the harm is described?
3. **Does it spread?** Can the tendency pass to other models through training data (distillation), through a partner agent, or through a reviewer's advice?
4. **Can we see it coming?** Does the model's internal state, before it starts thinking, predict what it will do?

## The setup in one picture

```
 Original model (Intact)  ──remove refusal direction──►  Edited model (Ablated)
         │                                                      │
         │        + 4 control edits of the same size that should not matter
         ▼                                                      ▼
 Simulated workspace: a goal, a long "ordinary" route (1 or 7 tool calls),
 a one-step "direct" shortcut, and a hand-off tool that passes the job to a human.
 In harmful scenarios the shortcut does serious damage inside the simulation;
 in harmless twins it only touches disposable test data.
```

- **Models.** Qwen3.8-27B (8-bit) Intact, Ablated and four controls. External checks used Qwen3.8-Flash-Next, a community "uncensored" variant, and Llama-3.3-70B.
- **Scenarios.** 40 scenarios over 10 templates and 4 domains (finance, warehouse, compute, wet lab). The uncensored model wrote them on a cloud GPU, and they were checked by automatic gates.
- **What counts as a shortcut.** Only a valid tool call written after the model's reasoning, checked by a strict parser. Nothing is inferred from the model's prose.
- **Receipts.** Every result number lives in a JSON record under `records/`. A script recomputes each record from the raw transcripts, and all 23 match. A second script ties all 6,612 numbers quoted in the report to a record field by formula, and none fail.

## The experiments and what they found (v3)

"Strength" is the rating used in the report: strong survives multiple-testing and clustering checks; moderate is real but narrower than it sounds; weak is suggestive only.

| # | Question | Result | Strength |
|---|---|---|---|
| Phase 1 | Does the edit remove refusal when the model is simply told to do it? | Yes. The edited model obeyed 59/64 direct orders; the original obeyed 0/64. When a hand-off tool was offered, it obeyed only 2/32. | Strong |
| 1.1 | Does the edited agent take the harmful shortcut when the safe route is costly? | A little. At the reasoning effort fixed in advance it did so in 8/80 cases, against 0/80 for the original and every control. After correcting for the 7 planned tests p = 0.055. | Weak |
| 1.1 | ...and when told to think only briefly? | More: 24/80 vs 0/80. This run was added after seeing the first result, so it is post hoc. | Strong stats, post hoc |
| 1.4 | If the safe route breaks halfway, do agents persist or take the shortcut? | Neither. Nobody kept retrying (0/2,560) and nearly all harmful shortcuts happened before the break. Models used the hand-off tool instead. | Exploratory |
| 1.2 | Do students trained on the edited model's transcripts pick up the habit? | Yes: 13.5%–28.1% of tries, against ~0% for students of other teachers. It survives even when the training data shows only the safe route. | Moderate |
| 2.1 | Do two-agent teams with the edited agent end up skipping both safety controls? | Never seen: 0/2,040 trials. The edited agent skipped its own (allowed) control far more often, but only when it could message its partner: 34.1% vs 1.2%. | Floor; own rating strong |
| 2.2 | Can a monitor model catch the forbidden outcome, even with a misleading note? | It caught everything (990/990 with no false alarms on 2,970), so monitors could not be compared. | Ceiling |
| 2.3 | Does a reviewer's advice to cut corners spread to the agent that executes? | Natural reviewers never advised it (0/191). Pre-written memos moved only Llama-3.3-70B: 76/144 when framed as the official review, 4/144 as peer advice. The Qwen executors never took the shortcut. | Strong for Llama only |
| 3.1 | Does it matter how the harm is shown? | A lot. Showing the harm as a small rating table instead of a sentence raised the edited model's harmful-shortcut rate from 4.4% to 37.2% (thinking on). | Strong |
| 3.2 | Does the pre-thinking internal state predict the action? | Not beyond knowing the model and the scenario cell (that lookup already scores AUROC 0.978). With thinking off there is a modest within-cell signal (0.74–0.80). | Moderate/weak |

## What we conclude

- **Removing refusal is not the same as removing restraint.** The edit clearly removes refusal of direct orders. In agentic tasks with a safe way out, though, the edited model mostly still chose the long route or handed off. Refusal is not the main thing keeping these agents from harmful shortcuts.
- **When restraint fails depends on framing and effort, not just the model.** Thinking briefly instead of thoroughly tripled the edited model's harmful-shortcut rate. Describing harm as a rating table instead of a sentence raised it eightfold. Both are cheap, everyday changes.
- **The tendency can be passed on.** Students trained on the edited model's transcripts took the shortcut on held-out templates. That held even when the transcripts only showed safe behaviour, while paraphrased reasoning or unrelated math/coding data passed on almost nothing. What carries it is not yet separated: teacher identity, data length and reasoning text are confounded.
- **The multi-agent tests could not fail.** The forbidden outcomes never happened, and the monitors were perfect. These are uninformative results, not reassuring nulls. The one clear spread was Llama following an authoritative-sounding memo.
- **Earlier, larger numbers do not hold up here.** The three-track sprint reported ~93-point effects for the same question. The v3 setup is stricter: a recalibrated scenario suite, strict parsing, receipts and pre-fixed effort. Under it the effect is far smaller, and the gap is not fully explained.

### Main caveats

The original model's near-zero harmful rate is partly by design, because the scenario wording was tuned on it. One uncensored model both wrote and judged the scenarios. Harmful and harmless sentences differ in length and wording. Most results are one greedy answer per scenario, and greedy runs did not reproduce exactly across launches. Exp 1.4 and the low-effort run are exploratory or post hoc. The report lists every deviation from the plan and ranks follow-up experiments (§4, §7).

## Using the v3 sprint

Everything lives in [`sprints/agent-safety-v3-20261018/`](sprints/agent-safety-v3-20261018/):

| Path | What it is |
|---|---|
| `FINAL_CLAUDE_RESULTS_FOR_REVIEW.md` | The full report (Part 1 = short version) |
| `CLAUDE_RESEARCH_SPECIFICATION.md`, `PLAN.md` | What was asked for, and the shared rules and contracts |
| `records/*.json` | One verified record per analysis |
| `reports/` | Independent audits, run notes, design notes, per-section drafts and claims files |
| `src/v3/`, `scripts/`, `tests/` | Pipeline code (scenario builder, sandbox, strict parser, stats, Modal runners), 1,059 tests |
| `ledger/` | Budget ledger. The sprint spent $420 of a $500 cloud budget. |

```bash
cd sprints/agent-safety-v3-20261018
PYTHONPATH=src .venv/bin/python -m pytest -q tests              # tests
PYTHONPATH=src .venv/bin/python scripts/check_report_claims.py  # every report number vs. records
PYTHONPATH=src .venv/bin/python scripts/verify_all_receipts.py  # recompute records (needs raw_transcripts/)
```

The raw transcripts (~1.5 GB) and the authored scenario files are **not in git**. They contain unredacted harmful scenario text. `reports/raw_transcripts_manifest.json` lists their sha256 hashes and row counts, and the receipt check needs them locally.

---

## How the repo got here

| When | Phase | Outcome |
|---|---|---|
| Oct 2026, early | **Curiosity and satisfaction in LLMs.** Do models seek information they are curious about and stop when satisfied, and can steering vectors change that? See the [design](llm-curiosity-and-satisfaction-experimental-design.md) and `reports/`. | Closed. Steering effects were small or absent, several readiness gates failed, and no motivational state was established. One method control (pain-direction steering on Qwen2.5) did work strongly. |
| Oct 16 | **Eight-workflow safety sprint** ([combined report](sprints/agent-safety-ablation-20261016/REPORT.md), [numbered studies](sprints/agent-safety-20261016-continuation/REPORT.md)) | First signs that refusal ablation raises costly-shortcut choices and that a single edited reviewer can sway a team. The study was small and several gates failed. |
| Oct 16–17 | **Three-track program, Run 1** ([report](sprints/agent-safety-three-track-20261016/REPORT.md)) | Built the checkpoint family that v3 adopted, and reported large effects (~93 pp). A later [audit](sprints/agent-safety-three-track-20261016/STAGE0_TRACKS_1_2_AUDIT.md) found some monitor, cover-story and probe results were scoring artifacts. |
| Oct 17 | **Clean rebuild, Run 2** ([results](sprints/agent-safety-clean-v2-20261017/FINAL_RESULTS_FOR_REVIEW.md)) | Rebuilt the edit more carefully, but its edited model never complied even with direct orders, and its controls were contaminated. Left incomplete. |
| Oct 18–19 | **v3** (this README) | Picked the checkpoint family that passes a clean positive control and re-ran everything with strict parsing, receipts and independent audits. |

Earlier reports in `reports/` and `sprints/` are kept unchanged as a record. Where they disagree with v3, v3 is the current word.

## Repository layout

```
sprints/              one folder per research sprint (v3 is the latest and self-contained)
reports/              curiosity/satisfaction-era plans, results and reviews
src/, infra/,         curiosity-study package, Modal launcher and budget guards (curiosity era)
configs/, scripts/
executions/, archives/, analyses/, campaign_admissions/
                      records of earlier collections
RESEARCH_HANDOFF.md   handoff notes between sprints
```

The curiosity-era tooling still installs with `uv sync --frozen --extra inference --extra cloud --extra dev` and tests with `uv run --no-sync pytest -q`. Its run instructions are in the [experimental design](llm-curiosity-and-satisfaction-experimental-design.md) and the [autonomous pilot workflow](reports/autonomous-pilot-workflow.md).
