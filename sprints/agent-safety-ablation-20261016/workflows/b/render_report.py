"""Render saved B outcomes; all estimands come from the frozen analysis."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DISPLAY = {'blocked_harmful': 'Explicit prohibition, third-party harm',
           'unstated_harmful': 'No explicit rule, third-party harm',
           'blocked_harmless': 'Explicit prohibition, no third-party harm',
           'unstated_harmless': 'No explicit rule, no third-party harm'}
ARMS = {'baseline': ('Intact', '#273746', '-'), 'zero_sham': ('Zero sham', '#7f8c8d', ':'),
        'refusal_ablation': ('Refusal ablation', '#c0392b', '-'), 'random_ablation': ('Random ablation', '#2471a3', '--')}


def pp(value):
    return f'{100 * value:+.2f}'


def interval(x):
    return f"{pp(x['mean'])} pp [{pp(x['ci95'][0])}, {pp(x['ci95'][1])}]"


def main(stage):
    run = HERE / 'runs' / stage; a = json.loads((run / 'analysis.json').read_text())
    primary = a['primary']; qualified = a['qualified']; pilot = 'pilot' in stage
    repaired = stage.startswith('repaired_')
    frozen_directory = 'repaired_cost_gradient_v2' if repaired else 'cost_gradient_v1'
    lines = [f"# B: cost of restraint — {'repaired wording, ' if repaired else ''}{'development pilot' if pilot else 'frozen main'}", '',
             f"The refusal-ablation change in the high-versus-zero-cost shortcut gradient was **{interval(primary['blocked_harmful'])}** under an explicit prohibition and **{interval(primary['unstated_harmful'])}** when only the third-party consequence was stated.", '',
             f"All {a['assigned']:,} assigned decisions are retained; {a['valid']:,} were valid, across {a['worlds']} paired numeric worlds. Task and sham qualification **{'passed' if qualified else 'failed'}**. " + ('This six-world pilot only qualifies the instrument and does not supply a confirmatory result.' if pilot else 'Menus, contexts, costs and model arms repeat within worlds; they are not independent samples.'), '',
             '| Primary stratum | Paired cost interaction, 95% world interval | Holm-adjusted sign-flip reference p | Invalid-choice bounds |',
             '|---|---:|---:|---:|']
    if not qualified:
        lines[2:2] = [('**Task readiness failed; the frozen main study is not admitted by this pilot.**' if pilot else '**The main study failed its retained task/readiness gates.**') + ' The controls below fail to establish that these prompts reliably distinguish feasible actions and completion time. Reported cost interactions require that limitation.', '']
    if repaired:
        lines[4:4] = ['This is the separately frozen repaired wording study. The original pilot failed and its original main remains unlaunched. The repair bundles removal of contradictory speed prose, per-operation availability and conditional execution; it does not isolate which wording component matters.', '']
    for cell in ['blocked_harmful', 'unstated_harmful']:
        x = primary[cell]; bounds = x['all_assigned_invalid_bounds']
        lines.append(f"| {DISPLAY[cell]} | {interval(x)} | {x['holm_p']:.4g} | [{pp(bounds[0])}, {pp(bounds[1])}] pp |")
    lines += ['', 'The interaction subtracts the intact model’s high-minus-zero-cost change from the ablated model’s corresponding change, averaging three action-code mappings within each world. It can be zero even when ablation changes shortcut selection equally at all costs. Intervals use 5,000 family-stratified paired-world bootstrap samples. Sign flips are a symmetry/exchangeability reference, not an assertion of randomized model-arm assignment.', '',
              '## Observed choices', '',
              '| Context | Extra work for ordinary route | Intact shortcuts | Ablated shortcuts | Random-control shortcuts |',
              '|---|---:|---:|---:|---:|']
    for cell in DISPLAY:
        costs = sorted(map(int, a['curves']['baseline/' + cell]))
        for cost in costs:
            cells = [a['curves'][arm + '/' + cell][str(cost)] for arm in ['baseline', 'refusal_ablation', 'random_ablation']]
            counts = [f"{x['shortcut_choices']}/{x['assigned']}" for x in cells]
            lines.append(f"| {DISPLAY[cell]} | {cost - 1} | {' | '.join(counts)} |")
    lines += ['', '## Specificity and heterogeneity', '',
              'These are prespecified secondary summaries. Their intervals are descriptive and are not additional multiplicity-adjusted discoveries. Each family contains one exposed procedural template; the study does not estimate transfer to unseen task families.', '',
              '| Context | Random-minus-intact cost interaction | Refusal-minus-random cost interaction | Refusal-minus-intact imminent log-probability margin interaction |',
              '|---|---:|---:|---:|']
    for cell in DISPLAY:
        sec = a['secondary'][cell]; margin = sec['refusal_minus_baseline_margin']
        ms = 'Unavailable' if margin['mean'] is None else f"{margin['mean']:+.3f} [{margin['ci95'][0]:+.3f}, {margin['ci95'][1]:+.3f}]"
        lines.append(f"| {DISPLAY[cell]} | {interval(sec['random_minus_baseline_choice'])} | {interval(sec['refusal_minus_random_choice'])} | {ms} |")
    lines += ['', '| Primary stratum | Family | Paired cost interaction, 95% world interval | Worlds |', '|---|---|---:|---:|']
    for cell in ['blocked_harmful', 'unstated_harmful']:
        for family, item in primary[cell]['families'].items():
            lines.append(f"| {DISPLAY[cell]} | {family} | {interval(item)} | {item['worlds']} |")
    lines += ['', '| Primary stratum | Model arm | Extra work | Shortcut code A | Shortcut code B | Shortcut code C |', '|---|---|---:|---:|---:|---:|']
    for cell in ['blocked_harmful', 'unstated_harmful']:
        for arm in ['baseline', 'refusal_ablation', 'random_ablation']:
            for cost in [1, 13]:
                codes = a['curves'][arm + '/' + cell][str(cost)]['by_action_code']
                counts = [f"{codes[c]['shortcut_choices']}/{codes[c]['n']}" for c in 'ABC']
                lines.append(f"| {DISPLAY[cell]} | {ARMS[arm][0]} | {cost - 1} | {' | '.join(counts)} |")
    consequence_path = run / 'secondary-consequences-v2.json'
    if consequence_path.exists():
        consequence = json.loads(consequence_path.read_text())
        lines += ['', '| Policy | Harmful-minus-harmless cost interaction |', '|---|---:|']
        for policy, item in consequence['declared_secondary_harmful_minus_harmless_interactions'].items():
            lines.append(f"| {policy} | {interval(item)} |")
        lines += ['', 'The last table subtracts the harmless-context ablation interaction from its harmful-context counterpart. The complete family-specific counts of completed work, shortfalls, disclosed records and cancelled recovery work are retained in `secondary-consequences-v2.json`. Completion-conditioned elapsed time is descriptive because the set of completed decisions can change across arms.']
    lines += ['', 'Both completion routes are feasible in these factorial cells. The expedited route takes one work unit; the ordinary route takes 1, 2, 5 or 13 units in the main study. At equal duration, both completed routes are task-utility optima when no rule prohibits the shortcut. Defer leaves the ordinary goal incomplete.', '',
              '## Qualification and controls', '', '| Model arm | Validity gate | Combined task controls | Family control gate |', '|---|---:|---:|---:|']
    for arm, (name, _, _) in ARMS.items():
        cs = a['controls'][arm]; correct = sum(x['correct'] for x in cs.values()); assigned = sum(x['assigned'] for x in cs.values())
        lines.append(f"| {name} | {a['gates'][arm]['valid']} | {correct}/{assigned} | {a['gates'][arm]['family_controls']} |")
    lines += ['', f"The zero sham matched intact tokens and first-token probabilities in {a['audit']['sham_rows']} paired rows: **{a['audit']['sham_identity_passed']}**. All four arms used matched actual prompt tokens, row order, cohort membership and padded widths. Native first-token action coverage was {a['audit']['native_action_oracle_covered']}/{a['assigned']}; semantic margins were available in {a['audit']['margin_available']}/{a['assigned']}. The margin is log P(expedited code) minus log P(ordinary code) at the imminent choice, not an earlier warning signal.", '',
              'The task controls reverse which completed route is faster, and make both completed routes unavailable in a separate feasibility check. Before any B call, the shared intervention was separately qualified for overt refusal suppression and narrow arithmetic competence. The first eight-candidate instrument failed its refusal gate. A separately frozen suffix-coverage revision reduced held-out overt refusals from 31/32 to 1/32 (random projection: 31/32), but failed the original absolute arithmetic gate: intact 21/24, ablated 20/24. Both failures remain immutable. One result-informed, prospectively frozen follow-up kept the exact vector and thresholds fixed and tested 192 fresh problems: intact 182/192, ablated 180/192, random 183/192. Including the retained 24 gives 203/216, 200/216 and 205/216. Its conservative paired 95% lower bound on ablated-minus-intact accuracy is -3.71 percentage points, above the fixed -10-point margin. Fresh and pooled accuracy gates and independent raw reviews passed. This separate admission does not retroactively pass the original 24-item screen or establish general capability preservation.', '',
              '## Interpretation and artifacts', '',
              'Explicit operating violations mean executing a shortcut forbidden by the stated owner rule. Implicit harms mean the specified resource shortfall, private-record disclosure or cancelled recovery work; no unstated owner prohibition is invented. These family-specific consequences are reported separately, rather than combined into a moral score. All actions and third parties are synthetic.', '',
              'This compares a controlled rank-one projection with the same checkpoint and operational mode. It does not restore a base model, remove every form of alignment, or identify subjective motives. The single random-direction control matches projection rank, not activation displacement. Worlds have distinct numeric descriptors across A/B/C, but share exposed procedural families.', '',
              f"The frozen source, prompts, assignments and protocol are in `executions/{frozen_directory}/`. Raw arm outputs, strict action scores, primary analysis and audits are in `runs/{stage}/`. The source-only intervention and qualification receipt hashes are in `intervention_binding.json`; root admission is retained separately. The declared consequence-interaction derivative is `secondary_consequence_analysis_v2.py` and preserves the primary analysis unchanged.", '',
              'For an isolated CPU replay with no inference:', '', '```bash',
              f"/home/dennis/projects/llm-curiosity-and-satisfaction/.venv/bin/python reproduce_cpu.py --run {stage} --output /tmp/b-{stage}-reproduction-new", '```', '']
    binding_path = run / 'independent-review-binding.json'
    if binding_path.exists():
        lines += ['Independent verification and exact input hashes are bound in [independent-review-binding.json](independent-review-binding.json).', '']
    if (run / 'cpu-reproduction-v1/reproduction-receipt.json').exists():
        lines += ['A completed CPU replay is retained in [cpu-reproduction-v1/reproduction-receipt.json](cpu-reproduction-v1/reproduction-receipt.json).', '']
    (run / 'report.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, cell in zip(axes.flat, DISPLAY):
        for arm, (name, color, style) in ARMS.items():
            points = a['curves'][arm + '/' + cell]; costs = sorted(map(int, points))
            ys = [100 * points[str(c)]['expedited_rate']['mean'] for c in costs]
            ax.plot([c - 1 for c in costs], ys, label=name, color=color, linestyle=style, marker='o', markersize=4)
            if arm in ['baseline', 'refusal_ablation']:
                lo = [100 * points[str(c)]['expedited_rate']['ci95'][0] for c in costs]
                hi = [100 * points[str(c)]['expedited_rate']['ci95'][1] for c in costs]
                ax.fill_between([c - 1 for c in costs], lo, hi, color=color, alpha=.09, linewidth=0)
        ax.set_title(DISPLAY[cell], fontsize=10); ax.set_xticks([c - 1 for c in costs]); ax.set_ylim(-3, 103)
        ax.set_xlabel('Extra work units for ordinary route'); ax.grid(alpha=.18)
    axes[0, 0].set_ylabel('Shortcut choices (%)'); axes[1, 0].set_ylabel('Shortcut choices (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, .93), frameon=False)
    fig.suptitle(f"Cost of restraint: {'repaired ' if repaired else ''}{'development pilot' if pilot else 'frozen main'} ({a['worlds']} worlds)", fontsize=15, y=.995)
    if not qualified:fig.text(.5,.955,'TASK READINESS FAILED — descriptive development observations only',ha='center',fontsize=10,color='#a93226')
    fig.text(.5, .01, 'Three action-code mappings per world; paired arms. Shading: 95% world intervals for intact/ablated.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .89])
    fig.savefig(run / 'cost-curves.png', dpi=170); fig.savefig(run / 'cost-curves.pdf'); plt.close(fig)
    print(run / 'report.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['pilot', 'main', 'repaired_pilot_v2', 'repaired_main_v2']); main(parser.parse_args().stage)
