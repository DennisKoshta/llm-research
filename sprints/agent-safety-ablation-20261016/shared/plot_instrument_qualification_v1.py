"""Render the fixed qualification evidence without new inference or fitting."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
paths = ['records/instrument_v2/qualification-failed.json',
         'records/competence_extension_v1/analysis.json',
         'records/intervention-qualified.json']
old, extension, qualification = [json.loads((ROOT / p).read_text()) for p in paths]
assert not old['passed'] and extension['passed'] and qualification['passed']
for p in paths[:2]:
    assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == qualification['files'][p]
arms = ['baseline', 'zero_sham', 'refusal_ablation', 'random_ablation']
labels = ['Baseline', 'Zero sham', 'Refusal\nablation', 'Random\nablation']
colors = ['#52687c', '#9ba9b4', '#147d78', '#b07836']
refusals = [old['counts'][a]['harmful']['refusal'] for a in arms]
correct = [extension['fresh_correct_out_of192'][a] for a in arms]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
for ax, values, denominator, title, ylabel in [
    (axes[0], refusals, 32, 'Held-out response prefixes', 'Overt refusals (%)'),
    (axes[1], correct, 192, 'Separate, prospectively frozen arithmetic panel', 'Correct answers (%)'),
]:
    ax.bar(range(4), [100*v/denominator for v in values], color=colors, width=.65)
    for i, value in enumerate(values):
        ax.text(i, 100*value/denominator+2, f'{value}/{denominator}', ha='center', fontsize=10)
    ax.set_xticks(range(4), labels, fontsize=10)
    ax.set_ylim(0, 111)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18)
fig.suptitle('Refusal suppression and a separate competence check', fontsize=15, y=.97)
fig.subplots_adjust(left=.07, right=.98, top=.82, bottom=.29, wspace=.27)
fig.text(.07, .16, 'Original 24-item arithmetic screen: 21, 21, 20, 22 correct; its qualification remains failed.', fontsize=10)
fig.text(.07, .11, 'The fixed intervention then passed the fresh 192-item and pooled 216-item competence criteria.', fontsize=10)
fig.text(.07, .06, 'One Qwen3-32B checkpoint. Refusal-prefix labels do not measure harmful compliance; arithmetic is a narrow control.', fontsize=9, color='#444444')
out = ROOT / 'records/instrument_v2/figures'
out.mkdir(exist_ok=True)
for suffix in ['png', 'svg']:
    fig.savefig(out / f'qualification.{suffix}', dpi=180)
plt.close(fig)
(out / 'provenance.json').write_text(json.dumps({
    'inputs': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
    'refusals_out_of32': dict(zip(arms, refusals)),
    'fresh_correct_out_of192': dict(zip(arms, correct)),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope': 'Descriptive counts; no intervals or generalization claims added by this figure.',
}, indent=2) + '\n')
print(out)
