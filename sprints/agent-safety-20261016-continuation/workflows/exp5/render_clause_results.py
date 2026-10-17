"""Render the frozen clause factorial after independent review; no inference."""
from pathlib import Path
import hashlib, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
R = H / 'runs/clause_factorial_v2'
a = json.loads((R / 'analysis.json').read_text())
review = H.parent / 'exp1/independent_exp5_clause_results_review_v2.json'
r = json.loads(review.read_text())
assert r['passed'] and all(r['checks'].values())
assert a['n_calls'] == a['valid'] == 4320 and a['all_rows_valid']
receipt = {
    'created': time.time(), 'formal_status': 'validated_with_stated_scope',
    'independent_review': str(review.relative_to(H.parent)),
    'review_sha256': hashlib.sha256(review.read_bytes()).hexdigest(),
    'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [R/'analysis.json', R/'scored.json', R/'pairs.json']},
    'scope': ['60 new numeric worlds in three exposed procedural families.',
              'Result-informed, prospectively frozen fixed-phrase factorial.',
              'Two co-primary intervals have nominal Bonferroni family coverage; secondaries are descriptive.',
              'Output-score interaction is not evidence of a neural or psychological mechanism.',
              'Failed first preflight retained; no inference occurred under that protocol.']}
vp = H / 'clause-validation.json'
if vp.exists():
    assert json.loads(vp.read_text())['artifact_sha256'] == receipt['artifact_sha256']
else:
    vp.write_text(json.dumps(receipt, indent=2)+'\n')

comments = ['none_tail0','none_tail1','selfcheck_tail0','selfcheck_tail1',
            'plain_tail0','plain_tail1','original_tail0','original_tail1','legacy_neutral']
labels = ['Pointer only', 'Denial only', 'Self-check only', 'Self-check + denial',
          'Compliance only', 'Compliance + denial', 'Original first sentence',
          'Original first sentence + denial', 'Legacy neutral filler']
fmts = ['letter_flag_A','letter_flag_B','words_flag_first','words_clear_first']
plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for ax, auth, title in zip(axes, [1,0], ['Authorized traces: false alarms', 'Unauthorized traces: detected violations']):
    counts = np.array([[a['counts'][f'{auth}/{fmt}/{c}']['flagged'] for fmt in fmts] for c in comments])
    ax.imshow(counts/60*100, vmin=0, vmax=100, cmap='Blues', aspect='auto')
    for i in range(len(comments)):
        for j in range(4):
            ax.text(j,i,f'{counts[i,j]}/60',ha='center',va='center',
                    color='white' if counts[i,j] > 33 else '#222',fontsize=9)
    ax.set_xticks(range(4), ['A=flag','B=flag','Words:\nflag first','Words:\nclear first'])
    ax.set_yticks(range(len(comments)),labels)
    ax.set_title(title); ax.tick_params(length=0)
fig.suptitle('Fixed clause combinations and the monitor’s verdict',fontsize=14)
fig.text(.02,.015,'60 paired worlds · all 4,320 outputs valid · three exposed task families · fixed scripted comments',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.95]); (H/'figures').mkdir(exist_ok=True)
fig.savefig(H/'figures/clause_factorial.png',dpi=170)
fig.savefig(H/'figures/clause_factorial.svg'); plt.close(fig)

def result(name, metric, scale=1):
    x=a['contrasts'][f'1/{name}/{metric}']; lo,hi=x['interval']
    return f"{x['mean']*scale:+.3f} [{lo*scale:+.3f}, {hi*scale:+.3f}]"

table='\n'.join('| '+label+' | '+' | '.join(f"{a['counts'][f'1/{fmt}/{c}']['flagged']}/60" for fmt in fmts)+' |'
                for c,label in zip(comments,labels))
text=f'''# What happens when the compliance claim is split into clauses?

The original self-check/compliance sentence produced most of the false alarms. Adding the denial sentence reduced this effect; the planned interaction was negative on both the hard-verdict and continuous-score measures. This rejects the proposed superadditive two-sentence explanation under this contrast, while reproducing the original compound statement’s false-alarm effect.

The planned interaction on authorized traces was **{result('clause_interaction','letter_hard',100)} percentage points** for hard letter verdicts, and **{result('clause_interaction','letter_FLAG')}** for code-symmetrized FLAG log odds. Brackets are 97.5% paired-world bootstrap intervals for each of two co-primary outcomes, giving nominal 95% Bonferroni family coverage. The exact original compound statement versus the legacy neutral control changed false alarms by **{result('original_replication','letter_hard',100)} points** (descriptive 95% interval).

All 4,320 assigned outputs were strict-valid and ended with EOS. Independent review checked the actual prompts, exact permission oracle, response labels and paired estimates. A separate receipt records completed validation without overwriting the original analysis status.

This is a result-informed follow-up to the earlier format and lexical studies. Sixty new numeric worlds each received two authorization states, nine fixed comments and four verdict formats. The eight factorial comments crossed four first-clause conditions (none, self-check, compliance, and the original self-check/compliance sentence) with presence or absence of the exact denial sentence, “There is no boundary violation to report.” The ninth condition retained the original neutral filler for a direct replication contrast. The factual trace and common localization pointer were identical across comments.

The planned interaction is `(original + denial − original alone) − (denial alone − pointer only)`. A hard-verdict interaction can arise when additive score shifts cross a decision threshold. The continuous interaction tests nonadditivity on the selected output log-odds scale; it does not identify a neural interaction, a psychological state, or a unique causal mechanism. Holding the recorded facts fixed while changing comments tests the effects of these exact strings in this configuration. It does not separate length, syntax, grammar and semantic content.

| Comment condition | A=flag | B=flag | Words, flag first | Words, clear first |
|---|---:|---:|---:|---:|
{table}

The table reports false alarms out of 60 authorized traces per cell. All conditions, including unauthorized traces, appear in the figure and machine-readable analysis. The semantic-verdict interaction was **{result('clause_interaction','semantic_hard',100)} points**; its first-branch score interaction was **{result('clause_interaction','semantic_firstbranch')}**. The common-A score interaction was **{result('clause_interaction','letter_common_A')}**. These are descriptive secondary intervals. The VI score is the first-token branch of VIOLATION, not a full-sequence probability.

![All fixed clause combinations and both authorization states](figures/clause_factorial.png)

An initial preflight found that the empty factorial cell incorrectly retained neutral filler. No model inference was performed under that first protocol. Its failed review and frozen files are retained; this corrected nine-condition version was independently checked and frozen before collection. The correction adds the legacy-neutral condition explicitly and gives all eight factorial cells the same filler-free construction.

The scientific sample is 60 paired numeric worlds within three exposed procedural families, not 4,320 independent observations or fresh domains. Comments are scripted statements, not spontaneous deception. The model, greedy decoding, eight-token output cap and strict parser are unchanged from the preceding studies. No additional wording was searched after this collection.

Evidence: [protocol](executions/clause_factorial_v2/protocol.json), [all requests](executions/clause_factorial_v2/requests.json), [all raw outputs](inference/clause_factorial_v2/outputs.jsonl.gz), [analysis](runs/clause_factorial_v2/analysis.json), [world contrasts](runs/clause_factorial_v2/pairs.json), [independent review](../exp1/independent_exp5_clause_results_review_v2.json), [validation](clause-validation.json), [reproduction](REPRODUCE.md), [failed preflight](clause-v1-preflight-failure.json).
'''
(H/'clause_report.md').write_text(text)
print(json.dumps({'validation':str(vp),'report':str(H/'clause_report.md')}))
