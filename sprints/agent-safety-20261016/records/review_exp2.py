"""Independent root reconstruction of central threshold/transfer metrics."""
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score
import numpy as np
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'workflows/exp2'
a=json.loads((W/'probe_analysis.json').read_text());groups=json.loads((W/'group_scores.json').read_text())
core=[g for g in groups if g['corpus']=='core' and g['phase']=='final' and g['split']=='test']
cal=[g for g in groups if g['corpus']=='core' and g['phase']=='final' and g['split']=='calibration' and g['condition']=='control']
threshold=float(np.quantile([g['peak'] for g in cal],.95))
controls=[g for g in core if g['condition']=='control']
fpr=sum(g['peak']>threshold for g in controls)
auc=roc_auc_score([g['configuration_positive'] for g in core],[g['peak'] for g in core])
assert len(cal)==6 and len(core)==36 and len(controls)==18 and fpr==9 and auc==1
fresh=[g for g in groups if g['corpus']=='specificity' and g['split']=='test']
assert len(fresh)==120
counts={c:sum(g['peak']>threshold for g in fresh if g['condition']==c) for c in {g['condition'] for g in fresh}}
assert all(v==0 for v in counts.values())
transfer=[json.loads(x) for x in (W/'transfer_exp1_scores.jsonl').read_text().splitlines()]
print('transfer row keys',list(transfer[0]))
result={'reviewer':'root','passed':True,'source_calibration_groups':len(cal),'source_test_groups':len(core),'threshold':threshold,'source_test_control_false_alarms':fpr,'source_test_control_n':18,'source_configuration_auc':float(auc),'specificity_flags':counts,'scope':'Independent threshold and group metric calculation from saved scores; owner standalone audit separately reconstructs projections and actual actions from raw rows. Review does not treat role labels as behavior or assert 5% FPR calibration.'}
(W/'independent-result-review-root.json').write_text(json.dumps(result,indent=2));print(result)
