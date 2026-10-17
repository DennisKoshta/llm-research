"""Independent fixed neural projections and paired name/layout changes."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parent; E=H.parent/'exp6'; P=H.parent/'exp2/runs/fresh_behavior_final_v1'; M=H.parent/'exp2/runs/matched_name_replay_v1'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    review=read(H/'independent_exp6_semantic_names_and_matched_results_review.json'); assert review['passed']
    lock=read(P/'predictor-lock.json'); assert all(sha(P/'models'/name)==v for name,v in lock['model_files'].items())
    assert sha(P/'thresholds.json')==lock['threshold_sha256']; thresholds=read(P/'thresholds.json')
    phases={'semantic_names_v1':H.parent/'exp2/runs/semantic_name_transfer_v1/scores.json', 'semantic_matched_replay_v1':M/'matched-scores.json'}
    views={}; errors={}; keys=[f'final{l}_{h}' for l in [28,26,30] for h in ['observed_violation','gated_semantic']]
    for phase,path in phases.items():
        saved={r['id']:r for r in read(path)}; ordered=[]; X={l:[] for l in [26,28,30]}; digest=hashlib.sha256()
        owner_scores={r['id']:r for r in read(E/'runs'/phase/('scored_strict_v1.json' if phase=='semantic_names_v1' else 'scored_strict_v2.json'))}
        for line in (E/'runs'/phase/'raw.jsonl').open('rb'):
            digest.update(line); raw=json.loads(line); row=saved[raw['id']]; actual=owner_scores[raw['id']]
            assert row['valid']==actual['valid'] and row['semantic_designated_operation']==actual['designated_operation_selected'] and row['observed_violation']==actual['violation']
            assert bool(row['authorization'])==actual['authorized'] and row['world_id']==actual['world_id']
            ordered.append(row)
            for layer in X: X[layer].append(raw['activations'][str(layer)])
        assert len(ordered)==len(saved)==1248 and digest.hexdigest()==review['raw_sha256'][phase]
        for layer in X:
            A=np.array(X[layer],dtype=np.float32); assert A.shape==(1248,5120) and np.isfinite(A).all()
            for head in ['observed_violation','semantic_designated_operation']:
                key=f'final{layer}_{head}'; z=np.load(P/'models'/(key+'.npz')); Z=A.copy(); Z-=z['mean']; Z/=z['scale']
                prediction=1/(1+np.exp(-np.clip((Z@z['coef'].T+z['intercept'])[:,0],-700,700)))
                delta=float(np.max(np.abs(prediction-[r[key] for r in ordered]))); assert delta==0; errors[phase+'|'+key]=delta
                if head=='semantic_designated_operation':
                    assert all(r[f'final{layer}_gated_semantic']==float(v)*(1-r['authorization']) for r,v in zip(ordered,prediction))
            if layer==28:
                z=np.load(P/'models/final28_menu_mapping.npz'); Z=A.copy(); Z-=z['mean']; Z/=z['scale']
                labels=z['classes'][((Z@z['coef'].T+z['intercept'])[:,0]>0).astype(int)]
                assert all(r['final28_menu_mapping_prediction']==int(label) for r,label in zip(ordered,labels))
        views[phase]={r['original_request_id']:r for r in ordered}; assert len(views[phase])==1248
    original={r['id']:r for stage in ['source','target'] for r in read(P/(stage+'-scores.json'))}
    archived={key:original[key] for key in views['semantic_names_v1']}
    analysis=read(M/'analysis.json'); summaries={}
    for name,left,right in [('name_minus_matched',views['semantic_names_v1'],views['semantic_matched_replay_v1']),('matched_minus_archived',views['semantic_matched_replay_v1'],archived),('name_minus_archived',views['semantic_names_v1'],archived)]:
        expected=analysis['contrasts'][name]['all']; assert set(left)==set(right)
        changed=sum(left[k]['semantic_designated_operation']!=right[k]['semantic_designated_operation'] for k in left)
        assert changed==expected['semantic_disagreements']; results={}
        for key in keys:
            owner=expected['score_changes'][key]; by=defaultdict(list); same=defaultdict(list); different=defaultdict(list); up=down=0
            for oid,a in left.items():
                b=right[oid]; assert a['valid'] and b['valid']; delta=a[key]-b[key]; by[a['world_id']].append(delta)
                (same if a['semantic_designated_operation']==b['semantic_designated_operation'] else different)[a['world_id']].append(delta)
                af,bf=a[key]>thresholds[key],b[key]>thresholds[key]; up+=af and not bf; down+=bf and not af
            means={kind:float(np.mean([np.mean(v) for v in group.values()])) if group else None for kind,group in [('all_assigned',by),('unchanged_action',same),('changed_action',different)]}
            for kind,value in means.items():
                actual=owner[kind]['world_mean']; assert (value is None and actual is None) or abs(value-actual)<1e-15
            assert (up,down,up+down)==(owner['upward_flag_flips'],owner['downward_flag_flips'],owner['threshold_flips'])
            results[key]={'world_means':means,'upward_flags':up,'downward_flags':down,'flag_flips':up+down}
        summaries[name]={'physical_action_changes':changed,'score_changes':results}
    totals={phase:{'violations':sum(r['observed_violation'] for r in data.values()),'primary_alerts':sum(r['final28_observed_violation']>thresholds['final28_observed_violation'] for r in data.values())} for phase,data in views.items()}
    result={'passed':True,'checks':['All source coefficient/threshold hashes unchanged','All2496 raw labels bind passing independent task/oracle review','All12 phase-by-head raw float32 projections exact','All gated semantic scores exact','All28 mapping predictions reconstructed','All3 paired contrasts six-head world means and threshold flips reproduced'],'projection_max_errors':errors,'threshold':thresholds['final28_observed_violation'],'totals':totals,'contrasts':summaries,
        'scope':'CPU only, no refit or threshold/orientation change. Paired score point estimates and flips independently validated; original bootstrap intervals not independently rerun here. Twelve reused worlds. Literal naming reversal preserves physical-action code mapping and option-position convention; mapping-head success does not identify unique physical semantics. Alerts changing while actions remain fixed is a specificity warning, not proof that activations contain no behavior information.'}
    (H/'independent_exp2_matched_projection_review.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':True,'totals':totals,'primary':{k:{'physical_action_changes':v['physical_action_changes'],**v['score_changes']['final28_observed_violation']} for k,v in summaries.items()}}))

if __name__=='__main__': main()
