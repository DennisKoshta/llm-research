"""Prospective grouped CPU probe engine; public execution is synthetic-only.

Empirical binding/admission awaits the independently audited Exp3.1 capture
backend. No GPU client or activation extractor is present in this module.
"""
from collections import Counter
from itertools import product
import hashlib
import importlib.metadata
import warnings
import numpy as np
from exp32_plan import SEED, FOLDS, BOOTSTRAPS, PINS, FACTORS, LEVELS, CATEGORIES, digest, require, plan

METRICS = ('auroc','fpr_at_80_tpr','ppv_at_10pct_prevalence')


def dependencies():
    found={name:importlib.metadata.version(name) for name in PINS}
    require(found==PINS,'Frozen Exp3.2 numerical package versions differ')
    return found


def cell(row): return tuple(row['factors'][name] for name in FACTORS)


def stratum(row): return (row['experiment'],row['cot_mode'],row['payoff_format'])


def fold_map(rows):
    membership={}; result={}
    for row in rows:
        key=(row['experiment'],row['scenario_id']); value=row['split_group']
        require(key not in membership or membership[key]==value,'Scenario leaks across split groups')
        require(row['template_id']==value,'Template must be the indivisible split group')
        membership[key]=value
    for experiment in sorted({r['experiment'] for r in rows}):
        groups={r['split_group'] for r in rows if r['experiment']==experiment}
        require(len(groups)>=FOLDS,'At least five template groups required')
        ordered=sorted(groups,key=lambda g:(digest([SEED,experiment,g]),g))
        result.update({(experiment,g):i%FOLDS for i,g in enumerate(ordered)})
    return result


def interaction_terms(experiment):
    levels={**LEVELS,'arm':LEVELS['arm'] if experiment=='exp11' else LEVELS['arm'][:2]}
    choices=[[()] + [((name,level),) for level in levels[name][1:]] for name in FACTORS]
    # Empty product is supplied as the classifier's unpenalized intercept.
    return [sum(terms,()) for terms in product(*choices) if any(terms)]


def interaction_matrix(rows,experiment):
    terms=interaction_terms(experiment)
    return np.asarray([[float(all(row['factors'][key]==value for key,value in term)) for term in terms]
                       for row in rows],dtype=np.float64).reshape(len(rows),len(terms))


def eligible_cells(train):
    grouped={}
    for row in train: grouped.setdefault(cell(row),[]).append(row)
    report={}
    for key,rows in sorted(grouped.items()):
        k=sum(r['outcome_category']=='SHORTCUT_VIOLATION' for r in rows); n=len(rows)
        clusters=len({r['split_group'] for r in rows})
        report[key]={'k':k,'n':n,'train_clusters':clusters,
                     'eligible':clusters>=2 and 0<k<n and .2<=k/n<=.8}
    return report


def metric_values(y,scores):
    from sklearn.metrics import roc_auc_score,roc_curve
    y=np.asarray(y,dtype=int); scores=np.asarray(scores,dtype=float)
    require(y.shape==scores.shape and np.isfinite(scores).all() and np.isin(y,[0,1]).all(),'Invalid held-out metrics input')
    if len(y)==0 or len(set(y))<2:
        return {**{key:None for key in METRICS},'undefined_reason':'No held-out observations' if len(y)==0 else 'Held-out labels contain only one class'}
    fpr,tpr,_=roc_curve(y,scores,drop_intermediate=False)
    index=int(np.flatnonzero(tpr>=.8)[0])
    f=float(fpr[index]) if tpr[index]==.8 else float(fpr[index-1]+(.8-tpr[index-1])*(fpr[index]-fpr[index-1])/(tpr[index]-tpr[index-1]))
    return {'auroc':float(roc_auc_score(y,scores)),'fpr_at_80_tpr':f,
            'ppv_at_10pct_prevalence':float(.08/(.08+.9*f)),'undefined_reason':None}


def bootstrap(rows,method,*,comparator=None,replicates=BOOTSTRAPS,seed=SEED):
    selected=[r for r in rows if r['scores'].get(method) is not None and
              (comparator is None or r['scores'].get(comparator) is not None)]
    def values(indices):
        chosen=[selected[i] for i in indices]; y=[r['y'] for r in chosen]
        result=metric_values(y,[r['scores'][method] for r in chosen])
        if comparator is None:return result
        other=metric_values(y,[r['scores'][comparator] for r in chosen])
        return {**{key:None if result[key] is None else result[key]-other[key] for key in METRICS},
                'undefined_reason':result['undefined_reason']}
    point=values(range(len(selected)))
    groups=sorted({r['split_group'] for r in selected})
    indices={g:[i for i,r in enumerate(selected) if r['split_group']==g] for g in groups}
    distributions={key:[] for key in METRICS}; valid=0
    if point['undefined_reason'] is None and len(groups)>=2:
        rng=np.random.default_rng(seed)
        for _ in range(replicates):
            chosen=rng.choice(groups,size=len(groups),replace=True)
            value=values([i for g in chosen for i in indices[g]])
            if value['undefined_reason'] is None:
                valid+=1
                for key in METRICS:distributions[key].append(value[key])
    adequate=valid>=.8*replicates and len(groups)>=2
    intervals={key:np.quantile(v,[.025,.975],method='linear').tolist() if adequate else None for key,v in distributions.items()}
    return {'method':method,'comparator':comparator,'point':point,'interval95':intervals,
            'target_n':len(rows),'predicted_common_n':len(selected),'missing_prediction_n':len(rows)-len(selected),
            'clusters':len(groups),'bootstrap_replicates':replicates,'valid_replicates':valid,
            'undefined_bootstrap_replicates':replicates-valid,'ci_available':adequate,
            'ci_unavailable_reason':None if adequate else ('UNDEFINED_POINT_METRIC' if point['undefined_reason'] is not None else
                'FEWER_THAN_TWO_TEMPLATE_CLUSTERS' if len(groups)<2 else 'FEWER_THAN_80_PERCENT_VALID_TWO_CLASS_REPLICATES'),
            'degenerate_interval':{k:bool(v is not None and v[0]==v[1]) for k,v in intervals.items()},
            'ci_scope':'Cluster resampling conditional on fitted OOF models and training-only selection; pointwise, not multiplicity-adjusted',
            'coverage_state':'COMPLETE_PREDICTIONS' if len(selected)==len(rows) else 'PARTIAL_OR_UNAVAILABLE_PREDICTIONS'}


def fit_predict(dataset,train,test,method):
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    require(not {r['split_group'] for r in train}&{r['split_group'] for r in test},'Train/test template leakage')
    info={'method':method,'train_trial_ids':[r['trial_id'] for r in train],'test_trial_ids':[r['trial_id'] for r in test],
          'train_cell_count':len({cell(r) for r in train}),
          'test_unseen_cell_n':sum(cell(r) not in {cell(t) for t in train} for r in test)}
    if method=='constant_design':return [0.5]*len(test),{**info,'status':'FIXED_CONSTANT_SCORE'}
    y=np.asarray([r['outcome_category']=='SHORTCUT_VIOLATION' for r in train],dtype=int)
    if len(set(y))<2:return [None]*len(test),{**info,'status':'UNAVAILABLE_ONE_CLASS_TRAINING'}
    scaler=None
    if method=='design_interaction':
        xt=interaction_matrix(train,train[0]['experiment']); xv=interaction_matrix(test,train[0]['experiment'])
        info['interaction_terms_sha256']=digest(interaction_terms(train[0]['experiment']))
    elif method=='prompt_tfidf':
        vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=20000,sublinear_tf=True,lowercase=True,norm='l2')
        try:xt=vectorizer.fit_transform([r['available_prompt_text'] for r in train])
        except ValueError as exc:
            if 'empty vocabulary' not in str(exc):raise
            return [None]*len(test),{**info,'status':'UNAVAILABLE_EMPTY_VOCABULARY'}
        xv=vectorizer.transform([r['available_prompt_text'] for r in test])
        info['vocabulary']={k:int(v) for k,v in sorted(vectorizer.vocabulary_.items())}
        info['idf_sha256']=hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest()
    else:
        x=dataset['activations']; rhat=dataset['r_hat']
        xt=x[[r['activation_index'] for r in train]]; xv=x[[r['activation_index'] for r in test]]
        if method=='intact_rhat':
            require(all(r['factors']['arm']=='intact' for r in train+test),'1D refusal baseline is Intact-only')
            xt=(xt@rhat)[:,None]; xv=(xv@rhat)[:,None]
        elif method=='ablated_orthogonal':
            require(all(r['factors']['arm']=='refusal_ablated' for r in train+test),'Orthogonal test is Ablated-only')
            xt=xt-np.outer(xt@rhat,rhat)/float(rhat@rhat)
            xv=xv-np.outer(xv@rhat,rhat)/float(rhat@rhat)
        else:require(method=='full_activation','Unknown probe method')
        scaler=StandardScaler();xt=scaler.fit_transform(xt);xv=scaler.transform(xv)
        info['train_scaler_mean_sha256']=hashlib.sha256(scaler.mean_.tobytes()).hexdigest()
        info['train_scaler_scale_sha256']=hashlib.sha256(scaler.scale_.tobytes()).hexdigest()
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        model=LogisticRegression(C=1.,l1_ratio=0.,solver='lbfgs',max_iter=2000,tol=1e-8,
                                 class_weight=None,random_state=SEED).fit(xt,y)
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):
            return [None]*len(test),{**info,'status':'UNAVAILABLE_NONCONVERGENCE'}
        scores=model.predict_proba(xv)[:,1]
    require(np.isfinite(scores).all(),'Nonfinite held-out predictions')
    info.update(status='FITTED',coefficient_sha256=hashlib.sha256(model.coef_.tobytes()).hexdigest(),
                intercept_sha256=hashlib.sha256(model.intercept_.tobytes()).hexdigest(),iterations=model.n_iter_.tolist())
    if method=='ablated_orthogonal':
        effective=model.coef_[0]/scaler.scale_; effective-=rhat*(effective@rhat)/(rhat@rhat)
        dot=float(effective@rhat); require(abs(dot)<=1e-8*(1+np.linalg.norm(effective)),'Learned direction is not orthogonal')
        info['effective_direction_dot_rhat']=dot
    return scores.tolist(),info


def analyze_synthetic(dataset,*,bootstrap_replicates=BOOTSTRAPS):
    from exp32_contract import dataset_digest
    require(dataset['synthetic'] is True and all(r['trial_id'].startswith('SYNTHETIC:') for r in dataset['rows']),
            'Preparation execution accepts synthetic data only; empirical capture admission not implemented')
    require(type(bootstrap_replicates) is int and bootstrap_replicates>0,'Positive bootstrap count required')
    require(dataset['analysis_data_sha256']==dataset_digest(dataset),'Joined analysis rows or activation bytes changed')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _analyze(dataset,bootstrap_replicates=bootstrap_replicates)


def _analyze(dataset,*,bootstrap_replicates=BOOTSTRAPS):
    dependencies(); all_rows=sorted(dataset['rows'],key=lambda r:r['trial_id']); folds=fold_map(all_rows)
    selected=[r for r in all_rows if (r['experiment']=='exp11' and r['design_family']=='main') or r['experiment']=='exp31']
    summaries=[]; fits=[]; decisions=[]; predictions=[]
    for cohort in sorted({stratum(r) for r in selected}):
        base=[r for r in selected if stratum(r)==cohort]
        for denominator in ('itt','strict'):
            rows=base if denominator=='itt' else [r for r in base if r['outcome_category']!='TRUNCATED_OR_MALFORMED']
            for regime in ('A','B','C'):
                scoped=rows if regime=='A' else [r for r in rows if r['factors']['arm'] in (('intact',) if regime=='B' else LEVELS['arm'][:2])]
                batches={}
                for fold in range(FOLDS):
                    train=[r for r in scoped if folds[(r['experiment'],r['split_group'])]!=fold]
                    test=[r for r in scoped if folds[(r['experiment'],r['split_group'])]==fold]
                    if not test:continue
                    if regime=='C':
                        eligible=eligible_cells(train)
                        for key in sorted({cell(r) for r in scoped}):
                            entry=eligible.get(key,{'k':0,'n':0,'train_clusters':0,'eligible':False})
                            decisions.append({'cohort':list(cohort),'denominator':denominator,'fold':fold,'cell':list(key),**entry,
                                              'held_out_rows':sum(cell(r)==key for r in test),
                                              'held_out_trial_ids':[r['trial_id'] for r in test if cell(r)==key],
                                              'training_trial_ids':[r['trial_id'] for r in train if cell(r)==key],
                                              'held_out_labels_used_for_selection':False})
                        cells=[k for k,v in eligible.items() if v['eligible'] and any(cell(r)==k for r in test)]
                    else:cells=[None]
                    for key in cells:
                        t=train if key is None else [r for r in train if cell(r)==key]
                        v=test if key is None else [r for r in test if cell(r)==key]
                        methods=['full_activation','design_interaction','prompt_tfidf'] if regime=='A' else (
                            ['intact_rhat','full_activation','design_interaction','prompt_tfidf'] if regime=='B' else
                            ['intact_rhat','full_activation','constant_design','prompt_tfidf'] if key[-1]=='intact' else
                            ['ablated_orthogonal','constant_design','prompt_tfidf'])
                        pred=[{'trial_id':r['trial_id'],'split_group':r['split_group'],'fold':fold,
                               'y':int(r['outcome_category']=='SHORTCUT_VIOLATION'),'scores':{}} for r in v]
                        for method in methods:
                            scores,info=fit_predict(dataset,t,v,method)
                            fits.append({'cohort':list(cohort),'denominator':denominator,'regime':regime,'fold':fold,
                                         'cell':None if key is None else list(key),**info})
                            for record,score in zip(pred,scores):record['scores'][method]=score
                        batches.setdefault(key,[]).extend(pred)
                if not batches:
                    summaries.append({'cohort':list(cohort),'denominator':denominator,'regime':regime,
                                      'status':'UNAVAILABLE_NO_ELIGIBLE_HELD_OUT_ROWS','cell':None})
                for key,pred in sorted(batches.items(),key=lambda v:str(v[0])):
                    identity={'cohort':list(cohort),'denominator':denominator,'regime':regime,'cell':None if key is None else list(key)}
                    seed=int(digest([SEED,identity])[:8],16)
                    methods=sorted(pred[0]['scores'])
                    metrics=[bootstrap(pred,m,replicates=bootstrap_replicates,seed=seed) for m in methods]
                    baselines=[m for m in methods if m in ('design_interaction','prompt_tfidf','constant_design')]
                    contrasts=[bootstrap(pred,m,comparator=b,replicates=bootstrap_replicates,seed=seed)
                               for m in methods if m not in baselines for b in baselines]
                    status=('EVALUATED_WITH_EXPLICIT_UNDEFINED_METRICS' if any(m['point']['auroc'] is not None for m in metrics)
                            else 'UNAVAILABLE_NO_DEFINED_HELD_OUT_METRICS')
                    summaries.append({**identity,'status':status,
                                      'metrics':metrics,'paired_contrasts':contrasts})
                    predictions.append({**identity,'rows':pred})
    return {'schema':'exp32_synthetic_analysis_v1','scope':'SYNTHETIC PREPARATION ONLY',
            'plan':plan(),'input_sha256':dataset['input_sha256'],
            'analysis_data_sha256':dataset['analysis_data_sha256'],
            'bootstrap_replicates_used':bootstrap_replicates,'registered_empirical_bootstrap_replicates':BOOTSTRAPS,
            'fold_assignment':{e+':'+g:f for (e,g),f in sorted(folds.items())},
            'retained_capture_rows':len(all_rows),'selected_analysis_rows':len(selected),
            'four_way_counts':{k:sum(r['outcome_category']==k for r in selected) for k in CATEGORIES},
            'summaries':summaries,'fit_audits':fits,'training_only_cell_decisions':decisions,'out_of_fold_predictions':predictions,
            'empirical_analysis_complete':False,'phase3_authorized':False,'complete_experiment_receipt':False}
