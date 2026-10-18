"""Deterministic vector estimation; refuses infeasible variance matching."""
from pathlib import Path
import json
import numpy as np


def variance_matched_pca(activation, reference, seed):
    centered = np.asarray(activation,dtype=np.float64)-np.mean(activation,axis=0)
    reference = np.asarray(reference,dtype=np.float64)
    target = float(np.var(centered@reference,ddof=1))
    _,singular,vt = np.linalg.svd(centered,full_matrices=False)
    eigenvalues = singular**2/(len(centered)-1)
    # Use positive-variance empirical PCs only; do not add a null-space control.
    usable = eigenvalues > eigenvalues[0]*1e-10
    eigenvalues,vt = eigenvalues[usable],vt[usable]
    above = np.flatnonzero(eigenvalues>target)
    below = np.flatnonzero(eigenvalues<target)
    if not len(above) or not len(below):
        raise ValueError('Reference variance is outside the positive PCA spectrum; cannot match honestly')
    rng=np.random.default_rng(seed)
    a=rng.normal(size=len(above)); a/=np.linalg.norm(a)
    b=rng.normal(size=len(below)); b/=np.linalg.norm(b)
    v_high=a@vt[above]; v_low=b@vt[below]
    s_high=float(np.sum(a*a*eigenvalues[above]))
    s_low=float(np.sum(b*b*eigenvalues[below]))
    weight=(target-s_low)/(s_high-s_low)
    vector=np.sqrt(weight)*v_high+np.sqrt(1-weight)*v_low
    observed=float(np.var(centered@vector,ddof=1))
    assert np.isclose(np.linalg.norm(vector),1,rtol=1e-10)
    assert np.isclose(observed,target,rtol=1e-9,atol=1e-10)
    return vector,{'seed':seed,'reference_variance':target,'control_variance':observed,
                   'pc_count':len(eigenvalues),'reference_cosine':float(vector@reference)}


def estimate(activations, labels, layers):
    # Stratified 96/32 pair split preserves 50% action and 50% text in each.
    x=np.asarray(activations,dtype=np.float64)
    labels=np.asarray(labels)
    assert x.shape[0]==256 and labels.shape==(256,)
    train=np.array([2*i+h for i in list(range(48))+list(range(64,112)) for h in (0,1)])
    validation=np.array([2*i+h for i in list(range(48,64))+list(range(112,128)) for h in (0,1)])
    vectors=[]; audits=[]
    for j,layer in enumerate(layers):
        h=x[train,j,:]
        rms=np.sqrt(np.mean(h*h,axis=0))
        masked=rms>10*np.median(rms)
        delta=h[labels[train]==1].mean(axis=0)-h[labels[train]==0].mean(axis=0)
        delta[masked]=0
        if np.linalg.norm(delta)==0: raise ValueError('Degenerate refusal candidate')
        direction=delta/np.linalg.norm(delta)
        projections=x[validation,j,:]@direction
        score=(projections[labels[validation]==1].mean()-projections[labels[validation]==0].mean())/max(projections.std(ddof=1),1e-12)
        vectors.append(direction)
        audits.append({'layer':int(layer),'masked_channels':np.flatnonzero(masked).tolist(),
                       'validation_standardized_separation':float(score)})
    chosen=max(range(len(layers)),key=lambda j:audits[j]['validation_standardized_separation'])
    direction=vectors[chosen]
    controls={}; matching=[]
    for seed in (101,102,103):
        v,audit=variance_matched_pca(x[validation,chosen,:],direction,seed)
        controls[f'pca_{seed}']=v; matching.append(audit)
    return {'refusal':direction,**controls}, {'selected_layer':int(layers[chosen]),
            'candidate_layers':audits,'variance_matching':matching,
            'outlier_rule':'RMS > 10 * median channel RMS on estimation subset',
            'selection_rule':'maximum validation standardized projection separation',
            'unrelated_control_status':'REQUIRES_SEPARATE_FORMAL_CASUAL_ACTIVATIONS'}


def main():
    root=Path(__file__).resolve().parents[1]
    data=np.load(root/'activations/phase1_source.npz',allow_pickle=False)
    labels=[int(str(t).endswith('_harmful')) for t in data['trial_id'][:256]]
    vectors,audit=estimate(data['activations'][:256],labels,data['layers'])
    j=list(data['layers']).index(audit['selected_layer'])
    style=data['activations'][256:320,j,:].astype(np.float64)
    ids=[str(t) for t in data['trial_id'][256:320]]
    assert all(t.startswith('style_') for t in ids)
    formal=np.array([t.endswith('_formal') for t in ids])
    unrelated=style[formal].mean(axis=0)-style[~formal].mean(axis=0)
    unrelated/=np.linalg.norm(unrelated)
    vectors['unrelated']=unrelated
    audit['unrelated_control_status']='FORMAL_MINUS_CASUAL_ESTIMATED'
    audit['unrelated_refusal_cosine']=float(unrelated@vectors['refusal'])
    np.savez_compressed(root/'activations/phase1_directions.npz',**vectors)
    (root/'activations/phase1_direction_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
