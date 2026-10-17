import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import json, warnings
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
SEED=20261016022

def auc(y,s):
 y=np.asarray(y,dtype=bool);s=np.asarray(s,float);n1=int(y.sum());n0=len(y)-n1
 return float((rankdata(s)[y].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None

def cluster_auc(rows,label,score,reps=2000):
 valid=[r for r in rows if r.get(label) is not None and r.get(score) is not None];worlds=sorted({r['world_id'] for r in valid});by={w:[r for r in valid if r['world_id']==w] for w in worlds};point=auc([r[label] for r in valid],[r[score] for r in valid])
 pos={r['world_id'] for r in valid if r[label]};neg={r['world_id'] for r in valid if not r[label]}
 out={'auc':point,'n':len(valid),'worlds':len(worlds),'positive_worlds':len(pos),'negative_worlds':len(neg)}
 if point is None:return {**out,'ci95':None,'reason':'Only one observed class'}
 rng=np.random.default_rng(SEED);values=[]
 for _ in range(reps):
  sample=[r for w in rng.choice(worlds,len(worlds),replace=True) for r in by[w]];a=auc([r[label] for r in sample],[r[score] for r in sample])
  if a is not None:values.append(a)
 return {**out,'ci95':np.quantile(values,[.025,.975]).tolist(),'usable_bootstraps':len(values)}

def fit(X,y,worlds):
 worlds=list(worlds);y=np.asarray(y);scale=StandardScaler().fit(X);Z=scale.transform(X);counts={w:worlds.count(w) for w in set(worlds)};weight=np.asarray([1/counts[w] for w in worlds]);weight*=len(weight)/weight.sum()
 model=LogisticRegression(C=1,solver='liblinear',max_iter=5000,random_state=SEED%(2**32-1))
 with warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always');model.fit(Z,y,sample_weight=weight)
 converged=not any(issubclass(w.category,ConvergenceWarning) for w in ww)
 if not converged:raise RuntimeError('Declared convergence gate failed')
 if not np.isfinite(model.coef_).all() or not np.linalg.norm(model.coef_):raise RuntimeError('Nonfinite or zero coefficient')
 return scale,model,{'converged':converged,'iterations':model.n_iter_.tolist(),'warnings':[str(w.message) for w in ww],'n':len(y),'worlds':len(set(worlds)),'features':X.shape[1]}

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False))
