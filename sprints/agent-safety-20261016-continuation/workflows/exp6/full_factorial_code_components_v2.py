"""Planned output-score algebra across all factorial cells; no neural factor model."""
import argparse,hashlib,itertools,json,math,shutil,time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;EX=ROOT/'executions/full_factorial_code_components_v2'
ARMS=['narrative','payoff_table','permission_colocated','permission_separated','payoff_counterfactual','fact_table','outcome_prose']
COMPONENTS=['physical_operation_logodds','common_A_logodds']
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(phase):
    p=ROOT/'runs'/phase;meta={r['id']:r for r in json.loads((p/'scored_strict_v2.json').read_text())};pairs=defaultdict(dict);clips=0;normalization=[];ties=0;sign_mismatches=[]
    for line in (p/'raw.jsonl').open():
        r=json.loads(line);m=meta[r['id']];probs=r['conditional_label_probabilities'];a,b=probs['A'],probs['B'];assert all(math.isfinite(x) and 0<=x<=1 for x in [a,b]);normalization.append(abs(a+b-1));qa,qb=max(1e-12,a),max(1e-12,b);clips+=int(qa!=a)+int(qb!=b);z=math.log(qa/qb);key=(m['world_id'],m['arm'],m['authorized'],m['utility'],m['display_order']);label=m['designated_operation_label'];assert label not in pairs[key];pairs[key][label]=(z,m)
        ties+=z==0
        if z!=0 and m['valid'] and ('A' if z>0 else 'B')!=m['action']:sign_mismatches.append(r['id'])
    records=[]
    for key,pair in pairs.items():
        assert set(pair)=={'A','B'};za,ma=pair['A'];zb,mb=pair['B'];assert ma['family']==mb['family'];physical=(za-zb)/2;common=(za+zb)/2
        assert abs((physical+common)-za)<1e-10 and abs((common-physical)-zb)<1e-10
        records.append({'world_id':key[0],'family':ma['family'],'arm':key[1],'authorized':key[2],'utility':key[3],'display_order':key[4],'physical_operation_logodds':physical,'common_A_logodds':common,'valid_both':ma['valid'] and mb['valid'],'phase':phase})
    assert len(records)*2==len(meta)
    return records,{'phase':phase,'assigned_rows':len(meta),'pairs':len(records),'invalid_pairs':sum(not r['valid_both'] for r in records),'clipped_probability_entries':clips,'max_probability_sum_error':max(normalization),'zero_logodds_ties':ties,'greedy_sign_mismatches':sign_mismatches,'strict_scores_sha256':sha(p/'scored_strict_v2.json')}
def statistics(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['world_id'],r['arm'],r['authorized'],r['utility']].append(r)
    cells={key:{'world_id':key[0],'family':v[0]['family'],**{c:sum(r[c] for r in v)/len(v) for c in COMPONENTS},'orders':len(v)} for key,v in grouped.items()}
    world_family={r['world_id']:r['family'] for r in rows};families=sorted(set(world_family.values()));summaries={};rng=np.random.default_rng(261016669)
    groups={'all_families':sorted(world_family),**{f:sorted(w for w in world_family if world_family[w]==f) for f in families}}
    for group,worlds in groups.items():
        n=len(worlds);indices={w:i for i,w in enumerate(worlds)};draw=np.empty((20000,n),dtype=np.int32);offset=0
        # Stratified at the family level; all variants of a world travel together.
        for family in sorted({world_family[w] for w in worlds}):
            columns=np.array([indices[w] for w in worlds if world_family[w]==family]);draw[:,offset:offset+len(columns)]=columns[rng.integers(0,len(columns),(20000,len(columns)))];offset+=len(columns)
        factorial={}
        for authorized,utility in itertools.product([False,True],['better','worse']):
            byarm={arm:np.array([[cells[w,arm,authorized,utility][c] for c in COMPONENTS] for w in worlds]) for arm in ARMS}
            contrasts={}
            for control,treatment in itertools.combinations(ARMS,2):
                differences=byarm[treatment]-byarm[control];means=differences[draw].mean(axis=1);point=differences.mean(axis=0)
                contrasts[treatment+'_minus_'+control]={'treatment':treatment,'control':control,'worlds':n,**{c:{'mean':float(point[i]),'descriptive95_world_bootstrap':np.quantile(means[:,i],[.025,.975]).tolist()} for i,c in enumerate(COMPONENTS)}}
            factorial[f'authorized{int(authorized)}|{utility}']={'cell_means':{arm:{c:float(byarm[arm][:,i].mean()) for i,c in enumerate(COMPONENTS)} for arm in ARMS},'all21_presentation_contrasts':contrasts}
        summaries[group]={'worlds':n,'factorial':factorial}
    return summaries,[{**v,'arm':k[1],'authorized':k[2],'utility':k[3]} for k,v in cells.items()]
def run(phase):
    phases=['source_v1','target_v1'] if phase=='combined' else [phase];rows=[];audits=[]
    for item in phases:
        x,a=extract(item);rows+=x;audits.append(a)
    summaries,cells=statistics(rows);p=ROOT/'runs'/('combined_source_target_v2' if phase=='combined' else phase)
    result={'phase':phase,'status':'Planned secondary output-score analysis; source exposed, target prospectively specified; no changed behavioral primary.','definitions':{'zA':'log(P(A)/P(B)) when the designated physical operation is assignedA','zB':'log(P(A)/P(B)) when the same physical operation is assignedB','physical_operation_logodds':'(zA-zB)/2','common_A_logodds':'(zA+zB)/2'},'numeric_audits':audits,'summaries':summaries,'world_cell_values':cells,'paired_values':rows,'limits':['Exact algebra of two observed output scores, not a neural decomposition, identifiable latent cause, or unique fixed-letter effect.','Common-A component may contain context-dependent letter, menu-order, tokenization and numerical interactions; physical component may also contain those interactions.','Raw probabilities are conditional A/B temperature1 scores; actual choices use greedy decoding. Hard behavior can be asymmetric even when both physical and common-code components shift.','All21pairwise presentation contrasts and all authorization/utility cells are included;95% intervals descriptive, not multiplicity adjusted or promoted to new confirmatory endpoints.','Display orders averaged withinworld; families stratified in20,000worldbootstraps. Whole-world means are conditional on this generator and preserve documented numerical repeats.','Invalid output pairs remain in the prompt-probability analysis and are counted; they are not interpreted as compliant behavioral labels.','Unbounded log-odds intervals have no bounded[-1,1]Hoeffding claim.']}
    dump(p/'full_factorial_code_components_v2.json',result);print(json.dumps({'phase':phase,'numeric_audits':audits,'families':list(summaries),'world_cells':len(cells)}))
def freeze():
    EX.mkdir(parents=True,exist_ok=True);assert not (EX/'manifest.json').exists()
    for name in ['full_factorial_code_components_v2.py','full_factorial_code_components_v2_protocol.json']:shutil.copyfile(ROOT/name,EX/name)
    dump(EX/'manifest.json',{'frozen_at':time.time(),'hashes':{p.name:sha(p) for p in EX.iterdir() if p.is_file()},'source_exposed':True,'target_outcomes_exist':(ROOT/'runs/target_v1/raw.jsonl').exists(),'changes_behavioral_primary':False})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('phase',nargs='?',choices=['source_v1','target_v1','combined']);a=ap.parse_args()
    if a.freeze:freeze()
    else:assert a.phase;run(a.phase)
