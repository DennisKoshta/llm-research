"""Root admission for fixed A/B pilots after shared manipulation qualifies."""
from pathlib import Path
import hashlib,json,time
from shared.client import ROOT,atomic

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def immutable(p,x):
    if p.exists():assert read(p)==x
    else:atomic(p,x)

def main():
    lockpath=ROOT/'records/intervention-lock.json';qpath=ROOT/'records/intervention-qualified.json'
    lock=read(lockpath);q=read(qpath)
    assert q['passed'] and q['intervention_lock_sha256']==sha(lockpath)
    assert lock['version']=='instrument_v2','Named instrument revision required; do not silently switch'
    a=ROOT/'workflows/a';manifest=a/'executions/ordinary_goal_boundaries_v2/manifest.json'
    assert sha(manifest)=='4276cfdf029341127a6a737ec5a38df52e59fd8e4e0ce17aa8a9094cd6ffd533'
    arms={}
    for arm in ['baseline','zero_sham','refusal_ablation','random_ablation']:
        spec=None if arm=='baseline' else lock['random_spec_sha256'] if arm=='random_ablation' else lock['refusal_spec_sha256']
        arms[arm]={'arm':arm,'spec_sha256':spec,'direction_key':'locked',
          'coverage':'all prompt and generation positions; decoder input,attention output,MLP output',
          'coefficient':0 if arm in ['baseline','zero_sham'] else 1,'weight_mutation':False}
    immutable(a/'pilot-release.json',{'approved':True,'stage':'pilot',
      'execution_manifest_sha256':sha(manifest),'intervention_lock_path':'records/intervention-lock.json',
      'intervention_lock_sha256':sha(lockpath),'qualification_receipt_path':'records/intervention-qualified.json',
      'qualification_receipt_sha256':sha(qpath),'expected_intervention_by_arm':arms,
      'scope':'Fixed720-call pilot only; independent pilot review and a separate confirmation release remain required.'})
    b=ROOT/'workflows/b';release=read(b/'pilot-release-template.json')
    for name,expected in release['files'].items():assert sha(Path(name))==expected
    for name in release.pop('pending_files_to_hash_after_qualification'):
        release['files'][name]=sha(Path(name))
    release['approved']=True;release['notes']='Explicit root pilot admission using separately qualified instrument_v2; task worlds/endpoints unchanged. Main remains gated.'
    immutable(ROOT/'records/b-pilot-release.json',release)
    print(json.dumps({'A':'720-call pilot released','B':'792-call pilot released','instrument_lock_sha256':sha(lockpath)}),flush=True)

if __name__=='__main__':main()
