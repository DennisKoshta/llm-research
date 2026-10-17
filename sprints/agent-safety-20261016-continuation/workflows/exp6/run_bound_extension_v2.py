"""Admission binding to independently reviewed prompt previews; no prompt changes."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main(phase):
    if phase=='semantic_names_v1':
        import semantic_names_v1 as runner
        requests,_,_=runner.prepare();ex=ROOT/'executions/semantic_names_v1';expected=json.loads((ex/'preflight_requests.json').read_text());review=ROOT.parent/'exp2/independent-semantic-name-request-review.json'
    else:
        import native_comparison_v1 as runner
        requests,_=runner.prepare(phase);ex=ROOT/'executions/native_comparison_v1';allq=json.loads((ex/'preflight_requests.json').read_text());allm=json.loads((ex/'preflight_assignments.json').read_text());is_pilot=phase=='native_pilot_v1';expected=[q for q,m in zip(allq,allm) if m['pilot']==is_pilot];review=ROOT.parent/'exp1/independent_exp6_native_request_review.json'
    assert review.exists(),'Independent request review must exist before admission'
    release=json.loads((ROOT/(phase+'-release.json')).read_text());assert release['approved'];assert requests==expected,'Prepared requests differ from reviewed preflight'
    # Review existence is already checked by root admission; retain exact preview binding here.
    p=ROOT/'runs'/phase;p.mkdir(parents=True,exist_ok=True);receipt={'phase':phase,'created_at':time.time(),'matched_requests':len(requests),'preflight_requests_sha256':sha(ex/'preflight_requests.json'),'preflight_manifest_sha256':sha(ex/'preflight_manifest.json'),'release_sha256':sha(ROOT/(phase+'-release.json')),'all_request_fields_equal_to_reviewed_preflight':True,'scope':'Named admission wrapper; no source/selection/prompt/model/parser changes.'}
    if review.exists():receipt['independent_request_review_path']=str(review);receipt['independent_request_review_sha256']=sha(review)
    (p/'preflight_binding_v1.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    if phase=='semantic_names_v1':
        runner.run();import audit_semantic_names_v1,analyze_semantic_names_v1;audit_semantic_names_v1.run();analyze_semantic_names_v1.run()
    else:
        runner.run(phase);import analyze_native_v1,analyze_native_secondaries_v2;analyze_native_v1.run();analyze_native_secondaries_v2.run()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['semantic_names_v1','native_pilot_v1','native_remaining_v1']);main(ap.parse_args().phase)
