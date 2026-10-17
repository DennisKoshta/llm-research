"""Atomic derivative certifying a fully written strict table; no labels changed."""
import hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=ROOT/'runs/target_v1';receipt=p/'strict_scoring_v2.json';table=p/'scored_strict_v2.json';dest=p/'strict_completion_digest_v1.json'
    while True:
        try:
            r=json.loads(receipt.read_text());blob=table.read_bytes();rows=json.loads(blob)
            assert r['assigned']==len(rows)==4992 and r['execution_valid']==sum(x['valid'] for x in rows)
            break
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(2)
    result={'version':'strict_completion_digest_v1','created_at':time.time(),'scored_strict_v2_sha256':hashlib.sha256(blob).hexdigest(),'strict_scoring_v2_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'assigned':len(rows),'scope':'Named atomic completion certificate; existing scorer, labels, estimands and immutable source files unchanged.'}
    if dest.exists():
        old=json.loads(dest.read_text());assert old['scored_strict_v2_sha256']==result['scored_strict_v2_sha256'];return
    tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,dest);print(json.dumps(result))
if __name__=='__main__':main()
