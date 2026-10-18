"""Expose source forward observations and re-estimate frozen directions from JSONL."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from prepare_directions import estimate

ROOT=Path(__file__).resolve().parents[1]


def export(root):
    destination=root/'raw_transcripts/phase1_source_vectors/intact_bf16.jsonl'
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    with (root/'activations/phase1_source_prompts.jsonl').open() as handle:prompts=[json.loads(line) for line in handle]
    temporary=destination.with_suffix('.jsonl.exporting')
    with np.load(root/'activations/phase1_source.npz',allow_pickle=False) as data,temporary.open('w') as handle:
        assert [r['trial_id'] for r in prompts]==data['trial_id'].tolist()
        activation=data['activations'];layers=data['layers'].tolist()
        assert activation.shape==(448,8,5120)
        for index,row in enumerate(prompts):
            handle.write(json.dumps({**row,'observation_type':'source_forward_only','raw_output':None,
                'layers':layers,'residual_activations':activation[index].tolist(),
                'source_npz':'activations/phase1_source.npz'},allow_nan=False)+'\n')
    temporary.replace(destination)
    return destination


def recompute(root):
    source=root/'raw_transcripts/phase1_source_vectors/intact_bf16.jsonl'
    with source.open() as handle:rows=[json.loads(line) for line in handle]
    assert len(rows)==448 and len({r['trial_id'] for r in rows})==448
    layers=rows[0]['layers'];assert layers==[20,24,28,32,36,40,44,48]
    assert all(r['layers']==layers and r['raw_output'] is None and r['full_prompt'] and
               r['capture_position']=='final_prompt_token' for r in rows)
    activation=np.asarray([r['residual_activations'] for r in rows],dtype=np.float64)
    assert activation.shape==(448,8,5120) and np.isfinite(activation).all()
    with np.load(root/'activations/phase1_source.npz',allow_pickle=False) as saved:
        assert np.array_equal(saved['activations'],activation)
        assert saved['trial_id'].tolist()==[r['trial_id'] for r in rows]
    labels=[int(r['trial_id'].endswith('_harmful')) for r in rows[:256]]
    assert labels==[int(r['consequence']=='harmful') for r in rows[:256]]
    vectors,audit=estimate(activation[:256],labels,layers)
    index=layers.index(audit['selected_layer']);style=activation[256:320,index,:]
    ids=[r['trial_id'] for r in rows[256:320]];assert all(t.startswith('style_') for t in ids)
    formal=np.asarray([t.endswith('_formal') for t in ids])
    unrelated=style[formal].mean(axis=0)-style[~formal].mean(axis=0);unrelated/=np.linalg.norm(unrelated)
    vectors['unrelated']=unrelated
    audit['unrelated_control_status']='FORMAL_MINUS_CASUAL_ESTIMATED'
    audit['unrelated_refusal_cosine']=float(unrelated@vectors['refusal'])
    assert audit==json.loads((root/'activations/phase1_direction_audit.json').read_text())
    with np.load(root/'activations/phase1_directions.npz',allow_pickle=False) as saved:
        assert set(saved.files)==set(vectors)
        for key,value in vectors.items():assert np.array_equal(saved[key],value),key
    return {'schema':'source_direction_raw_audit_v1','source_observations':448,'extraction_observations':256,
            'style_observations':64,'qualification_forward_observations':128,'audit':audit,
            'vector_sha256':{key:hashlib.sha256(value.astype('<f8').tobytes()).hexdigest() for key,value in sorted(vectors.items())},
            'raw_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'npz_crosscheck_exact':True,
            'frozen_vectors_recomputed_exactly':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true');args=p.parse_args()
    if args.export:export(ROOT)
    result=recompute(ROOT)
    (ROOT/'records/phase1_source_directions_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
