"""Recompute the post-FP8 intervention gate from saved activation arrays."""
from pathlib import Path
import argparse,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def recompute(root=ROOT,attempt='v1'):
    means={};counts={}
    ids=None
    for arm in ('intact','refusal_ablated'):
        path=root/f'activations/phase1_activation_{attempt}/{arm}.npz'
        with np.load(path,allow_pickle=False) as data:
            h=data['activations'].astype(np.float64);r=data['direction'].astype(np.float64)
            assert h.shape==(64,5120) and np.isfinite(h).all()
            assert abs(np.linalg.norm(r)-1)<1e-6
            if ids is None:ids=data['trial_id']
            else:assert np.array_equal(ids,data['trial_id'])
            assert len(set(data['trial_id']))==64
            means[arm]=float(np.mean(np.abs(h@r)/np.linalg.norm(h,axis=1)))
            counts[arm]=len(h)
    reduction=1-means['refusal_ablated']/means['intact']
    return {'schema':'phase1_post_fp8_v1','attempt':attempt,'n_by_arm':counts,
            'mean_absolute_normalized_projection':means,'fraction_reduction':reduction,
            'required_fraction_reduction_strictly_greater_than':.95,
            'passed':bool(reduction>.95),'source':'materialized FP8 forward passes, final prompt token'}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',default='v1');args=parser.parse_args()
    result=recompute(attempt=args.attempt)
    folder=ROOT/'qualification_status';folder.mkdir(exist_ok=True)
    (folder/f'activation_gate_{args.attempt}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
