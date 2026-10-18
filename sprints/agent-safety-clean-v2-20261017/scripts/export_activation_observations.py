"""Expose measured NPZ values as inspectable raw JSONL, without inventing completions."""
from pathlib import Path
import argparse,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def export(stage,attempt,arms):
    for arm in arms:
        prefix=ROOT/f'activations/phase1_{stage}_{attempt}'
        rows=[json.loads(line) for line in (prefix/f'{arm}_prompts.jsonl').read_text().splitlines()]
        destination=ROOT/f'raw_transcripts/phase1_{stage}_vectors_{attempt}/{arm}.jsonl'
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.jsonl.exporting')
        with np.load(prefix/f'{arm}.npz',allow_pickle=False) as data, temporary.open('w') as handle:
            assert [r['trial_id'] for r in rows]==data['trial_id'].tolist()
            activation=data['activations'];direction=data['direction'].tolist();layer=int(data['layer'])
            entries=data['entry_activations'] if 'entry_activations' in data else None
            logits=data['first_token_logits'] if 'first_token_logits' in data else None
            for i,row in enumerate(rows):
                observation={**row,'observation_type':'forward_only','raw_output':None,
                    'capture_position':'final_prompt_token','layer':layer,
                    'residual_activation':activation[i].tolist(),'projection_direction':direction,
                    'source_npz':str((prefix/f'{arm}.npz').relative_to(ROOT))}
                if entries is not None:observation['layer19_activation']=entries[i].tolist()
                if logits is not None:observation['first_token_logits']=logits[i].tolist()
                handle.write(json.dumps(observation,allow_nan=False)+'\n')
        temporary.replace(destination)
        print(str(destination.relative_to(ROOT)))
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',default='activation');p.add_argument('--attempt',required=True)
    p.add_argument('--arms',default='intact,refusal_ablated');a=p.parse_args();export(a.stage,a.attempt,a.arms.split(','))
if __name__=='__main__':main()
