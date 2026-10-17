"""Preserve conditional resampling values; prohibit singleton structural CIs."""
import argparse,json
from pathlib import Path
import descriptor_sensitivity_v1
ROOT=Path(__file__).resolve().parent
def run(phase):
    descriptor_sensitivity_v1.run(phase)
    p=ROOT/'runs'/phase;data=json.loads((p/'descriptor_sensitivity_v1.json').read_text())
    for comparison in data['contrasts'].values():
        for value in comparison['sensitivities'].values():
            if not isinstance(value,dict) or 'descriptive95_cluster_bootstrap' not in value:continue
            conditional=value.pop('descriptive95_cluster_bootstrap')
            value['conditional95_resampling_interval_descriptive']=conditional
            missing=value['families_with_unidentified_between_structure_variance']
            value['between_descriptor95_interval']=None if missing else conditional
            value['between_descriptor_interval_qualified']=not missing
    data['qualification_revision']='v2 explicitly returns null between-descriptor intervals if any family has fewer than two clusters. Original conditional bootstrap output remains preserved in v1 and labelled conditional/descriptive here.'
    (p/'descriptor_sensitivity_qualified_v2.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['source_v1','target_v1']);run(ap.parse_args().phase)
