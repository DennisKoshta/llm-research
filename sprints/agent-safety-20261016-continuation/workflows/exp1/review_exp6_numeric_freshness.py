"""Name-invariant numeric-task overlap; no outcomes read, no refitting."""
import collections,hashlib,json
from pathlib import Path
from review_exp6_source import oracle
H=Path(__file__).resolve().parent;E=H.parent/'exp6';res={};worlds={};hashes={}
for phase in ['pilot_v1','source_v1']:
 p=E/'runs'/phase;aa=json.loads((p/'assignments.json').read_text());gg=collections.defaultdict(dict);ww={w['world_id']:w for w in json.loads((p/'worlds.json').read_text())};worlds.update(ww)
 for q in json.loads((p/'requests.json').read_text()):
  a=aa[q['id']];gg[a['world_id']][a['utility']]=oracle(q['messages'][-1]['content'],a['family'])['semantic']
 res[phase]={w:ww[w]['family']+'|'+json.dumps(v,sort_keys=True) for w,v in gg.items()};hashes[phase]=hashlib.sha256((p/'requests.json').read_bytes()).hexdigest()
groups=collections.defaultdict(list)
for w,v in res['source_v1'].items():groups[v].append(w)
report={'status':'overlap_detected_preserve_originals','request_hashes':hashes,'source_instances':48,'source_unique_numeric_tasks':len(groups),'source_duplicate_groups':[[{'world_id':w,'split':worlds[w]['split'],'site':worlds[w]['site']} for w in v] for v in groups.values() if len(v)>1],'source_pilot_overlap':[{'source':s,'source_split':worlds[s]['split'],'pilot':p} for s,v in res['source_v1'].items() for p,u in res['pilot_v1'].items() if u==v],'normalization':'Parse both utility variants from actual prompts. Retain all utility values, deadlines, durations/setup/travel; ignore opaque IDs and site naming.','interpretation':'No source train-to-test numeric duplicate. One source calibration-to-test duplicate, one within-train duplicate pair, and one pilot-to-calibration overlap. Source data/splits and primary fitted thresholds stay frozen; source-test0018 exclusion is a separately labeled no-refit sensitivity. Fresh archive/reliability target-family transfer is unaffected.'}
(H/'independent_exp6_numeric_freshness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
