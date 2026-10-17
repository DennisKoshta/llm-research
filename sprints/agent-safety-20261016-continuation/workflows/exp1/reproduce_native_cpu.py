"""Recompute frozen native main analysis into a separate directory; no inference."""
import importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;NAME='organization_native_confirmation_v7_144';R=H/'runs'/NAME;F=H/'executions'/NAME
sys.path.insert(0,str(F));spec=importlib.util.spec_from_file_location('frozen_native_organization',F/'organization_native_v7_144.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
worlds=json.loads((R/'worlds.json').read_text());rows=[json.loads(line) for line in (R/'scored.jsonl').open()];plans=json.loads((R/'plans.json').read_text());out=R/'cpu_reproduction';out.mkdir(exist_ok=True);result=module.analysis(out,worlds,rows,plans);expected=json.loads((R/'analysis.json').read_text());assert result==expected;receipt={'passed':True,'exact_primary_analysis_match':True,'worlds':len(worlds),'final_decisions':len(rows),'model_calls':0};(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
