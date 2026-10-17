"""Guarded entrypoint for the frozen B source; no experimental settings change."""
import argparse,json,subprocess,sys
from pathlib import Path
from run import sha
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('stage',choices=['bind','pilot','main']);p.add_argument('--engineering-receipt',type=Path);a=p.parse_args();freeze=H/'executions/cost_gradient_v1';m=json.loads((freeze/'manifest.json').read_text())
for name,h in m['files'].items():
 assert sha(freeze/name)==h
 if name.endswith('.py'):assert sha(H/name)==h,'Working implementation differs from reviewed frozen source: '+name
cmd=[sys.executable,str(H/'run.py'),a.stage]
if a.engineering_receipt is not None:cmd+=['--engineering-receipt',str(a.engineering_receipt.resolve())]
subprocess.run(cmd,cwd=H,check=True)
