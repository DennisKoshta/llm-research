"""Launch the immutable execution, retaining the workflow as its asset root."""
import os
from pathlib import Path
import runpy
import sys
root=Path(__file__).resolve().parent
os.environ['C_ASSET_ROOT']=str(root)
execution=root/'executions/mixed_teams_v3'
sys.path.insert(0,str(execution))
runpy.run_path(str(execution/'experiment.py'),run_name='__main__')
