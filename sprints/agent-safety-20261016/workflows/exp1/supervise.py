#!/usr/bin/env python3
"""CPU-only supervisor; all paid inference goes through the shared broker."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
SPRINT=HERE.parents[1]
campaign=json.loads((SPRINT/"campaign.json").read_text())
deadline=campaign["inference_deadline_epoch"]
os.environ.update({"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})


def run(stage):
    print(f"Starting {stage}",flush=True)
    result=subprocess.run([sys.executable,str(HERE/"run.py"),stage],cwd=SPRINT)
    if result.returncode:
        print(f"Stage {stage} failed; attempt retained. No scientific resampling.",flush=True)
        subprocess.run([sys.executable,str(HERE/"run.py"),"closeout"],cwd=SPRINT)
        raise SystemExit(result.returncode)


while not (SPRINT/"shared"/"client.py").exists():
    if time.time()>deadline:
        raise SystemExit("Inference deadline reached before shared client was available")
    time.sleep(5)

run("pilot")
pilot=json.loads((HERE/"runs"/"pilot"/"analysis.json").read_text())
if not pilot["readiness_passed"]:
    run("closeout")
    raise SystemExit(0)
print("Pilot qualified; awaiting independent confirmation freeze",flush=True)
while not (HERE/"confirmation_freeze.json").exists():
    if time.time()>deadline:
        run("closeout")
        raise SystemExit(0)
    time.sleep(5)
run("confirmation")
if time.time()<deadline:
    run("live")
run("closeout")
