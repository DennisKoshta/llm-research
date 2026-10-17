"""Approved collection followed by named strict scoring and CPU-only audits."""
import argparse,json,time
from pathlib import Path
import experiment,strict_scoring_v2,analyze_strict_v2,audit_strict_v2
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['source_v1','target_v1']);phase=ap.parse_args().phase
    experiment.run(phase)
    strict_scoring_v2.run(phase)
    analyze_strict_v2.run(phase)
    audit_strict_v2.run(phase)
    p=ROOT/'runs'/phase;a=json.loads((p/'audit_strict_v2.json').read_text())
    experiment.dump(ROOT/'status.json',{'status':phase+'_collected_and_scored','audit_passed':a['passed'],'early_state_mismatches':len(a['early_state_mismatches']),'updated':time.time(),'further_research_requires_separate_phase':True})
