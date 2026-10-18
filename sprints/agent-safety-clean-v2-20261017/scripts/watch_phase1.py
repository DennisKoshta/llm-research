"""Bounded evidence recovery and automatic gate computation for launched jobs.

This supervises existing jobs only. It cannot launch a downstream experiment or
waive missing authoring/calibration requirements.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('intact', 'refusal_ablated', 'pca_101', 'pca_102', 'pca_103', 'unrelated')


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def recover(stage, attempt):
    result = subprocess.run([sys.executable, str(ROOT/'scripts/sync_phase1_artifacts.py'),
                             '--stage', stage, '--attempt', attempt], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])


def progress(path):
    if not path.exists():
        return {'pilot_n': 0, 'sweep_n': 0}
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]
    trials = [r for r in rows if r['split'] == 'qualification']
    return {'pilot_n': sum(r['split'] == 'pilot' for r in rows), 'sweep_n': len(trials),
            'capped': sum(r['generated_tokens'] >= r['max_new_tokens'] or r['finish_reason'] == 'length' for r in trials)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary-attempt', default='full_v2_q1')
    parser.add_argument('--external-attempt', default='v3')
    parser.add_argument('--external-intact-attempt')
    parser.add_argument('--external-uncensored-attempt')
    parser.add_argument('--logit-stage', default='qualification')
    parser.add_argument('--logit-attempt')
    parser.add_argument('--logit-arm-attempts',help='JSON file containing all six forward-arm attempt IDs')
    parser.add_argument('--duration-seconds', type=int, default=14400)
    parser.add_argument('--poll-seconds', type=int, default=120)
    parser.add_argument('--status-file', default='infrastructure/phase1_watch_status.json')
    args = parser.parse_args()
    assert 0 < args.duration_seconds <= 14400
    assert 60 <= args.poll_seconds <= 600
    sys.path.insert(0, str(ROOT/'scripts'))
    from export_activation_observations import export
    from phase1_evidence import recompute_primary, recompute_external, resolve_logit_attempts
    from primary_mechanics import inspect as inspect_mechanics
    logit_arm_attempts=json.loads(Path(args.logit_arm_attempts).read_text()) if args.logit_arm_attempts else None
    logit_sources=resolve_logit_attempts(args.primary_attempt,args.logit_attempt,logit_arm_attempts)
    start = time.monotonic()
    previous = None
    primary_done = False; external_done = False
    external_sources = {
        'flash_next_intact': args.external_intact_attempt or args.external_attempt,
        'flash_next_uncensored': args.external_uncensored_attempt or args.external_attempt,
    }
    while time.monotonic()-start < args.duration_seconds:
        try:
            if not primary_done:
                recover('qualification', args.primary_attempt)
                for logit_attempt in sorted(set(logit_sources.values())):
                    if args.logit_stage != 'qualification' or logit_attempt != args.primary_attempt:
                        selected=[arm for arm in ARMS if logit_sources[arm]==logit_attempt]
                        complete=True
                        for arm in selected:
                            status=ROOT/f'qualification_status/{args.logit_stage}_{logit_attempt}_{arm}.json'
                            vectors=ROOT/f'activations/phase1_{args.logit_stage}_{logit_attempt}/{arm}.npz'
                            raw=ROOT/f'raw_transcripts/phase1_{args.logit_stage}_{logit_attempt}/{arm}.jsonl'
                            if not (status.exists() and vectors.exists() and raw.exists()):
                                complete=False;break
                            if json.loads(status.read_text()).get('status')!='ACTIVATIONS_COMPLETE':
                                complete=False;break
                        # Immutable finished forward observations need no repeated
                        # cloud listing while behavioral generation continues.
                        # Full independent receipt checks still run below.
                        if not complete:recover(args.logit_stage,logit_attempt)
            if not external_done:
                for attempt in sorted(set(external_sources.values())):
                    recover('external', attempt)
            snapshot = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                        'primary_attempt': args.primary_attempt, 'external_attempt': args.external_attempt,
                        'external_sources': external_sources,
                        'logit_sources':logit_sources,
                        'primary': {}, 'external': {}, 'phase2_authorized_by_all_prerequisites': False,
                        'remaining_phase1_prerequisites': ['scenario authoring', 'Intact-only calibration', 'alpha dose-response']}
            for arm in ARMS:
                snapshot['primary'][arm] = progress(ROOT/f'raw_transcripts/phase1_qualification_{args.primary_attempt}/{arm}.jsonl')
                logit_attempt = logit_sources[arm]
                source = ROOT/f'activations/phase1_{args.logit_stage}_{logit_attempt}/{arm}.npz'
                destination = ROOT/f'raw_transcripts/phase1_{args.logit_stage}_vectors_{logit_attempt}/{arm}.jsonl'
                if source.exists() and not destination.exists():
                    export(args.logit_stage, logit_attempt, [arm])
            for arm in ('flash_next_intact', 'flash_next_uncensored'):
                snapshot['external'][arm] = progress(ROOT/f'raw_transcripts/phase1_external_{external_sources[arm]}/{arm}.jsonl')
            vectors_ready = all((ROOT/f'raw_transcripts/phase1_{args.logit_stage}_vectors_{logit_sources[a]}/{a}.jsonl').exists() for a in ARMS)
            mechanics = inspect_mechanics(ROOT, args.primary_attempt)
            snapshot['primary_mechanical_precheck'] = mechanics
            if mechanics['terminal'] and not mechanics['passed']:
                snapshot['state'] = 'PRIMARY_MECHANICAL_GATE_FAILED_BEFORE_EFFECT_COMPUTATION'
                save(ROOT/args.status_file, snapshot)
                save(ROOT/'qualification_status'/f'primary_mechanical_gate_{args.primary_attempt}.json', mechanics)
                print(json.dumps(snapshot), flush=True)
                return
            if not primary_done and vectors_ready and all(x['sweep_n'] == 128 for x in snapshot['primary'].values()):
                assert mechanics['passed'], 'Mechanical gates must pass before held-out effect computation'
                result = recompute_primary(ROOT, args.primary_attempt, 'full_v2', args.logit_stage, args.logit_attempt,logit_arm_attempts)
                save(ROOT/'qualification_status'/f'primary_raw_audit_{args.primary_attempt}.json', result)
                save(ROOT/'records'/f'phase1_primary_qualification_{args.primary_attempt}_receipt.json', result)
                primary_done = True
                with (ROOT/'PROGRESS_AND_GATES.md').open('a') as handle:
                    handle.write(f'\n- Automatic raw-data audit completed all six primary arms ({args.primary_attempt}). Primary checkpoint qualification: **{result["primary_checkpoints_qualified"]}**. Full metrics and tests: `records/phase1_primary_qualification_{args.primary_attempt}_receipt.json`.\n')
            if not external_done and all(x['sweep_n'] == 64 for x in snapshot['external'].values()):
                result = recompute_external(ROOT, args.external_attempt, external_sources)
                save(ROOT/'qualification_status'/f'external_raw_audit_{args.external_attempt}.json', result)
                save(ROOT/'records'/f'phase1_external_qualification_{args.external_attempt}_receipt.json', result)
                external_done = True
                with (ROOT/'PROGRESS_AND_GATES.md').open('a') as handle:
                    handle.write(f'\n- Automatic raw-data audit completed both external arms ({args.external_attempt}). External pair qualification: **{result["external_pair_qualified"]}**. Metrics and tests: `records/phase1_external_qualification_{args.external_attempt}_receipt.json`.\n')
            snapshot['primary_audited'] = primary_done; snapshot['external_audited'] = external_done
            save(ROOT/args.status_file, snapshot)
            counts = json.dumps({'primary': snapshot['primary'], 'external': snapshot['external']}, sort_keys=True)
            if counts != previous:
                print(json.dumps(snapshot), flush=True); previous = counts
            if primary_done and external_done:
                return
        except Exception as exc:
            event = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                     'error': type(exc).__name__+': '+str(exc), 'action': 'retry existing artifact recovery; no new GPU jobs'}
            with (ROOT/'infrastructure/watch_recovery_errors.jsonl').open('a') as handle:
                handle.write(json.dumps(event)+'\n')
            print(json.dumps(event), flush=True)
        time.sleep(args.poll_seconds)
    print('Bounded monitoring deadline reached; scientific completion is not implied.', flush=True)


if __name__ == '__main__':
    main()
