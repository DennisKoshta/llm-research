"""Supplemental calibration provenance audit; no scientific effect is computed.

The old worker did not record generation-time tokenizer hashes or input token
IDs. A later CPU capture verifies the immutable checkpoint's eighteen shards
and captures its tokenizer bytes. Exact prompt reconstruction is therefore a
post-launch consistency check, not invented generation-time attestation.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import re

from audit_dose_response import file_hash, read_rows, prompt_renderer, complete_observations
from audit_sandbox_episodes import audit_stream
from prepare_calibration import prepare

ROOT = Path(__file__).resolve().parents[1]
NS = 'agent-safety-clean-v2-20261017'
SCHEMA = 'calibration_provenance_audit_v1'


def identifier(value):
    assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', value), 'Unsafe identifier'
    return value


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def capture_folder(root, attempt, capture_id=None):
    folder = root/'infrastructure/calibration_provenance'/identifier(attempt)
    if capture_id is not None:
        return folder/identifier(capture_id)
    completed = sorted(folder.glob('*/capture.json'))
    if not completed:
        raise FileNotFoundError(2, 'Checkpoint-bound CPU tokenizer capture is pending', str(folder/'*/capture.json'))
    assert len(completed) == 1, 'Multiple captures require an explicit capture_id'
    return completed[0].parent


def recompute(root, attempt, author_attempt, capture_id=None):
    """Return supplemental PASS/FAIL; missing artifacts raise FileNotFoundError.

    This function never changes numerical receipts or certifies calibration's
    scientific thresholds. Call ``audit`` for a PENDING-aware status object.
    """
    from modal_sandbox_evaluation import authorize_calibration
    from primary_model_loader import checkpoint_manifest
    from verify_all_receipts import audit_request_journal
    root = Path(root); identifier(attempt); identifier(author_attempt)
    folder = capture_folder(root, attempt, capture_id)
    capture_path = folder/'capture.json'; capture = json.loads(capture_path.read_text())
    assert capture['schema'] == 'calibration_checkpoint_capture_v1'
    assert capture['attempt'] == attempt and capture['capture_id'] == folder.name
    assert capture['generation_time_tokenizer_attestation'] is False
    manifest_path = root/'checkpoints_full_residual_v2/intact_materialization.json'
    assert capture['manifest_sha256'] == file_hash(manifest_path)
    model_path = f'/weights/{NS}/checkpoints_full_residual_v2/intact'
    manifest = checkpoint_manifest(model_path, root)
    assert capture['checkpoint_path'] == manifest['path'] == model_path
    assert capture['verified_shard_sha256'] == manifest['shard_sha256']
    assert len(capture['verified_shard_sha256']) == 18
    assert capture['capture_timing'] == 'POST_LAUNCH_CHECKPOINT_CONSISTENCY'
    bound = [capture_path, manifest_path]
    for name, expected in capture['captured_file_sha256'].items():
        path = folder/name
        path.resolve().relative_to(folder.resolve())
        assert file_hash(path) == expected, 'Captured tokenizer/config bytes changed'
        bound.append(path)
    hashes = capture['tokenizer_file_sha256']
    assert {'tokenizer_config.json', 'tokenizer.json'} <= hashes.keys()
    for name, expected in hashes.items():
        assert Path(name).name == name
        assert capture['captured_file_sha256']['checkpoint_doses_v1/tokenizer_audit/'+name] == expected
    render, _ = prompt_renderer(folder, hashes)
    config = prepare(root, author_attempt)
    authorization = authorize_calibration(root, config)
    stem = f'phase1_calibration_{attempt}_intact_cot_on'
    payload_path = root/f'sandbox_inputs/{stem}.json'
    ledger_path = root/f'group_calls/{stem}.json'
    payload = json.loads(payload_path.read_text()); ledger = json.loads(ledger_path.read_text())
    assert payload == {**config, 'namespace': NS, 'attempt': attempt,
                       'model_path': model_path, 'authorization': authorization}
    assert ledger['payload_sha256'] == digest(payload) and ledger['call_id']
    for path in (payload_path, ledger_path):
        name = str(path.relative_to(root))
        assert capture['execution_file_sha256'][name] == file_hash(path)
    bound.extend([payload_path, ledger_path])
    raw_path = root/f'raw_transcripts/phase1_calibration_{attempt}/intact_cot_on.jsonl'
    request_path = raw_path.with_name('intact_cot_on_requests.jsonl')
    # A missing terminal worker status is incomplete evidence, not a malformed
    # partial raw stream. The running worker only publishes terminal statuses.
    status_path = root/f'evaluation_status/{stem}.json'; status = json.loads(status_path.read_text())
    rows = read_rows(raw_path); requests = read_rows(request_path)
    episodes, hygiene = audit_stream(raw_path)
    journal = audit_request_journal(request_path)
    assert not journal['pending_trial_ids'] and journal['requested'] == journal['returned'] == len(rows)
    assert [row['trial_id'] for row in rows] == [row['trial_id'] for row in requests]
    expected = {row['trial_id']: (split, row) for split, group in
                [('pilot', config['pilots']), ('evaluation', config['trials'])] for row in group}
    assert len(expected) == 205 and len(episodes) == 205
    assert {row['episode_id'] for row in episodes} == set(expected)
    assert max(i for i, row in enumerate(rows) if row['split'] == 'pilot') < min(
        i for i, row in enumerate(rows) if row['split'] == 'evaluation'), 'Full sweep preceded pilot completion'
    grouped = defaultdict(list); token_id_rows = 0
    for row, request in zip(rows, requests):
        split, source = expected[row['episode_id']]
        assert row['checkpoint_path'] == model_path
        assert row['model_arm'] == 'intact' and row['cot_mode'] == 'cot_on' and row['split'] == split
        assert row['max_new_tokens'] == config['cap'] and row['temperature'] == 0 and row['forced_prefix'] is None
        assert row['observation_type'] == 'tool_turn'
        assert request['observation_type'] == 'generation_request' and request['raw_output'] is None
        for key, value in source.items():
            if key != 'trial_id':
                assert row[key] == request[key] == value, 'Raw/request source trial mismatch: '+key
        for key in ('trial_id', 'episode_id', 'turn_index', 'model_arm', 'cot_mode', 'split',
                    'full_prompt', 'max_new_tokens', 'temperature', 'forced_prefix'):
            assert row[key] == request[key], 'Raw/request mismatch: '+key
        if 'prompt_token_ids' in row or 'prompt_token_ids' in request:
            # Optional capture: token IDs are never substituted for byte replay.
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(folder/'checkpoint_doses_v1/tokenizer_audit/tokenizer.json'))
            actual = tokenizer.encode(row['full_prompt'], add_special_tokens=True).ids
            for item in (row, request):
                if 'prompt_token_ids' in item:
                    assert item['prompt_token_ids'] == actual, 'Prompt token IDs disagree with captured tokenizer'
            token_id_rows += 1
        grouped[row['episode_id']].append(row)
    for episode_id, turns in grouped.items():
        source = expected[episode_id][1]; history = [dict(message) for message in source['messages']]
        for row, observation in zip(turns, complete_observations(turns, source)):
            assert row['full_prompt'] == render(history), 'Prompt bytes differ from independently reconstructed history'
            history.extend([{'role': 'assistant', 'content': row['raw_output']},
                            {'role': 'user', 'content': 'Sandbox observation: '+json.dumps(observation)}])
    trajectory_path = root/f'trajectories/phase1_calibration_{attempt}/intact_cot_on.jsonl'
    assert read_rows(trajectory_path) == [{k: v for k, v in row.items() if k not in
                                         {'direct_taken', 'any_capped', 'pilot_valid'}} for row in episodes]
    assert status['experiment'] == 'phase1_calibration' and status['attempt'] == attempt
    assert status['arm'] == 'intact' and status['cot_mode'] == 'cot_on'
    assert status['status'] == ('COMPLETE' if hygiene['H1'] else 'H1_FAILED')
    assert status['pilot_pass'] == hygiene['H5'] and status['n_episodes'] == 200
    assert status['n_turns'] == hygiene['n_evaluation_turns']
    assert status['capped_turns'] == hygiene['capped_turns']
    assert status['capped_episodes'] == sum(row['any_capped'] for row in episodes if row['split'] == 'evaluation')
    pilot_path = root/f'pilots/{stem}_pilot_audit.md'
    pilot_rows = [row for row in rows if row['split'] == 'pilot']
    assert pilot_path.read_text() == '# Multi-step sandbox pilot\n\nStatus: '+('PASS' if hygiene['H5'] else 'FAIL')+'\n\n'+'\n\n'.join(
        '```json\n'+json.dumps(row, indent=2)+'\n```' for row in pilot_rows)
    for key, path in [('raw_path', raw_path), ('episode_path', trajectory_path), ('audit_path', pilot_path)]:
        assert status[key] == f'/artifacts/{NS}/'+str(path.relative_to(root))
    assert len(status['module_inventory']) == 1
    inventory = status['module_inventory'][0]
    assert inventory['expected_primary_unfused_modules'] == 400
    names = inventory['fp8_module_names']
    assert len(names) == len(set(names)) == inventory['loaded_primary_fused_fp8_modules'] > 0
    assert len([name for name in manifest['quantized_modules'] if not name.startswith('mtp.')]) == 400
    bound.extend([raw_path, request_path, trajectory_path, status_path, pilot_path])
    passed = all(hygiene[key] for key in ('H1', 'H2', 'H5'))
    return {'schema': SCHEMA, 'attempt': attempt, 'author_attempt': author_attempt,
            'capture_id': folder.name, 'status': 'PASS' if passed else 'FAIL', 'provenance_passed': passed,
            'exact_prompt_history_replay': True, 'request_raw_source_trial_match': True,
            'immutable_intact_checkpoint_bound': True, 'worker_fp8_inventory_verified': True,
            'generation_time_tokenizer_attestation': False, 'prompt_token_ids_available_rows': token_id_rows,
            'prompt_rows_replayed': len(rows), 'hygiene': hygiene,
            'limitations': ['Tokenizer bytes were captured after launch; no generation-time tokenizer digest was recorded.',
                           'Input token IDs were optional and are audited only where recorded.'],
            'artifact_sha256': {str(path.relative_to(root)): file_hash(path) for path in sorted(set(bound))}}


def audit(root, attempt, author_attempt, capture_id=None):
    try:
        return recompute(root, attempt, author_attempt, capture_id)
    except FileNotFoundError as exc:
        return {'schema': SCHEMA, 'attempt': attempt, 'status': 'PENDING', 'provenance_passed': False,
                'missing_artifact': str(exc.filename)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', required=True)
    parser.add_argument('--author-attempt', required=True); parser.add_argument('--capture-id')
    args = parser.parse_args()
    print(json.dumps(audit(ROOT, args.attempt, args.author_attempt, args.capture_id), indent=2, allow_nan=False))
