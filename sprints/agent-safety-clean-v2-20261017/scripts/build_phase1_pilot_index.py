"""Build the Phase 1 pilot index only after complete independent raw audits.

This creates no completion receipt, modifies no individual pilot, and authorizes
no subsequent phase. A clean negative qualification result remains a failure.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile

from phase1_readiness import normalize, pilot_markdown, relative_path, sha256, strict_json

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = 'infrastructure/phase1_component_paths_q4_descriptive.json'
OUTPUT = 'pilots/phase1_pilot_audit.md'
SOURCES = ('FINAL_RESEARCH_SPECIFICATION.md', 'scripts/build_phase1_pilot_index.py',
           'scripts/complete_experiment_receipts.py', 'scripts/phase1_readiness.py',
           'scripts/phase1_evidence.py', 'scripts/primary_mechanics.py')


def selected_pilots(receipts):
    """Select actual attempts, including per-arm external and forward retries."""
    from phase1_evidence import ARMS, resolve_logit_attempts
    result = []

    def add(component, stage, attempt, arm, *, whole=False, independent=True, multi=False):
        stem = 'phase1_'+stage+('_'+attempt if attempt else '')
        markdown = stem+'_pilot_audit.md' if component in {'source', 'author'} else stem+'_'+arm+'_pilot_audit.md'
        result.append({'component': component, 'stage': stage, 'attempt': attempt, 'arm': arm,
                       'raw_path': f'raw_transcripts/{stem}/{arm}.jsonl',
                       'markdown_path': 'pilots/'+markdown, 'whole_stream': whole,
                       'independently_parse': independent, 'multi_turn': multi})

    add('source', 'source_pilot', '', 'source_bf16', whole=True)
    # Source pilot uses the historical, already-existing selected log filename.
    result[-1]['markdown_path'] = 'pilots/phase1_source_pilot_audit.md'
    for arm in ('intact', 'refusal_ablated'):
        add('post_fp8', 'activation', receipts['post_fp8']['attempt'], arm)
    primary = receipts['primary']
    forwards = resolve_logit_attempts(primary['attempt'], primary.get('logit_attempt'), primary.get('logit_arm_attempts'))
    for arm in ARMS:
        jobs = {('qualification', primary['attempt']),
                (primary.get('logit_stage', 'qualification'), forwards[arm])}
        for stage, attempt in sorted(jobs):
            add('primary', stage, attempt, arm)
    external = receipts['external']
    for arm in ('flash_next_intact', 'flash_next_uncensored'):
        attempt = external.get('arm_attempts', {}).get(arm, external['attempt'])
        add('external', 'external', attempt, arm)
    add('author', 'author', receipts['author']['attempt'], 'flash_next_uncensored', independent=False)
    add('calibration', 'calibration', receipts['calibration']['attempt'], 'intact_cot_on', independent=False, multi=True)
    dose = receipts['dose']
    assert len(dose['empirical_curve']) == 5
    for entry in dose['empirical_curve']:
        add('dose', 'dose_response', dose['attempt'], entry['arm']+'_cot_on', independent=False, multi=True)
    assert len({row['raw_path'] for row in result}) == len(result), 'Duplicate selected pilot stream'
    assert len({row['markdown_path'] for row in result}) == len(result), 'Duplicate selected pilot log'
    return result


def reconstruct(root, components_path=COMPONENTS):
    """Read-only full reconstruction; absence/failure produces no index file."""
    from complete_experiment_receipts import reconstruct as audit_complete
    root = Path(root).resolve()
    bound = {}

    def bind(name, expected=None):
        name = relative_path(root, name)
        value = sha256(root/name)
        assert expected is None or value == expected, 'Bound evidence changed: '+name
        assert name not in bound or bound[name] == value, 'Evidence changed during index construction: '+name
        bound[name] = value
        return value

    selection_hash = bind(components_path)
    selection = strict_json((root/components_path).read_text())
    sources = normalize(root, selection)
    # Check every receipt exists before numerical reconstruction or output.
    receipts = {}
    for key, source in sources.items():
        assert source['receipt_path'], 'Selected component is absent: '+key
        bind(source['receipt_path'])
        receipts[key] = strict_json((root/source['receipt_path']).read_text())
    for name in SOURCES:
        bind(name)
    complete = audit_complete(root, 'phase1', 'receipt',
                              {'phase1_components': {'path': components_path, 'sha256': selection_hash}})
    assert complete['complete_coverage_receipt'] is True and complete['complete_experiment_receipt'] is True
    result = complete['result']; readiness = result['phase1_readiness']
    assert result['empirical_coverage_complete'] is True and result['statistical_analysis_complete'] is True
    assert complete['phase3_authorized'] is result['phase3_authorized'] is False
    assert readiness['component_sources'] == sources
    for key, component in readiness['components'].items():
        assert component['recomputed'] is True and component['status'] in {'PASS', 'FAIL'}
        bind(sources[key]['receipt_path'], component['receipt_sha256'])
    qualified = receipts['primary']['primary_checkpoints_qualified']
    assert result['primary_qualification_passed'] is qualified
    assert result['phase2_authorized'] is readiness['primary_phase2_authorized']
    if not qualified:
        assert result['phase2_authorized'] is False
        assert result['measurement_completion']['status'] == 'COMPLETE_MEASUREMENTS_PRIMARY_QUALIFICATION_FAILED'
        provenance = result['measurement_completion']['independent_measurement_provenance']
    else:
        provenance = readiness['provenance']
    assert provenance['status'] == 'PASS'
    for name, expected in provenance['artifact_sha256'].items():
        bind(name, expected)
    for name, expected in receipts['dose']['source_sha256'].items():
        bind(name, expected)

    pilots = []
    for selected in selected_pilots(receipts):
        raw = root/relative_path(root, selected['raw_path'])
        markdown = root/relative_path(root, selected['markdown_path'])
        raw_hash, markdown_hash = bind(selected['raw_path']), bind(selected['markdown_path'])
        pilot_markdown(root, raw, markdown, only_pilots=not selected['whole_stream'],
                       independently_parse=selected['independently_parse'])
        rows = [strict_json(line) for line in raw.read_text().splitlines()]
        group = rows if selected['whole_stream'] else [row for row in rows if row.get('split') == 'pilot']
        assert len({row['trial_id'] for row in group}) == len(group), 'Duplicate pilot raw trial IDs'
        ids = {row['episode_id'] if selected['multi_turn'] else row['trial_id'] for row in group}
        assert len(ids) == 5, 'Every selected job must have exactly five fresh pilot episodes'
        if not selected['multi_turn']:
            assert len(group) == 5
        pilots.append({**selected, 'pilot_episodes': len(ids), 'pilot_raw_records': len(group),
                       'total_raw_records': len(rows), 'raw_sha256': raw_hash, 'markdown_sha256': markdown_hash})
    for name, expected in bound.items():
        assert sha256(root/name) == expected, 'Evidence changed before index publication: '+name
    return {'schema': 'phase1_selected_pilot_index_v1', 'mechanical_pilot_audit_passed': True,
            'primary_qualification_passed': qualified, 'phase2_authorized': result['phase2_authorized'],
            'phase3_authorized': False, 'full_program_complete': False,
            'external_pair_qualified': receipts['external']['external_pair_qualified'],
            'measurement_status': result['measurement_completion']['status'],
            'complete_measurement_audit_sha256': hashlib.sha256(json.dumps(complete, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'selected_jobs': len(pilots), 'pilot_episodes': sum(row['pilot_episodes'] for row in pilots),
            'pilot_raw_records': sum(row['pilot_raw_records'] for row in pilots),
            'pilots': pilots, 'source_sha256': dict(sorted(bound.items()))}


def render(audit):
    qualified = 'PASS' if audit['primary_qualification_passed'] else 'FAIL'
    lines = ['# Phase 1 selected pilot audit', '', 'Status: PASS', '',
             'The status above certifies the selected mechanical pilot evidence. '
             'All seven Phase 1 measurement components were independently reconstructed before this index was created.', '',
             f'Primary scientific qualification: **{qualified}**. '
             f'Phase 2 authorized: **{str(audit["phase2_authorized"]).lower()}**. '
             'Phase 3 authorized: **false**. Full research program complete: **false**.', '',
             'This index preserves and links the original individual pilot transcripts. '
             'It does not waive qualification thresholds or authorize subsequent experiments.', '',
             f'Selected jobs: {audit["selected_jobs"]}; fresh pilot episodes: {audit["pilot_episodes"]}; '
             f'raw pilot generation records: {audit["pilot_raw_records"]}. Multi-turn episodes can contain multiple records.', '',
             '| Component / job | Episodes | Raw pilot records | Original audit | Raw stream |',
             '| --- | ---: | ---: | --- | --- |']
    for row in audit['pilots']:
        lines.append(f'| {row["component"]} / {row["stage"]} / {row["attempt"] or "source"} / {row["arm"]} '
                     f'| {row["pilot_episodes"]} | {row["pilot_raw_records"]} '
                     f'| [Audit](../{row["markdown_path"]}) | [JSONL](../{row["raw_path"]}) |')
    lines += ['', 'Exact counts, selected attempts, and SHA-256 evidence bindings:', '',
              '```json', json.dumps(audit, indent=2, sort_keys=True, allow_nan=False), '```', '']
    return '\n'.join(lines)


def build(root=ROOT, components_path=COMPONENTS):
    root = Path(root).resolve()
    audit = reconstruct(root, components_path)
    content = render(audit).encode()
    destination = root/OUTPUT
    relative_path(root, OUTPUT)
    if destination.exists():
        assert not destination.is_symlink() and destination.read_bytes() == content, 'Never replace a different pilot index'
        return audit
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A complete file is linked atomically; the required name is never a placeholder.
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.phase1_index_', delete=False) as handle:
            name = handle.name
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.link(name, destination)
    finally:
        if name is not None:
            Path(name).unlink()
    return audit


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--components', default=COMPONENTS)
    args = parser.parse_args()
    result = build(components_path=args.components)
    print(json.dumps({key: result[key] for key in ('selected_jobs', 'pilot_episodes', 'pilot_raw_records',
                     'primary_qualification_passed', 'phase2_authorized')}, indent=2))
