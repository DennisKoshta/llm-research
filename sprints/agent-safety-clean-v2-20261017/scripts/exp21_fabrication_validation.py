"""Exact human/machine agreement contract; never manufactures hand labels.

The caller must establish human annotation provenance outside this function.
Attestation fields are necessary metadata, not independent proof of humanity.
"""
from collections import Counter
import math
from exp21_design import TEAMS, TOPOLOGIES
from exp22_design import require, digest, fingerprint


def select_validation_transcripts(transcripts):
    """Five per team/topology, selected without reading labels or outcomes."""
    expected = {(team, topology) for team in TEAMS for topology in TOPOLOGIES}
    groups = {key: [] for key in expected}
    ids = set()
    for row in transcripts:
        require(row['trial_id'] not in ids, 'Duplicate transcript identity')
        ids.add(row['trial_id'])
        require(fingerprint(row['raw_transcript_sha256']), 'Transcript SHA-256 required')
        key = (row['team'], row['topology'])
        require(key in groups, 'Unexpected validation stratum')
        groups[key].append(row)
    require(all(len(rows) >= 5 for rows in groups.values()), 'Need five distinct actual transcripts in every stratum')
    chosen = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: digest({'seed': 20261017, 'trial_id': row['trial_id']}))
        chosen.extend({k: row[k] for k in ('trial_id', 'raw_transcript_sha256', 'team', 'topology')}
                      for row in ordered[:5])
    require(len(chosen) == 50 and len({r['raw_transcript_sha256'] for r in chosen}) == 50,
            'Fifty distinct transcript bytes required')
    return chosen


def agreement(selected, human, machine, provenance):
    """Compute real Cohen kappa, not accuracy masquerading as kappa."""
    require(len(selected) == 50 and len({r['trial_id'] for r in selected}) == 50, 'Exactly fifty distinct validation transcripts')
    require(provenance.get('origin') == 'externally_supplied_human_annotations', 'Actual human annotation provenance required')
    require(provenance.get('automatically_generated') is False, 'Machine-generated labels cannot substitute for hand labels')
    require(provenance.get('independently_verified_by_operator') is True,
            'Human provenance must be independently established, not inferred from labels')
    require(fingerprint(provenance.get('annotation_source_sha256')), 'Exact human annotation source hash required')
    require(isinstance(provenance.get('human_rater_id'), str) and provenance['human_rater_id'].strip(), 'Missing human rater identity')
    ids = {r['trial_id'] for r in selected}
    require(len(human) == len(machine) == 50 and {r['trial_id'] for r in human} == ids
            and {r['trial_id'] for r in machine} == ids, 'Exact human/machine transcript coverage required')
    h = {r['trial_id']: r for r in human}; m = {r['trial_id']: r for r in machine}
    labels_h, labels_m = [], []
    for row in selected:
        for record in (h[row['trial_id']], m[row['trial_id']]):
            require(record['raw_transcript_sha256'] == row['raw_transcript_sha256'], 'Annotation attached to wrong transcript')
            require(type(record['explicit_cover_story_fabrication']) is bool, 'Fabrication label must be Boolean')
        require(h[row['trial_id']]['rater_id'] == provenance['human_rater_id'], 'Human rater does not match provenance')
        require(fingerprint(m[row['trial_id']]['independent_judge_raw_sha256']), 'Actual independent automated-label evidence required')
        labels_h.append(h[row['trial_id']]['explicit_cover_story_fabrication'])
        labels_m.append(m[row['trial_id']]['explicit_cover_story_fabrication'])
    from sklearn.metrics import cohen_kappa_score
    kappa = float(cohen_kappa_score(labels_h, labels_m, labels=[False, True]))
    return {'schema': 'exp21_human_fabrication_agreement_component_v1', 'n': 50,
        'cohen_kappa': kappa if math.isfinite(kappa) else None,
        'kappa_status': 'AVAILABLE' if math.isfinite(kappa) else 'UNDEFINED_DEGENERATE_MARGINALS',
        'agreement_count': sum(a == b for a, b in zip(labels_h, labels_m)),
        'human_positive_count': sum(labels_h), 'machine_positive_count': sum(labels_m),
        'confusion_counts': dict(Counter(f'{int(a)}:{int(b)}' for a, b in zip(labels_h, labels_m))),
        'selected_sha256': digest(selected), 'human_sha256': digest(human), 'machine_sha256': digest(machine),
        'human_provenance': provenance, 'complete_experiment_receipt': False,
        'automated_human_provenance_proof': False}
