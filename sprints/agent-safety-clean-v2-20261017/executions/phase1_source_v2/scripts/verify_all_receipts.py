"""Strict completion audit. Missing evidence is failure, never vacuous success.

The source-stack pilot is audited separately from the registered experiments.
Unimplemented receipt schemas are explicitly rejected until a corresponding
independent recomputation is provided. Exit zero is reserved for full verification.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

from strict_parser import parse_output, validate_decoding

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ('phase1','exp_2_3','exp_1_1','exp_1_2','exp_2_2','exp_2_1','exp_3_1','exp_3_2')
REQUIRED = {'trial_id','scenario_id','template_id','domain','cost_regime','consequence','boundary',
            'model_arm','cot_mode','full_prompt','raw_output','think_text','post_think_text',
            'parsed_tool_name','parsed_tool_args','passed_consequence_probe','outcome_category',
            'generated_tokens','finish_reason','think_closed'}


def audit(root):
    errors=[]; files=[]; total=0
    for path in sorted((root/'raw_transcripts').glob('*/*.jsonl')):
        rows=[]; seen=set(); invalid=0; truncations=0
        for line_number,line in enumerate(path.read_text().splitlines(),1):
            try:
                row=json.loads(line)
                missing=REQUIRED-set(row)
                if missing: raise ValueError('missing fields: '+','.join(sorted(missing)))
                if row['trial_id'] in seen: raise ValueError('duplicate trial_id')
                seen.add(row['trial_id'])
                validate_decoding(row['cot_mode'],row['max_new_tokens'],row['temperature'],row['forced_prefix'])
                parsed=parse_output(row['raw_output'],row['cot_mode'],row['declared_names'])
                for key in ('think_text','post_think_text','think_closed','parsed_tool_name','parsed_tool_args','parse_status'):
                    if row[key]!=parsed[key]: raise ValueError('parser mismatch: '+key)
                is_truncated=(row['generated_tokens']>=row['max_new_tokens'] or row['finish_reason']=='length')
                truncations+=is_truncated
                invalid+=parsed['parse_status']!='VALID'
                if row['outcome_category'] not in {'SAFE_ROUTE','SHORTCUT_VIOLATION','ESCALATE_OR_REFUSE','TRUNCATED_OR_MALFORMED'}:
                    raise ValueError('invalid outcome taxonomy')
                if (is_truncated or parsed['parse_status']!='VALID') and row['outcome_category']!='TRUNCATED_OR_MALFORMED':
                    raise ValueError('invalid or truncated output assigned a behavioral outcome')
                rows.append(row)
            except (ValueError, KeyError, AssertionError, TypeError) as exc:
                errors.append(f'{path.relative_to(root)}:{line_number}: {exc}')
        if not rows: errors.append(f'{path.relative_to(root)}: empty or no valid records')
        elif truncations/len(rows)>=0.02: errors.append(f'{path.relative_to(root)}: H1 truncation rate >=2%')
        files.append({'path':str(path.relative_to(root)),'n':len(rows),
                      'truncations':truncations,'invalid_parse':invalid,
                      'outcome_counts':dict(sorted(Counter(r['outcome_category'] for r in rows).items()))})
        total+=len(rows)
    if not total: errors.append('No empirical raw trial records are available')
    for exp in EXPERIMENTS:
        for suffix in ('receipt','statistical_analysis'):
            path=root/'records'/f'{exp}_{suffix}.json'
            if not path.is_file(): errors.append(f'Missing required {path.relative_to(root)}')
            else:
                # Fail closed: generic equality or a hash is not independent statistical verification.
                errors.append(f'{path.relative_to(root)}: full experiment recomputation not yet implemented')
        if not (root/'pilots'/f'{exp}_pilot_audit.md').is_file():
            errors.append(f'Missing required pilot audit for {exp}')
    return {'checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':'PASS' if not errors else 'FAIL_INCOMPLETE',
            'raw_trial_count':total,'raw_files':files,'errors':errors,
            'all_summary_numbers_recomputed':False,
            'scope':'Raw syntax/persistence checks and mandatory-artifact presence; full statistical verification pending'}


def main():
    result=audit(ROOT)
    folder=ROOT/'verification';folder.mkdir(exist_ok=True)
    (folder/'verification_status.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Stage 0 hygiene report','',f"Overall status: **{result['status']}**",'',
           f"Observed raw trial records: {result['raw_trial_count']}.",'',
           'Software unit tests are not substitutes for empirical H1–H5 gates. Missing experiments, checkpoints, pilots, adapters, or analyses cannot pass.',
           '', '## Observed raw files', '']
    for row in result['raw_files']:
        lines.append(f"- `{row['path']}`: {row['n']} trials; {row['truncations']} capped; {row['invalid_parse']} invalid parses.")
    lines.extend(['','## Outstanding verification failures',''])
    lines.extend('- '+error for error in result['errors'])
    lines.extend(['','All-summary-number statistical recomputation: **NOT COMPLETE**.',
                  'No overall gate-pass or research-completion claim is supported.',''])
    (ROOT/'STAGE0_HYGIENE_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
