"""Shared raw generation evidence audits; no experiment completion dispatch.

Extracted verbatim from verify_all_receipts; persistence and byte identity only.
"""
import json


def requires_return_evidence(row):
    return (row.get('observation_type') in {'exp23_authoring','structured_record','exp11_respondent_probe'}
            or row.get('observation_type')=='tool_turn' and 'payload_sha256' in row
            or 'generation_return_record_sha256' in row)


def audit_return_evidence(root, request_path, requests, completed):
    """Bind every prepared return to its journal and parsed completion.

    Older Phase1 streams without return-chain links retain their original audit.
    This checks persistence and byte identity, not experiment completeness.
    """
    if not any(requires_return_evidence(r) for r in completed):return None
    from exp23_author_worker import read_chain
    assert requests, 'Linked completions require a writeahead journal'
    anchor=requests[0]['payload_sha256']
    read_chain(request_path,anchor)
    completion_path=request_path.with_name(request_path.name.removesuffix('_requests.jsonl')+'.jsonl')
    read_chain(completion_path,anchor)
    kinds={r.get('observation_type') for r in completed}
    assert len(kinds)==1, 'Mixed return evidence formats in one stream'
    kind=next(iter(kinds))
    folders={'exp23_authoring':'author_artifacts','structured_record':'structured_artifacts',
             'exp11_respondent_probe':'probe_artifacts'}
    if kind in folders:
        return_path=root/folders[kind]/request_path.parent.name/'model_returns.jsonl'
    else:
        assert kind=='tool_turn', 'Unsupported return evidence format'
        return_path=request_path.with_name(request_path.name.removesuffix('_requests.jsonl')+'_returns.jsonl')
    batches,_=read_chain(return_path,anchor)
    journal={r['trial_id']:r for r in requests};raw={r['trial_id']:r for r in completed};seen=set()
    for batch in batches:
        ids=batch['trial_ids'];outputs=batch['outputs'];links=batch['request_record_sha256']
        assert len(ids)==len(outputs)==len(links) and ids, 'Returned generation batch shape differs'
        for index,(trial_id,output,link) in enumerate(zip(ids,outputs,links)):
            assert trial_id not in seen, 'Duplicate returned generation'
            seen.add(trial_id);written=journal[trial_id];row=raw[trial_id]
            assert link==written['chain_sha256']==row['request_record_sha256'], 'Returned writeahead link differs'
            assert row['generation_return_record_sha256']==batch['chain_sha256'], 'Raw return link differs'
            if kind=='exp23_authoring':
                choices=output
                assert batch['full_prompts'][index]==row['full_prompt'], 'Author return prompt differs'
            else:
                assert row['return_index']==index, 'Return position differs'
                assert output['prompt_token_ids']==written['prompt_token_ids']==row['prompt_token_ids'], 'Return prompt tokens differ'
                choices=output['choices']
            assert choices==[{'raw_output':row['raw_output'],'token_ids':row['generated_token_ids'],
                             'finish_reason':row['finish_reason']}], 'Raw completion differs from unparsed return'
    assert seen==set(raw), 'Completion missing from preserved return batches'
    return {'return_path':str(return_path.relative_to(root)),'batches':len(batches),'returned_generations':len(seen)}


def audit_request_journal(path, root=None):
    """Every returned generation must match its persisted, pre-generation prompt."""
    assert path.name.endswith('_requests.jsonl')
    completed_path=path.with_name(path.name.removesuffix('_requests.jsonl')+'.jsonl')
    with path.open() as handle:
        requests=[json.loads(line) for line in handle]
    completed=[]
    if completed_path.exists():
        with completed_path.open() as handle:
            completed=[json.loads(line) for line in handle]
    request_by_id={r['trial_id']:r for r in requests}
    returned_by_id={r['trial_id']:r for r in completed}
    assert len(request_by_id)==len(requests), 'Duplicate journal trial ID'
    assert len(returned_by_id)==len(completed), 'Duplicate completion trial ID'
    assert returned_by_id.keys()<=request_by_id.keys(), 'Completion has no write-ahead prompt'
    for trial_id,row in returned_by_id.items():
        request=request_by_id[trial_id]
        for field in ('full_prompt','model_arm','split','max_new_tokens','temperature','forced_prefix'):
            assert row[field]==request[field], f'Prompt journal/completion mismatch: {trial_id} {field}'
        for field in ('prompt_token_ids','tokenizer_file_sha256','source_manifest','source_manifest_sha256',
                      'request_sha256','payload_sha256','model_id','model_revision','family'):
            if field in request:assert row[field]==request[field], f'Journal source binding differs: {trial_id} {field}'
    result={'requested':len(requests),'returned':len(completed),
            'pending_trial_ids':sorted(request_by_id.keys()-returned_by_id.keys())}
    if root is not None:
        returns=audit_return_evidence(root,path,requests,completed)
        if returns is not None:result['return_evidence']=returns
    return result

