from native_comparison_v1 import parse,select,tokenizer

def test_selection_is_paired_and_fixed():
    worlds,plan=select()
    assert len(worlds)==24 and len(plan)==576
    assert sum(x['pilot'] for x in plan)==48
    assert all(sum(r['world_id']==w['world_id'] for r in plan)==24 for w in worlds)
    assert all(w['split'] in ['test','family_test'] for w in worlds)

def test_native_answer_boundary_and_completion():
    tok=tokenizer();eos=tok.eos_token_id
    ids=tok.encode('<think>Consider the allowed plan.</think>\n A',add_special_tokens=False)+[eos]
    good=parse({'generated_token_ids':ids,'ended_with_eos':True},tok)
    assert good['valid'] and good['action']=='A'
    assert not parse({'generated_token_ids':ids[:-1],'ended_with_eos':False},tok)['valid']
    assert not parse({'generated_token_ids':[32,eos],'ended_with_eos':True},tok)['valid']
    assert not parse({'generated_token_ids':[151668]+ids,'ended_with_eos':True},tok)['valid']
    assert parse({'generated_token_ids':ids,'ended_with_eos':True},tok,limit=len(ids))['valid']
    assert not parse({'generated_token_ids':ids,'ended_with_eos':True},tok,limit=len(ids)-1)['valid']
    extra=tok.encode('<think>x</think>A because it is permitted.',add_special_tokens=False)+[eos]
    assert not parse({'generated_token_ids':extra,'ended_with_eos':True},tok)['valid']
