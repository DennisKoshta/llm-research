import pytest
from semantic_names_v1 import selection,swap_names

def test_simultaneous_names_preserve_physical_codes_and_other_numbers():
    text='Plan 1: ALLOW. Plan 2: BLOCK. Plan 10 is unrelated. Time 12. Action menu: A = Plan 1; B = Plan 2.'
    swapped=swap_names(text)
    assert swapped=='Plan 2: ALLOW. Plan 1: BLOCK. Plan 10 is unrelated. Time 12. Action menu: A = Plan 2; B = Plan 1.'
    assert swap_names(swapped)==text
    with pytest.raises(AssertionError):swap_names('__EXP6_NAME_ONE_SENTINEL__')

def test_fixed_world_factorial_selection():
    worlds,plan=selection()
    assert len(worlds)==12 and len(plan)==1248
    assert all(sum(x['world_id']==w['world_id'] for x in plan)==104 for w in worlds)
    assert all(w['split'] in ['test','family_test'] for w in worlds)
