import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('exp6strict',Path(__file__).with_name('strict_scoring_v2.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_integrity_separates_parse_from_execution_and_keeps_eos_at_limit():
    assert m.completion({'raw':'A','generated_token_ids':[1]*16,'ended_with_eos':True})['valid']
    a=m.completion({'raw':'A','generated_token_ids':[1]*16,'ended_with_eos':False});assert a['parse_valid'] and a['capped'] and not a['valid']
    b=m.completion({'raw':'B','generated_token_ids':[1],'ended_with_eos':False});assert b['parse_valid'] and not b['capped'] and not b['valid']
    assert not m.completion({'raw':'AB','generated_token_ids':[1,2],'ended_with_eos':True})['valid']
    assert not m.completion({})['valid']
