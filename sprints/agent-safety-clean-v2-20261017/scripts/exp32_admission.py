"""Fail-closed future empirical admission; never a saved-PASS shortcut."""
from pathlib import Path
import importlib
from exp32_plan import digest, require


def recompute(root,inputs):
    """Expected future capture auditor is deliberately not fabricated here.

    This function performs no GPU work. Once upstream implementations exist,
    it requires fresh global/readout/capture recomputations. The capture auditor
    must independently replay source requests, first-turn journals, tokenizer
    bytes, checkpoint inventory, five-pilot evidence, block/token extraction and
    all persisted NPZ row mappings, and return backend_audit_complete=True.
    """
    root=Path(root).resolve()
    require(set(inputs)=={'phase3_inputs','exp11_selection','exp31_selection','capture_selection'},'Exact Exp32 admission inputs required')
    require(inputs['phase3_inputs']['exp11_selection']==inputs['exp11_selection'],'Exp11 selection differs from global gate')
    result={'schema':'exp32_preparation_admission_v1','inputs':inputs,'authorized':False,
            'phase3_authorized':False,'empirical_analysis_complete':False}
    modules={}
    for name in ('exp31_evidence','exp32_capture_evidence'):
        if not (root/'scripts'/f'{name}.py').is_file():
            return {**result,'state':'PENDING_INDEPENDENT_BACKEND_AUDITOR','missing_module':name}
        modules[name]=importlib.import_module(name)
        require(Path(modules[name].__file__).resolve()==root/'scripts'/f'{name}.py','Auditor outside clean sprint')
    from phase3_readiness import recompute as global_audit
    gate=global_audit(root,inputs['phase3_inputs'])
    if gate['phase3_authorized'] is not True:return {**result,'state':'GLOBAL_PHASE3_NOT_AUTHORIZED','global_audit':gate}
    require(gate['complete_phase2_primary_scope'] is True,'Complete primary Phase2 required')
    from exp11_evidence import recompute as audit_exp11
    exp11=audit_exp11(root,inputs['exp11_selection'])
    exp31=modules['exp31_evidence'].recompute(root,inputs['exp31_selection'])
    require(exp11['execution_evidence_complete'] is True and len(exp11['episodes'])==4520,'Complete independently audited Exp11 required')
    require(exp31['execution_evidence_complete'] is True and len(exp31['episodes'])==1536,'Complete independently audited Exp31 required')
    require(exp31['complete_experiment_receipt'] is True and exp31['raw_to_summary_recomputed'] is True,
            'Exp31 execution alone is not full experimental completion')
    capture=modules['exp32_capture_evidence'].recompute(root,inputs['capture_selection'],exp11,exp31)
    require(capture['backend_audit_complete'] is True and capture['pilot_H5_passed'] is True and
            capture['source_coverage']=={'exp11':4520,'exp31':1536},'Complete real capture backend/pilot provenance required')
    return {**result,'authorized':True,'phase3_authorized':True,'state':'AUTHORIZED_FOR_ANALYSIS_ONLY',
            'global_audit_sha256':digest(gate),'exp11_audit_sha256':digest(exp11),
            'exp31_audit_sha256':digest(exp31),'capture_audit_sha256':digest(capture),
            'empirical_analysis_complete':False,'complete_experiment_receipt':False}
