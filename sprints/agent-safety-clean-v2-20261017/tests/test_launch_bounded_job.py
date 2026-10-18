"""Admission tests use local synthetic files and mocks; never launch cloud work."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib,json,os
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import launch_bounded_job as launcher
import reconcile_stopped_job_bounds as reconciler

SOURCE="""@app.function(gpu='H100', cpu=(4,4), memory=(65536,65536), timeout=1800,
 startup_timeout=300, retries=0, max_containers=2, scaledown_window=2)
def worker(payload): pass
@app.function(cpu=(1,1), memory=(1024,1024), timeout=7200,
 retries=0, max_containers=1, scaledown_window=2)
def coordinate(payload): pass
"""


class LaunchAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        (self.root/'scripts').mkdir();(self.root/'infrastructure').mkdir()
        self.source=self.root/'scripts/fixture.py';self.source.write_text(SOURCE)
        functions=launcher._decorators(SOURCE)
        self.config={'job':'fixture','gpus':2,'single_gpu_workers':2,'tp8_workers':0,
          'log':'infrastructure/fixture.log','upper_bound_usd':30,
          'command':['modal','run','--detach','scripts/fixture.py'],
          'resource_profile':{'source':'scripts/fixture.py','source_sha256':hashlib.sha256(self.source.read_bytes()).hexdigest(),
          'functions':{name:{'max_calls':6 if name=='worker' else 1,'limits':limits} for name,limits in functions.items()},
          'excluded_functions':{},'call_bound_basis':'Synthetic fixture: six worker calls and one durable coordinator.'}}
        self.state={'limits':{'gpus':10,'single_gpu_workers':4,'tp8_workers':1},'jobs':[]}
        self.budget={'ceiling_usd':500,'reserved_usd':10,'reservations':[{'job':'earlier','upper_bound_usd':10}]}
        self.write_ledgers()

    def write_ledgers(self):
        (self.root/'infrastructure/resource_allocations.json').write_text(json.dumps(self.state))
        (self.root/'infrastructure/budget_reservations.json').write_text(json.dumps(self.budget))

    def test_profile_audits_defaults_cpu_coordinator_and_bound(self):
        audit=launcher.validate_resource_profile(self.config,self.root)
        self.assertEqual(audit['whole_app_allocation']['gpus'],2)
        coordinator=next(x for x in audit['function_bounds'] if x['function']=='coordinate')
        self.assertEqual(coordinator['seconds_per_call_bound'],7200+7200+120)
        self.assertEqual(audit['minimum_compute_reservation_usd'],18)

    def test_hash_decorator_and_underreservation_tampering_fail(self):
        for mutation in ('hash','decorator','budget','missing_cpu'):
            with self.subTest(mutation=mutation):
                cfg=deepcopy(self.config)
                if mutation=='hash':cfg['resource_profile']['source_sha256']='0'*64
                if mutation=='decorator':cfg['resource_profile']['functions']['worker']['limits']['timeout']=10
                if mutation=='budget':cfg['upper_bound_usd']=1
                if mutation=='missing_cpu':del cfg['resource_profile']['functions']['coordinate']
                with self.assertRaises(ValueError):launcher.validate_resource_profile(cfg,self.root)

    def test_named_unused_function_requires_explicit_exclusion(self):
        cfg=deepcopy(self.config)
        cfg['resource_profile']['excluded_functions']={'coordinate':'Synthetic example: command invokes worker directly.'}
        del cfg['resource_profile']['functions']['coordinate']
        self.assertEqual(launcher.validate_resource_profile(cfg,self.root)['whole_app_allocation']['gpus'],2)
        cfg['resource_profile']['excluded_functions']['coordinate']=''
        with self.assertRaises(ValueError):launcher.validate_resource_profile(cfg,self.root)

    def test_invalid_counts_and_per_app_limits(self):
        self.assertEqual(launcher.validate_counts({'gpus':4,'single_gpu_workers':4,'tp8_workers':0},True)['gpus'],4)
        for values in ((-1,-1,0),(True,1,0),(5,5,0),(16,0,2),(9,1,1)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):launcher.validate_counts(dict(zip(launcher.RESOURCE_KEYS,values)),True)

    def test_cpu_only_profile_and_warm_pool_rejection(self):
        cfg=deepcopy(self.config)
        cfg.update(gpus=0,single_gpu_workers=0)
        del cfg['resource_profile']['functions']['worker']
        cfg['resource_profile']['excluded_functions']={'worker':'CPU-only fixture entry point does not call the worker.'}
        self.assertEqual(launcher.validate_resource_profile(cfg,self.root)['whole_app_allocation']['gpus'],0)
        with self.assertRaisesRegex(ValueError,'warm pools'):
            launcher._decorators(SOURCE.replace("gpu='H100',","gpu='H100', min_containers=1,"))

    def test_actual_capture_logits_selects_named_worker(self):
        source=Path(__file__).resolve().parents[1]/'scripts/modal_qualify.py'
        limits=launcher._decorators(source.read_text())
        self.assertEqual(limits['capture_logits']['timeout'],3600)
        self.assertEqual(limits['qualify']['timeout'],7200)
        cfg=deepcopy(self.config)
        cfg.update(gpus=1, single_gpu_workers=1, upper_bound_usd=40)
        self.source.write_text(source.read_text())
        profile=cfg['resource_profile']
        profile['source_sha256']=hashlib.sha256(self.source.read_bytes()).hexdigest()
        profile['functions']={name:{'max_calls':6 if name=='capture_logits' else 1,'limits':limits[name]}
                              for name in ('capture_logits','run_group')}
        profile['excluded_functions']={'qualify':'stage=logits selects capture_logits remotely; qualify is called only with .local.'}
        audit=launcher.validate_resource_profile(cfg,self.root)
        self.assertEqual(audit['whole_app_allocation']['gpus'],1)

    def test_budget_sum_nonfinite_and_duplicate_integrity(self):
        for mutation in ('stale','nan','duplicate','ceiling'):
            budget=deepcopy(self.budget)
            if mutation=='stale':budget['reserved_usd']=0
            if mutation=='nan':budget['reservations'][0]['upper_bound_usd']=float('nan')
            if mutation=='duplicate':budget['reservations']*=2;budget['reserved_usd']=20
            if mutation=='ceiling':budget['ceiling_usd']=501
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):launcher.budget_total(budget)

    def test_shared_lock_excludes_reconciliation(self):
        with patch.object(reconciler,'ROOT',self.root),patch.object(reconciler,'reconcile_locked') as action:
            with launcher.launch_authority_lock(self.root):
                with self.assertRaises(BlockingIOError):reconciler.main()
            action.assert_not_called()
            reconciler.main();action.assert_called_once()

    def test_unresolved_detached_launch_and_unknown_app_remain_blocking(self):
        for row in ({'job':'old','gpus':0,'single_gpu_workers':0,'tp8_workers':0,'log':'infrastructure/missing.log'},
                    {'job':'old','gpus':0,'single_gpu_workers':0,'tp8_workers':0,'log':'infrastructure/missing.log','app_id':'ap-unknown'}):
            self.state['jobs']=[row];self.write_ledgers()
            with patch.object(launcher,'ROOT',self.root),patch.object(launcher.subprocess,'check_output',return_value='[]') as apps,patch.object(launcher.subprocess,'Popen') as popen:
                with self.assertRaises(ValueError):launcher.launch(self.config)
                popen.assert_not_called();self.assertEqual(apps.call_args.kwargs['timeout'],60)
            self.assertEqual(json.loads((self.root/'infrastructure/budget_reservations.json').read_text()),self.budget)

    def test_unregistered_owned_app_blocks_launch(self):
        app={'app_id':'ap-unregistered','description':launcher.NS+'-fixture','state':'running'}
        with patch.object(launcher,'ROOT',self.root),patch.object(launcher.subprocess,'check_output',return_value=json.dumps([app])),patch.object(launcher.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(ValueError,'Unregistered'):launcher.launch(self.config)
            popen.assert_not_called()

    def test_active_existing_bound_blocks_overallocation(self):
        row={'job':'old','app_id':'ap-existing','gpus':8,'single_gpu_workers':0,'tp8_workers':1,'log':'infrastructure/old.log'}
        self.state['jobs']=[row];self.state['limits']['gpus']=8;self.write_ledgers()
        app={'app_id':'ap-existing','description':launcher.NS+'-fixture','state':'running'}
        with patch.object(launcher,'ROOT',self.root),patch.object(launcher.subprocess,'check_output',return_value=json.dumps([app])),patch.object(launcher.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(ValueError,'capacity exceeded'):launcher.launch(self.config)
            popen.assert_not_called()

    def test_popen_failure_keeps_reserved_unresolved_state(self):
        with patch.object(launcher,'ROOT',self.root),patch.object(launcher.subprocess,'check_output',return_value='[]'),patch.object(launcher.subprocess,'Popen',side_effect=OSError('synthetic launch failure')):
            with self.assertRaises(OSError):launcher.launch(self.config)
        budget=json.loads((self.root/'infrastructure/budget_reservations.json').read_text())
        state=json.loads((self.root/'infrastructure/resource_allocations.json').read_text())
        self.assertEqual(budget['reserved_usd'],40)
        self.assertEqual(state['jobs'][-1]['allocation_status'],'reserved_pending_launch')
        self.assertNotIn('launch_pid',state['jobs'][-1])

    def test_changed_frozen_stimulus_blocks_before_reservation(self):
        stimulus=self.root/'stimulus.json';stimulus.write_text('{"condition":"fixed"}')
        self.config['frozen_files']={'stimulus.json':hashlib.sha256(stimulus.read_bytes()).hexdigest()}
        def change_during_remote_poll(*args,**kwargs):
            stimulus.write_text('{"condition":"changed"}')
            return '[]'
        with patch.object(launcher,'ROOT',self.root),patch.object(launcher.subprocess,'check_output',side_effect=change_during_remote_poll),patch.object(launcher.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(ValueError,'Frozen prerequisite changed'):
                launcher.launch(self.config)
            popen.assert_not_called()
        self.assertEqual(json.loads((self.root/'infrastructure/budget_reservations.json').read_text()),self.budget)
        self.assertEqual(json.loads((self.root/'infrastructure/resource_allocations.json').read_text()),self.state)


if __name__=='__main__':unittest.main()
