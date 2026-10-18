"""Synthetic Modal visibility tests; all subprocess calls mocked, no cloud work."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import launch_bounded_job as launcher

SOURCE="""@app.function(gpu='H100', cpu=(1,1), memory=(1024,1024), timeout=60,
 startup_timeout=30, retries=0, max_containers=1, scaledown_window=2)
def worker(payload): pass
"""


class AppVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        (self.root/'scripts').mkdir();(self.root/'infrastructure').mkdir()
        (self.root/'scripts/synthetic.py').write_text(SOURCE)
        self.app_id='ap-SyntheticStopped'
        self.observation={'app_id':self.app_id,'description':launcher.NS+'-synthetic',
            'lifecycle':{'state':'stopped','stopped_at':'2026-10-18T00:00:00+00:00'}}
        self.state={'limits':{'gpus':1,'single_gpu_workers':1,'tp8_workers':1},
            'jobs':[{'job':'synthetic_old','app_id':self.app_id,'gpus':8,'single_gpu_workers':0,
                'tp8_workers':1,'log':'infrastructure/synthetic_old.log'}]}
        self.budget={'ceiling_usd':500,'reserved_usd':2,
            'reservations':[{'job':'synthetic_old','upper_bound_usd':2}]}
        self.state_path=self.root/'infrastructure/resource_allocations.json'
        self.budget_path=self.root/'infrastructure/budget_reservations.json'
        self.state_path.write_text(json.dumps(self.state));self.budget_path.write_text(json.dumps(self.budget))
        self.config={'job':'synthetic_new','gpus':1,'single_gpu_workers':1,'tp8_workers':0,
            'log':'infrastructure/synthetic_new.log','upper_bound_usd':1,
            'command':['SYNTHETIC_NEVER_EXECUTE','run','scripts/synthetic.py'],
            'resource_profile':{'source':'scripts/synthetic.py','source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),
                'functions':{'worker':{'max_calls':1,'limits':launcher._decorators(SOURCE)['worker']}},
                'excluded_functions':{},'call_bound_basis':'Synthetic single bounded call, never executed.'}}

    def mock_cli(self,observation,apps=None):
        def run(command,**kwargs):
            self.assertEqual(kwargs,{'cwd':self.root,'text':True,'timeout':60})
            if command==[str(self.root/'.venv/bin/modal'),'app','list','--json']:
                return json.dumps([] if apps is None else apps)
            self.assertEqual(command,[str(self.root/'.venv/bin/modal'),'app','info',self.app_id,'--json'])
            return json.dumps(observation)
        return run

    def test_exact_stopped_owned_app_is_observed_without_mutation(self):
        with patch.object(launcher.subprocess,'check_output',side_effect=self.mock_cli(self.observation)) as cli:
            result=launcher.observe_stopped_app(self.app_id,self.root)
        self.assertEqual(result,self.observation)
        self.assertEqual(cli.call_count,1)
        self.assertEqual(json.loads(self.state_path.read_text()),self.state)
        self.assertEqual(json.loads(self.budget_path.read_text()),self.budget)

    def test_absent_known_stopped_app_releases_only_after_individual_confirmation(self):
        with patch.object(launcher,'ROOT',self.root),\
             patch.object(launcher.subprocess,'check_output',side_effect=self.mock_cli(self.observation)) as cli,\
             patch.object(launcher.subprocess,'Popen') as popen:
            popen.return_value.pid=99999999
            result=launcher.launch(self.config)
        self.assertEqual(cli.call_count,2);popen.assert_called_once()
        self.assertEqual(result['total_gpu_bound'],1)
        state=json.loads(self.state_path.read_text());old=state['jobs'][0]
        self.assertEqual(old['allocation_status'],'released_after_modal_stop')
        self.assertEqual(old['individual_stopped_observation']['observation'],self.observation)
        self.assertEqual(old['individual_stopped_observation']['source'],'modal app info --json')
        self.assertTrue(old['individual_stopped_observation']['checked_utc'])
        self.assertEqual(json.loads(self.budget_path.read_text())['reserved_usd'],3)

    def bad_observations(self):
        values=[]
        for label,patch_value in [('wrong_id',{'app_id':'ap-SomeoneElse'}),
              ('non_owned',{'description':'different-sprint'}),('missing_description',{'description':None}),
              ('active',{'lifecycle':{'state':'running','stopped_at':'2026-10-18'}}),
              ('unknown',{'lifecycle':{'state':'unknown','stopped_at':'2026-10-18'}}),
              ('missing_stop_date',{'lifecycle':{'state':'stopped'}}),
              ('empty_stop_date',{'lifecycle':{'state':'stopped','stopped_at':''}}),
              ('invalid_lifecycle',{'lifecycle':None})]:
            values.append((label,{**deepcopy(self.observation),**patch_value}))
        values.extend([('empty_object',{}),('list_response',[]),('null_response',None)])
        return values

    def test_bad_individual_observations_fail_closed(self):
        for label,value in self.bad_observations():
            with self.subTest(label=label),patch.object(launcher.subprocess,'check_output',side_effect=self.mock_cli(value)):
                with self.assertRaises(ValueError):launcher.observe_stopped_app(self.app_id,self.root)

    def test_invalid_fallback_never_reserves_or_launches(self):
        before=(self.state_path.read_bytes(),self.budget_path.read_bytes())
        for label,value in self.bad_observations():
            with self.subTest(label=label),patch.object(launcher,'ROOT',self.root),\
                 patch.object(launcher.subprocess,'check_output',side_effect=self.mock_cli(value)),\
                 patch.object(launcher.subprocess,'Popen') as popen:
                with self.assertRaises(ValueError):launcher.launch(self.config)
                popen.assert_not_called()
            self.assertEqual((self.state_path.read_bytes(),self.budget_path.read_bytes()),before)

    def test_currently_listed_stopped_app_does_not_trigger_fallback(self):
        listed={'app_id':self.app_id,'description':launcher.NS+'-synthetic','state':'stopped'}
        with patch.object(launcher,'ROOT',self.root),\
             patch.object(launcher.subprocess,'check_output',side_effect=self.mock_cli(None,[listed])) as cli,\
             patch.object(launcher,'observe_stopped_app') as fallback,\
             patch.object(launcher.subprocess,'Popen') as popen:
            popen.return_value.pid=99999999
            launcher.launch(self.config)
        cli.assert_called_once();fallback.assert_not_called();popen.assert_called_once()

    def test_currently_live_owned_app_remains_allocated_without_fallback(self):
        listed={'app_id':self.app_id,'description':launcher.NS+'-synthetic','state':'running'}
        with patch.object(launcher,'ROOT',self.root),\
             patch.object(launcher.subprocess,'check_output',side_effect=self.mock_cli(None,[listed])),\
             patch.object(launcher,'observe_stopped_app') as fallback,\
             patch.object(launcher.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(ValueError,'capacity exceeded'):launcher.launch(self.config)
        fallback.assert_not_called();popen.assert_not_called()
        self.assertEqual(json.loads(self.budget_path.read_text()),self.budget)

    def test_invalid_id_rejected_before_any_cli_call(self):
        for value in ('ap-bad/id','ap-ID --force','',None):
            with self.subTest(value=value),patch.object(launcher.subprocess,'check_output') as cli:
                with self.assertRaises(ValueError):launcher.observe_stopped_app(value,self.root)
                cli.assert_not_called()

    def test_cli_failure_timeout_and_invalid_json_fail_closed(self):
        for error in (subprocess.CalledProcessError(1,['SYNTHETIC']),
                      subprocess.TimeoutExpired(['SYNTHETIC'],60)):
            with self.subTest(error=type(error).__name__),patch.object(launcher.subprocess,'check_output',side_effect=error):
                with self.assertRaises(type(error)):launcher.observe_stopped_app(self.app_id,self.root)
        with patch.object(launcher.subprocess,'check_output',return_value='not JSON'):
            with self.assertRaises(ValueError):launcher.observe_stopped_app(self.app_id,self.root)


if __name__=='__main__':unittest.main()
