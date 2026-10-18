import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPRINT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPRINT/'scripts'))
import exp11_omnibus_subprocess as m


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=SPRINT/'infrastructure', prefix='omnibus_wrapper_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(m, 'ROOT', self.root); patcher.start(); self.addCleanup(patcher.stop)
        for name in ('scripts/exp11_omnibus_analysis.py', 'scripts/exp11_omnibus_subprocess.py',
                     'infrastructure/exp11_omnibus_plan_v1.json',
                     'infrastructure/omnibus_validation_v1/requirements_lock_v2.txt',
                     '.omnibus-venv/bin/python', 'posterior/invocation.json', 'posterior/component.json'):
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
        self.destination = self.root/'posterior'; self.audit = self.root/'audit'
        (self.destination/'component.json').write_text(json.dumps({'fits': {
            'itt': {'status': 'CONVERGED'}, 'strict': {'status': 'CONVERGED'}}}))
        for name in ('itt', 'strict'):
            folder = self.destination/name; folder.mkdir()
            (folder/'beta.npy').write_bytes(b'synthetic test byte payload; not a numerical artifact')
            (folder/'arrays.json').write_text(json.dumps({'beta.npy': m.sha(folder/'beta.npy')}))
        self.result = {'component': {'synthetic': True}, 'raw_to_sampler_replay_verified': True,
                       'saved_chain_summary_recomputed': True, 'phase3_authorized': False}

    def fake_process(self, command, **kwargs):
        self.command = command; self.kwargs = kwargs
        request_path = Path(command[-1]); request = json.loads(request_path.read_text())
        output = request_path.parent/'result.json'; m.write(output, self.result)
        m.write(output.with_name('result_binding.json'), {'request_sha256': m.sha(request_path),
                'result_sha256': m.sha(output), 'episodes_sha256': request['episodes_sha256'],
                'raw_to_sampler_replay_verified': True})
        class Process:
            pid = 987654321
            def wait(self, **kwargs): return 0
            def poll(self): return 0
        return Process()

    def run_fake(self, side_effect=None):
        with patch.object(m.subprocess, 'Popen', side_effect=side_effect or self.fake_process):
            return m.recompute([{'synthetic': True}], self.destination, audit_directory=self.audit)

    def test_returns_original_scientific_result_and_preserves_exact_request(self):
        self.assertEqual(self.run_fake(), self.result)
        self.assertEqual(self.command[:3], ['taskset', '-c', '0,1'])
        self.assertEqual(self.command[3], str(self.root/'.omnibus-venv/bin/python'))
        self.assertTrue(self.kwargs['start_new_session'])
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(self.kwargs['env'][key], '1')
        self.assertEqual(json.loads((self.audit/'audited_episodes.json').read_text()), [{'synthetic': True}])
        self.assertTrue(json.loads((self.audit/'execution.json').read_text())['fresh_sampler_replay'])
        with self.assertRaises(FileExistsError): self.run_fake()

    def test_parent_import_needs_no_numerical_library(self):
        code = "import sys; sys.path.insert(0,'scripts'); import exp11_omnibus_subprocess; assert not any(k in sys.modules for k in ('numpy','scipy','pymc','pytensor','arviz'))"
        result = subprocess.run([sys.executable, '-c', code], cwd=SPRINT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_changed_input_or_source_before_success_is_rejected(self):
        for target in ('audited_episodes.json', 'source'):
            with self.subTest(target=target):
                self.audit = self.root/('audit_'+target)
                def changed(command, **kwargs):
                    process = self.fake_process(command, **kwargs)
                    path = self.audit/'audited_episodes.json' if target != 'source' else self.root/'scripts/exp11_omnibus_analysis.py'
                    path.write_text('changed')
                    return process
                with self.assertRaises(ValueError): self.run_fake(changed)
                self.assertEqual(json.loads((self.audit/'execution.json').read_text())['status'], 'FAILED')

    def test_changed_result_or_missing_proof_is_rejected(self):
        def changed(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            (self.audit/'result.json').write_text('{}')
            return process
        with self.assertRaises(ValueError): self.run_fake(changed)

    def test_posterior_array_mutated_during_replay_is_rejected(self):
        def changed(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            (self.destination/'itt/beta.npy').write_bytes(b'changed after sampler loaded old bytes')
            return process
        with self.assertRaisesRegex(ValueError, 'Posterior array differs'): self.run_fake(changed)
        self.assertEqual(json.loads((self.audit/'execution.json').read_text())['status'], 'FAILED')

    def test_posterior_manifest_and_array_replaced_together_during_replay_rejected(self):
        def changed(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            path = self.destination/'strict/beta.npy'; path.write_bytes(b'new consistent bytes')
            path.with_name('arrays.json').write_text(json.dumps({'beta.npy': m.sha(path)}))
            return process
        with self.assertRaisesRegex(ValueError, 'binding changed'): self.run_fake(changed)

    def test_missing_array_or_traversal_rejected_before_subprocess(self):
        path = self.destination/'itt/arrays.json'
        for declared in ({'absent.npy': '0'*64}, {'../beta.npy': '0'*64}):
            path.write_text(json.dumps(declared))
            with patch.object(m.subprocess, 'Popen') as spawn:
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.run_fake()
                spawn.assert_not_called()

    def test_process_failure_is_never_success_even_if_output_exists(self):
        def failed(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            process.wait = lambda **kwargs: 7
            return process
        with self.assertRaisesRegex(ValueError, 'code 7'): self.run_fake(failed)
        self.assertEqual(json.loads((self.audit/'execution.json').read_text())['status'], 'FAILED')

    def test_timeout_kills_process_group_and_preserves_failed_audit(self):
        def timeout(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            def wait(**kwargs):
                if kwargs: raise subprocess.TimeoutExpired(command, 1)
                return -9
            process.wait = wait
            return process
        with patch.object(m.os, 'killpg') as kill:
            with self.assertRaises(m.ReplayProcessError): self.run_fake(timeout)
            kill.assert_called_once_with(987654321, m.signal.SIGKILL)
        self.assertEqual(json.loads((self.audit/'execution.json').read_text())['status'], 'FAILED')

    def test_incomplete_replay_cannot_be_returned(self):
        self.result['raw_to_sampler_replay_verified'] = False
        with self.assertRaises(ValueError): self.run_fake()

    def test_path_escape_overlap_and_timeout_bounds_rejected_before_spawn(self):
        for folder in (self.root.parent/'escape', self.destination/'audit', self.root):
            with self.subTest(folder=folder), self.assertRaises(ValueError):
                m.recompute([], self.destination, audit_directory=folder)
        for timeout in (0, 3601, True, 1.5):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                m.recompute([], self.destination, audit_directory=self.audit, timeout_seconds=timeout)

    def test_environment_removes_inherited_python_injection(self):
        with patch.dict(m.os.environ, {'PYTHONPATH': '/fake', 'PYTHONHOME': '/fake'}):
            value = m.environment()
        self.assertNotIn('PYTHONPATH', value); self.assertNotIn('PYTHONHOME', value)
        self.assertEqual(value['PYTHONNOUSERSITE'], '1')

    def test_bootstrap_matches_complete_initialized_environment_without_relaxation(self):
        expected = {'packages': {'numerical': '1', 'bundled': '2'}}
        (self.destination/'invocation.json').write_text(json.dumps({'environment': expected}))
        values = iter([{'packages': {'numerical': '1'}}, expected])
        result = m.bootstrap_environment(SimpleNamespace(environment=lambda: next(values)), self.destination)
        self.assertEqual(result['initialized'], expected)
        values = iter([expected, {'packages': {'numerical': '1'}}])
        with self.assertRaisesRegex(ValueError, 'environment differs'):
            m.bootstrap_environment(SimpleNamespace(environment=lambda: next(values)), self.destination)


if __name__ == '__main__':
    unittest.main()
