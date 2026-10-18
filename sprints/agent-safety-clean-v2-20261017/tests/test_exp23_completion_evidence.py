"""Completed-call evidence is stricter than admitted first-launch overlap evidence."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from exp23_statistical_analysis import audit_completed_call


class CompletedBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = {'experiment': 'exp23_executor_unit_fixture', 'payload_sha256': 'a'*64}
        self.path = 'group_calls/exp23_executor_unit_fixture/full.json'
        self.start = self.root/'group_calls/exp23_executor_unit_fixture/full_worker_started.json'
        self.start.parent.mkdir(parents=True)
        self.ledger = {'schema': 'exp23_executor_call_v1', 'state': 'SPAWNED', 'stage': 'full',
                       'payload_sha256': 'a'*64, 'worker_module': 'modal_exp23_executor',
                       'worker_function': 'execute', 'gpu': 'H200:1', 'call_id': 'fc-UnitFull'}
        self.started = {'call_id': 'fc-UnitFull', 'payload_sha256': 'a'*64, 'stage': 'full'}
        (self.root/self.path).write_text(json.dumps(self.ledger))
        self.start.write_text(json.dumps(self.started))
        self.backend = {'call_id': 'fc-UnitFull', 'function_name': 'execute',
                        'module_name': 'modal_exp23_executor', 'status': 'SUCCESS'}

    def audit(self, backend=None):
        return audit_completed_call(self.root, self.payload, self.path,
                                    lookup_call=lambda call_id: deepcopy(self.backend if backend is None else backend))

    def test_success_requires_concrete_matching_worker_and_backend(self):
        result = self.audit()
        self.assertTrue(result['backend_success_verified'])
        self.assertTrue(result['worker_start_verified'])
        self.assertEqual(result['full_call_id'], self.backend['call_id'])
        self.assertEqual(len(result['worker_start_sha256']), 64)

    def test_pending_running_failed_or_missing_backend_never_complete(self):
        for status in ('PENDING', 'RUNNING', 'FAILURE', 'TIMEOUT', None):
            with self.subTest(status=status):
                backend = {**self.backend, 'status': status}
                with self.assertRaisesRegex(ValueError, 'not completed successfully'): self.audit(backend)

    def test_missing_start_is_required_even_after_backend_success(self):
        self.start.unlink()
        with self.assertRaises(FileNotFoundError): self.audit()

    def test_mismatched_start_and_extra_start_fields_rejected(self):
        for key, value in [('call_id', 'fc-Other'), ('payload_sha256', 'b'*64), ('stage', 'pilot'), ('extra', True)]:
            with self.subTest(key=key):
                self.start.write_text(json.dumps({**self.started, key: value}))
                with self.assertRaisesRegex(ValueError, 'worker-start identity'): self.audit()

    def test_wrong_backend_role_or_call_rejected(self):
        for key, value in [('call_id', 'fc-Other'), ('function_name', 'records'), ('module_name', 'modal_structured_records')]:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, 'not completed successfully'): self.audit({**self.backend, key: value})

    def test_start_or_ledger_mutation_during_lookup_rejected(self):
        for target in (self.start, self.root/self.path):
            with self.subTest(target=target.name):
                self.start.write_text(json.dumps(self.started))
                (self.root/self.path).write_text(json.dumps(self.ledger))
                def backend(call_id):
                    target.write_text(target.read_text()+'\n')
                    return self.backend
                with self.assertRaisesRegex(ValueError, 'changed during backend audit'):
                    audit_completed_call(self.root, self.payload, self.path, lookup_call=backend)

    def test_wrong_call_stage_and_payload_fail_before_backend(self):
        for key, value in [('stage', 'pilot'), ('payload_sha256', 'b'*64), ('worker_module', 'another_worker')]:
            with self.subTest(key=key):
                (self.root/self.path).write_text(json.dumps({**self.ledger, key: value}))
                with self.assertRaises(ValueError): self.audit()


if __name__ == '__main__': unittest.main()
