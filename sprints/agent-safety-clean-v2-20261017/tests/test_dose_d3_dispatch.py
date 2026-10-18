"""Synthetic dispatch regressions; no empirical dose data or cloud execution."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import audit_dose_response as old
import audit_dose_response_d3 as new
import phase1_readiness as readiness
from verify_all_receipts import audit as verify

D2 = 'author_v1_d2_descriptive'
D3 = 'author_v1_d3_cap8192_descriptive'


class DoseD3DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tests')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'records').mkdir()

    def receipt(self, attempt):
        return {'schema':'phase1_dose_response_raw_audit_v1', 'attempt':attempt,
                'author_attempt':'v1', 'primary_qualification_passed':False,
                'phase2_authorized':False, 'synthetic_only':True}

    def save(self, value):
        path = self.root/'records/synthetic_dose_receipt.json'
        path.write_text(json.dumps(value)+'\n')
        return path

    def test_readiness_d3_uses_new_auditor(self):
        saved = self.receipt(D3)
        with patch.object(new,'recompute',return_value=deepcopy(saved)) as fresh, \
             patch.object(old,'recompute',side_effect=AssertionError('Old auditor selected')) as frozen:
            self.assertEqual(readiness.recompute_component(self.root,'dose',saved),saved)
            fresh.assert_called_once_with(self.root,D3,'v1'); frozen.assert_not_called()

    def test_readiness_d2_uses_unchanged_auditor(self):
        saved = self.receipt(D2)
        with patch.object(old,'recompute',return_value=deepcopy(saved)) as frozen, \
             patch.object(new,'recompute',side_effect=AssertionError('Retry auditor selected')) as fresh:
            self.assertEqual(readiness.recompute_component(self.root,'dose',saved),saved)
            frozen.assert_called_once_with(self.root,D2,'v1'); fresh.assert_not_called()

    def test_verifier_routes_both_versions_without_claiming_program_completion(self):
        for attempt, selected, rejected in ((D2,old,new),(D3,new,old)):
            with self.subTest(attempt=attempt):
                saved = self.receipt(attempt); self.save(saved)
                with patch.object(selected,'recompute',return_value=deepcopy(saved)) as called, \
                     patch.object(rejected,'recompute',side_effect=AssertionError('Wrong auditor')) as other:
                    result=verify(self.root)
                    called.assert_called_once_with(self.root,attempt,'v1'); other.assert_not_called()
                self.assertEqual(result['receipt_checks'][0]['status'],'PASS')
                self.assertEqual(result['status'],'FAIL_INCOMPLETE')
                self.assertFalse(result['all_summary_numbers_recomputed'])

    def test_unknown_near_match_attempts_cannot_use_d3_pass(self):
        for attempt in (D3+'_extra', D3.upper(), D3+' ', 'unknown_dose_attempt'):
            with self.subTest(attempt=attempt):
                saved=self.receipt(attempt); self.save(saved)
                with patch.object(new,'recompute',return_value=deepcopy(saved)) as fresh:
                    with self.assertRaises((AssertionError,OSError)):
                        readiness.recompute_component(self.root,'dose',saved)
                    result=verify(self.root)
                    fresh.assert_not_called()
                self.assertEqual(result['receipt_checks'][0]['status'],'FAIL')
                self.assertFalse(result['all_present_receipts_recomputed'])

    def test_exact_d3_name_cannot_relabel_old_frozen_design(self):
        saved=self.receipt(D3); self.save(saved)
        frozen=self.root/f'group_calls/phase1_dose_response_{D3}/frozen_input.json'
        frozen.parent.mkdir(parents=True)
        frozen.write_text(json.dumps({'config':{'schema':'phase1_descriptive_dose_evaluation_design_v1',
            'cap':4096,'descriptive_attempt':D2},'authorization':{}}))
        with patch.object(old,'recompute',return_value=deepcopy(saved)) as frozen_auditor:
            with self.assertRaises(AssertionError):readiness.recompute_component(self.root,'dose',saved)
            self.assertEqual(verify(self.root)['receipt_checks'][0]['status'],'FAIL')
            frozen_auditor.assert_not_called()

    def test_d3_cannot_change_author_attempt(self):
        saved={**self.receipt(D3),'author_attempt':'other_author'}; self.save(saved)
        frozen=self.root/f'group_calls/phase1_dose_response_{D3}/frozen_input.json'
        frozen.parent.mkdir(parents=True)
        frozen.write_text(json.dumps({'config':{},'authorization':{}}))
        with self.assertRaises(AssertionError):readiness.recompute_component(self.root,'dose',saved)
        self.assertEqual(verify(self.root)['receipt_checks'][0]['status'],'FAIL')

    def test_verifier_rejects_corrupted_saved_d3_result_after_correct_dispatch(self):
        exact=self.receipt(D3)
        self.save({**exact,'primary_qualification_passed':True,'phase2_authorized':True})
        with patch.object(new,'recompute',return_value=exact) as fresh, \
             patch.object(old,'recompute',side_effect=AssertionError('Old auditor selected')) as frozen:
            result=verify(self.root)
            fresh.assert_called_once_with(self.root,D3,'v1'); frozen.assert_not_called()
        self.assertEqual(result['receipt_checks'][0]['status'],'FAIL')
        self.assertIn('does not exactly match',result['receipt_checks'][0]['error'])

    def test_missing_attempt_is_rejected_without_auditor_fallback(self):
        saved=self.receipt(D3); del saved['attempt']; self.save(saved)
        with patch.object(new,'recompute') as fresh, patch.object(old,'recompute') as frozen:
            with self.assertRaises(KeyError):readiness.recompute_component(self.root,'dose',saved)
            self.assertEqual(verify(self.root)['receipt_checks'][0]['status'],'FAIL')
            fresh.assert_not_called(); frozen.assert_not_called()


if __name__=='__main__':unittest.main()
