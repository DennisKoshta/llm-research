"""Receipt byte identity across recomputation; synthetic fixtures only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from verify_all_receipts import audit


class ReceiptBytesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'records').mkdir()
        self.path=self.root/'records/synthetic_post_fp8.json'
        self.saved={'schema':'phase1_post_fp8_v1','attempt':'SYNTHETIC_ONLY','passed':False,
                    'fixture_description':'Synthetic byte-identity test only.'}
        self.original=(json.dumps(self.saved,indent=2)+'\n').encode()
        self.path.write_bytes(self.original)

    def run_audit(self,mutation=None):
        def recompute(root,attempt):
            self.assertEqual(root,self.root)
            self.assertEqual(attempt,'SYNTHETIC_ONLY')
            if mutation:mutation()
            return deepcopy(self.saved)
        with patch('check_phase1_activation_gate.recompute',side_effect=recompute) as checker:
            result=audit(self.root)
            checker.assert_called_once()
        self.assertEqual(result['status'],'FAIL_INCOMPLETE')
        self.assertFalse(result['all_summary_numbers_recomputed'])
        return result

    def test_pass_row_binds_exact_audited_bytes_without_claiming_completion(self):
        result=self.run_audit()
        self.assertEqual(result['receipt_checks'],[{'path':'records/synthetic_post_fp8.json',
            'status':'PASS','schema':'phase1_post_fp8_v1',
            'sha256':hashlib.sha256(self.original).hexdigest()}])
        self.assertTrue(result['all_present_receipts_recomputed'])
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_semantically_identical_reformat_during_recompute_is_rejected(self):
        replacement=json.dumps(self.saved,separators=(',',':')).encode()
        self.assertEqual(json.loads(replacement),json.loads(self.original))
        self.assertNotEqual(replacement,self.original)
        result=self.run_audit(lambda:self.path.write_bytes(replacement))
        row=result['receipt_checks'][0]
        self.assertEqual(row['status'],'FAIL')
        self.assertIn('Receipt bytes changed during raw-data recomputation',row['error'])
        self.assertNotIn('sha256',row)
        self.assertFalse(result['all_present_receipts_recomputed'])

    def test_changed_number_during_recompute_is_rejected_even_when_old_result_matches(self):
        replacement={**self.saved,'passed':True}
        result=self.run_audit(lambda:self.path.write_text(json.dumps(replacement)))
        self.assertEqual(result['receipt_checks'][0]['status'],'FAIL')
        self.assertIn('Receipt bytes changed',result['receipt_checks'][0]['error'])
        self.assertFalse(result['all_present_receipts_recomputed'])

    def test_deleted_receipt_during_recompute_fails_without_pass_hash(self):
        result=self.run_audit(self.path.unlink)
        self.assertEqual(result['receipt_checks'][0]['status'],'FAIL')
        self.assertNotIn('sha256',result['receipt_checks'][0])
        self.assertFalse(result['all_present_receipts_recomputed'])


if __name__=='__main__':unittest.main()
