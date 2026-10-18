"""Budget retries preserve every prompt and retain old raw-audit compatibility."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from exp23_author_worker import build_payload, validate_payload, RUNTIME_FILES


def payload(cap):
    return build_payload('synthetic', source_hashes={name: '0'*64 for name in RUNTIME_FILES},
        specification_sha256='1'*64, download_manifest_sha256='2'*64, preflight_sha256='3'*64,
        external_mechanical_receipt_sha256='4'*64, cap=cap)


class AuthorBudgetTests(unittest.TestCase):
    def test_all_fifty_three_requests_preserve_content_and_only_raise_cap(self):
        old, new = payload(4096), payload(16384)
        for left, right in zip(old['pilots']+old['requests'], new['pilots']+new['requests']):
            self.assertEqual({k:v for k,v in left.items() if k!='max_new_tokens'},
                             {k:v for k,v in right.items() if k!='max_new_tokens'})
            self.assertEqual(right['max_new_tokens'], new['cap'])
        self.assertEqual(new['max_model_len'], 32768)
        validate_payload(old); validate_payload(new)

    def test_undeclared_budget_and_metadata_mismatch_rejected(self):
        with self.assertRaises(ValueError): payload(8192)
        value = payload(16384); value['requests'][0]['max_new_tokens'] = 4096
        with self.assertRaises(ValueError): validate_payload(value)
