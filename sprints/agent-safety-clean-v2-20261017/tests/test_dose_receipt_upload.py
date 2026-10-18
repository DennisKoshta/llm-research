"""Missing remote objects use the SDK's actual FileNotFoundError surface."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import admit_phase1_doses as supervisor


class VolumeFixture:
    def __init__(self): self.files = {}; self.uploads = 0
    def read_file(self, name):
        if name not in self.files: raise FileNotFoundError(name)
        return iter([self.files[name]])
    def batch_upload(self, force):
        assert force is False
        return self
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def put_file(self, path, name):
        assert name not in self.files
        self.files[name] = Path(path).read_bytes(); self.uploads += 1


class DoseReceiptUploadTests(unittest.TestCase):
    def test_missing_then_idempotent_verified_upload_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory); (root/'records').mkdir()
            path = root/'records/phase1_dose_response_synthetic_receipt.json'
            path.write_text('{"synthetic_test_fixture":true}\n')
            volume = VolumeFixture()
            with patch.object(supervisor, 'ROOT', root), patch('modal.Volume.from_name', return_value=volume):
                first = supervisor.upload_receipt(SimpleNamespace(attempt='synthetic'))
                second = supervisor.upload_receipt(SimpleNamespace(attempt='synthetic'))
                self.assertEqual(first, second); self.assertEqual(volume.uploads, 1)
                self.assertTrue(first['receipt_uploaded'])
                volume.files[next(iter(volume.files))] = b'changed'
                with self.assertRaises(AssertionError):
                    supervisor.upload_receipt(SimpleNamespace(attempt='synthetic'))
