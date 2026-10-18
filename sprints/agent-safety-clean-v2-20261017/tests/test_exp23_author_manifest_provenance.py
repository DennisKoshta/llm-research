"""Exact author manifest bytes remain distinct from canonical JSON equality."""
from pathlib import Path
import json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_exp23_launch import audit_author_download_manifest
from exp23_author_worker import MODEL_ID,MODEL_REVISION,MODEL_PATH,hash_bytes


class AuthorManifestTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  (self.root/'infrastructure').mkdir()
  self.local=self.root/'infrastructure'/(MODEL_ID.replace('/','__')+'_download.json')
  self.mirror=self.root/'infrastructure/exp23_author_fixture_volume_download_manifest.json'
  self.value={'repo_id':MODEL_ID,'revision':MODEL_REVISION,'volume_path':MODEL_PATH,'status':'downloaded'}
  self.local.write_text(json.dumps(self.value,sort_keys=True,indent=2)+'\n')
  self.mirror.write_text(json.dumps(self.value,separators=(',',':')))
  self.payload={'attempt':'fixture','download_manifest_sha256':hash_bytes(self.local),
                'download_manifest_volume_sha256':hash_bytes(self.mirror)}
 def test_different_serializations_require_both_exact_hashes(self):
  hashes=audit_author_download_manifest(self.root,self.payload)
  self.assertEqual(len(hashes),2);self.assertNotEqual(*hashes.values())
 def test_old_missing_volume_key_requires_identical_bytes(self):
  del self.payload['download_manifest_volume_sha256']
  with self.assertRaisesRegex(ValueError,'Exact mirrored'):audit_author_download_manifest(self.root,self.payload)
  self.mirror.write_bytes(self.local.read_bytes())
  self.assertEqual(len(audit_author_download_manifest(self.root,self.payload)),2)
 def test_wrong_mirror_hash_or_changed_canonical_provenance_fails(self):
  self.payload['download_manifest_volume_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'Exact mirrored'):audit_author_download_manifest(self.root,self.payload)
  self.mirror.write_text(json.dumps({**self.value,'revision':'f'*40}))
  self.payload['download_manifest_volume_sha256']=hash_bytes(self.mirror)
  with self.assertRaisesRegex(ValueError,'different provenance'):audit_author_download_manifest(self.root,self.payload)
 def test_missing_mirror_and_changed_local_bytes_fail(self):
  self.local.write_text(self.local.read_text()+' ')
  with self.assertRaisesRegex(ValueError,'local author'):audit_author_download_manifest(self.root,self.payload)
  self.local.write_text(json.dumps(self.value,sort_keys=True,indent=2)+'\n');self.mirror.unlink()
  with self.assertRaises(FileNotFoundError):audit_author_download_manifest(self.root,self.payload)


if __name__=='__main__':unittest.main()
