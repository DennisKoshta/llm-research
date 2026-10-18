"""Offline closeout fixtures and fake-volume writes only; never connects to Modal."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import persist_review_closeout as closeout


class FakeVolume:
    def __init__(self):self.files={};self.fail_read=None;self.list_error=None;self.upload_calls=0
    def listdir(self,path):
        if self.list_error:raise self.list_error
        found=[name for name in self.files if name.startswith(path+'/')]
        if not found:raise FileNotFoundError(path)
        return found
    @contextmanager
    def batch_upload(self,force=False):
        assert force is False
        self.upload_calls+=1
        yield self
    def put_file(self,path,name):
        if name in self.files:raise FileExistsError(name)
        self.files[name]=Path(path).read_bytes()
    def read_file(self,name):
        value=self.files[name]
        if self.fail_read and name.endswith(self.fail_read):value+=b'corrupted-readback'
        yield value[:20];yield value[20:]


class CloseoutTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name in closeout.REQUIRED:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('{}\n' if path.suffix=='.json' else '# Synthetic closeout fixture\n')
        self.record=self.root/'records/synthetic_receipt.json';self.record.parent.mkdir()
        self.record.write_text(json.dumps({'schema':'synthetic_only','passed':False})+'\n')
        self.status={'status':'FAIL_INCOMPLETE','all_present_receipts_recomputed':True,
            'checked_at_utc':'2026-10-18T00:00:00+00:00',
            'receipt_checks':[{'path':'records/synthetic_receipt.json','status':'PASS',
                'sha256':hashlib.sha256(self.record.read_bytes()).hexdigest()}]}
        self.save_status()
        self.timestamp='20261018T000000Z'

    def save_status(self):
        (self.root/'verification/verification_status.json').write_text(json.dumps(self.status)+'\n')

    def build(self,name='one',**kwargs):
        output=self.root/'closeout_packages'/name
        result=closeout.build(self.root,self.timestamp,output,**kwargs)
        return output,result

    def test_deterministic_bundle_manifest_and_truthful_incomplete_status(self):
        one,package=self.build();two,_=self.build('two')
        self.assertEqual((one/'snapshot.tar').read_bytes(),(two/'snapshot.tar').read_bytes())
        self.assertEqual((one/'manifest.json').read_bytes(),(two/'manifest.json').read_bytes())
        self.assertFalse(package['upload_performed'])
        _,manifest=closeout.validate_package(one)
        self.assertFalse(manifest['program_completion_claim'])
        self.assertEqual(manifest['verification']['overall_status'],'FAIL_INCOMPLETE')
        with tarfile.open(one/'snapshot.tar') as bundle:
            for member in bundle.getmembers():
                self.assertTrue(member.isfile());self.assertEqual(member.mtime,0)
                row=next(row for row in manifest['files'] if row['path']==member.name)
                self.assertEqual(hashlib.sha256(bundle.extractfile(member).read()).hexdigest(),row['sha256'])

    def test_stale_or_failed_receipt_cannot_package(self):
        self.record.write_text(self.record.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Receipt bytes differ'):self.build()
        self.status['receipt_checks'][0]['status']='FAIL';self.save_status()
        with self.assertRaisesRegex(ValueError,'PASS hash'):self.build()

    def test_missing_new_duplicate_and_unverified_receipts_fail(self):
        original=json.loads(json.dumps(self.status))
        for rows in ([],self.status['receipt_checks']*2):
            self.status={**original,'receipt_checks':rows};self.save_status()
            with self.assertRaisesRegex(ValueError,'inventory'):self.build()
        self.status=original;self.save_status()
        (self.root/'records/extra.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'inventory'):self.build()

    def test_safe_allowlist_excludes_env_credentials_raw_and_model_files(self):
        for name in ('.env','infrastructure/credentials.json','raw_transcripts/foo/raw.jsonl',
                     'infrastructure/model.safetensors','infrastructure/debug.log','infrastructure/activation.npy',
                     '../outside.json'):
            with self.subTest(name=name),self.assertRaises((ValueError,FileNotFoundError)):
                closeout.select_paths(self.root,[name])
        path=self.root/'infrastructure/status.json';path.parent.mkdir(exist_ok=True);path.write_text('{}')
        paths=closeout.select_paths(self.root,['infrastructure/status.json'])
        self.assertIn('infrastructure/status.json',paths)

    def test_symlink_and_credential_literal_rejected(self):
        path=self.root/'infrastructure/provenance.json';path.parent.mkdir(exist_ok=True)
        path.symlink_to(self.record)
        with self.assertRaisesRegex(ValueError,'Symlinks'):self.build(extra_paths=['infrastructure/provenance.json'])
        path.unlink();path.write_text('hf_'+'A'*30)
        with self.assertRaisesRegex(ValueError,'credential-like'):self.build(extra_paths=['infrastructure/provenance.json'])

    def test_credential_field_in_generic_metadata_is_rejected_without_disclosure(self):
        path=self.root/'infrastructure/status.json';path.parent.mkdir(exist_ok=True)
        for key in ('MODAL_TOKEN_SECRET','api_key','access_token'):
            path.write_text(json.dumps({'nested':{key:'synthetic-private-value'}}))
            with self.assertRaisesRegex(ValueError,'credential-like') as error:
                self.build(extra_paths=['infrastructure/status.json'])
            self.assertNotIn('synthetic-private-value',str(error.exception))

    def test_oversized_selected_metadata_rejected(self):
        with patch.object(closeout,'MAX_FILE_BYTES',5):
            with self.assertRaisesRegex(ValueError,'size'):self.build()

    def test_output_must_be_fresh_and_confined(self):
        self.build()
        with self.assertRaises(FileExistsError):self.build()
        for name in ('../outside','records/closeout','closeout_packages/../escaped'):
            with self.assertRaises(ValueError):closeout.build(self.root,self.timestamp,name)
        with self.assertRaises(ValueError):closeout.build(self.root,'not-UTC','closeout_packages/new')

    def test_optional_sources_and_execution_metadata_only(self):
        for name in ('scripts/example.py','tests/test_example.py','executions/job/source_manifest.json',
                     'executions/job/worker.log','executions/job/source_files/raw.jsonl'):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# synthetic\n')
        self.assertNotIn('scripts/example.py',closeout.select_paths(self.root))
        selected=closeout.select_paths(self.root,include_sources=True)
        self.assertIn('scripts/example.py',selected)
        self.assertIn('executions/job/source_manifest.json',selected)
        self.assertNotIn('executions/job/worker.log',selected)
        self.assertNotIn('executions/job/source_files/raw.jsonl',selected)

    def test_upload_reads_back_all_bytes_and_marks_only_after_verification(self):
        output,package=self.build();volume=FakeVolume()
        receipt=closeout.upload(self.root,output,volume=volume)
        self.assertTrue(receipt['all_uploaded_bytes_readback_verified'])
        self.assertFalse(receipt['program_completion_claim'])
        self.assertEqual(receipt['overall_verification_status'],'FAIL_INCOMPLETE')
        self.assertEqual(set(volume.files),{package['remote_directory']+'/'+name for name in
            ('manifest.json','snapshot.tar','package.json','CLOSEOUT_VERIFIED.json')})
        with self.assertRaises(FileExistsError):closeout.upload(self.root,output,volume=volume)

    def test_remote_existing_or_transport_error_never_overwrites(self):
        output,package=self.build();volume=FakeVolume()
        volume.files[package['remote_directory']+'/unrelated']=b'preserved'
        with self.assertRaises(FileExistsError):closeout.upload(self.root,output,volume=volume)
        self.assertEqual(volume.upload_calls,0)
        volume=FakeVolume();volume.list_error=RuntimeError('Synthetic transport failure')
        with self.assertRaisesRegex(RuntimeError,'transport'):closeout.upload(self.root,output,volume=volume)
        self.assertEqual(volume.upload_calls,0)

    def test_remote_corruption_never_writes_verified_marker(self):
        output,_=self.build();volume=FakeVolume();volume.fail_read='snapshot.tar'
        with self.assertRaisesRegex(ValueError,'readback differs'):closeout.upload(self.root,output,volume=volume)
        self.assertFalse(any(name.endswith('CLOSEOUT_VERIFIED.json') for name in volume.files))

    def test_changed_package_or_live_evidence_blocks_upload(self):
        output,_=self.build();volume=FakeVolume()
        self.record.write_text(self.record.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Receipt bytes differ'):closeout.upload(self.root,output,volume=volume)
        self.assertEqual(volume.upload_calls,0)
        (output/'snapshot.tar').write_bytes((output/'snapshot.tar').read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'bundle changed'):closeout.validate_package(output)

    def test_changed_snapshot_during_capture_is_rejected(self):
        original=closeout.verify_receipts
        def mutate(root,captured=None):
            value=original(root,captured)
            path=self.root/'FINAL_RESULTS_FOR_REVIEW.md';path.write_text(path.read_text()+'changed')
            return value
        with patch.object(closeout,'verify_receipts',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'changed during capture'):self.build()

    def test_incomplete_snapshot_cannot_promote_completion_claim(self):
        output,_=self.build()
        manifest=json.loads((output/'manifest.json').read_bytes());manifest['program_completion_claim']=True
        data=closeout.json_bytes(manifest);(output/'manifest.json').write_bytes(data)
        package=json.loads((output/'package.json').read_bytes());package['manifest_sha256']=closeout.sha(data)
        (output/'package.json').write_bytes(closeout.json_bytes(package))
        with self.assertRaisesRegex(ValueError,'completion claim'):closeout.validate_package(output)


if __name__=='__main__':unittest.main()
