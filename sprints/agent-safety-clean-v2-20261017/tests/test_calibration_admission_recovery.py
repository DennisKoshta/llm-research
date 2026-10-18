"""Control-plane recovery preserves science and cannot duplicate a paid job."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import admit_parallel_calibration as admission


class RecoveryTests(unittest.TestCase):
 def setUp(self):
  self.stack=ExitStack();self.addCleanup(self.stack.close)
  self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=Path(__file__).parent)))
  self.stack.enter_context(patch.object(admission,'ROOT',self.root))
  self.args=SimpleNamespace(attempt='fixture',author_attempt='author',recover_control_plane=True)
  self.snapshot=admission.cache_path('fixture').parent;self.snapshot.mkdir(parents=True)
  (self.root/'infrastructure').mkdir()
  self.files=['scripts/admit_parallel_calibration.py','scripts/launch_bounded_job.py','scripts/science.py','stimuli/calibration_author.json']
  hashes={}
  for name in self.files:
   content=('original fixture '+name).encode();path=self.root/name;path.parent.mkdir(exist_ok=True)
   path.write_bytes(content);(self.snapshot/Path(name).name).write_bytes(content)
   hashes[name]=hashlib.sha256(content).hexdigest()
  self.authorization={'phase2_authorized':False,'scope':'Intact fixture only'}
  self.cfg={'job':'phase1_intact_calibration_fixture','command':['unchanged fixture command'],'upper_bound_usd':12}
  cache={'schema':'immutable_calibration_authorization_v1','attempt':'fixture','author_attempt':'author',
   'authorization':self.authorization,'read_prerequisite_sha256':hashes,
   'environment':{'python':sys.version,'numpy':__import__('numpy').__version__,'sklearn':__import__('sklearn').__version__}}
  for name,value in [('authorization_cache.json',cache),('source_manifest.json',hashes),
                     ('authorization.json',self.authorization),('resource_config.json',self.cfg)]:
   (self.snapshot/name).write_text(json.dumps(value))
  self.original={path.name:path.read_bytes() for path in self.snapshot.iterdir()}
  for name in admission.CONTROL_PLANE_FILES:(self.root/name).write_text('updated control plane fixture')

 def test_only_control_sources_can_change_and_original_cache_stays_immutable(self):
  with self.assertRaisesRegex(ValueError,'Frozen prerequisite changed'):
   admission.verify_cached('fixture','author')
  self.assertEqual(admission.verify_cached('fixture','author',True)['authorization'],self.authorization)
  admission.freeze_snapshot(self.args,self.authorization,self.cfg)
  for name,value in self.original.items():self.assertEqual((self.snapshot/name).read_bytes(),value)
  recovery=list((self.snapshot/'control_plane_recovery').glob('*/recovery.json'))
  self.assertEqual(len(recovery),1)
  self.assertFalse(json.loads(recovery[0].read_text())['scientific_execution_changed'])
  admission.freeze_snapshot(self.args,self.authorization,self.cfg)
  (self.root/'scripts/science.py').write_text('changed scientific implementation')
  with self.assertRaisesRegex(ValueError,'Frozen prerequisite changed'):
   admission.verify_cached('fixture','author',True)

 def test_recovery_rejects_old_snapshot_payload_or_identity_changes(self):
  with self.assertRaisesRegex(ValueError,'identity mismatch'):
   admission.verify_cached('fixture','different_author',True)
  with self.assertRaisesRegex(ValueError,'execution payload'):
   admission.freeze_snapshot(self.args,self.authorization,{**self.cfg,'upper_bound_usd':13})
  (self.snapshot/'launch_bounded_job.py').write_text('tampered old authority')
  with self.assertRaisesRegex(ValueError,'Original control-plane snapshot'):
   admission.verify_cached('fixture','author',True)

 def test_launch_envelope_keeps_same_job_and_binds_current_control_plus_old_science(self):
  authority=SimpleNamespace(launch=Mock(return_value={'fixture':'reserved_once'}))
  with patch.object(admission,'build_config',return_value=self.cfg),patch.object(admission,'fresh_launch_authority',return_value=authority):
   self.assertEqual(admission.launch_current(self.args),{'fixture':'reserved_once'})
  launched=authority.launch.call_args.args[0]
  self.assertEqual({key:value for key,value in launched.items() if key!='frozen_files'},self.cfg)
  for name,expected in launched['frozen_files'].items():self.assertEqual(admission.digest(self.root/name),expected)
  self.assertEqual(launched['job'],'phase1_intact_calibration_fixture')
  self.assertEqual((self.snapshot/'resource_config.json').read_bytes(),self.original['resource_config.json'])

 def test_fresh_authority_does_not_reuse_loaded_limit_or_stale_bytecode(self):
  path=self.root/'scripts/launch_bounded_job.py'
  path.write_text('LIMIT=4\ndef launch(config): return LIMIT\n')
  old=admission.fresh_launch_authority()
  path.write_text('LIMIT=6\ndef launch(config): return LIMIT\n')
  self.assertEqual(old.launch({}),4)
  self.assertEqual(admission.fresh_launch_authority().launch({}),6)

 def test_reserved_job_only_recovers_observer_never_launches(self):
  registry={'jobs':[{'job':'phase1_intact_calibration_fixture','allocation_status':'reserved_pending_launch'}]}
  (self.root/'infrastructure/resource_allocations.json').write_text(json.dumps(registry))
  with patch.object(admission,'start_observer',return_value={'state':'fixture_recovery'}), \
       patch.object(admission,'audit_operation') as operation:
   self.assertEqual(admission.supervise(self.args,123),0)
   operation.assert_not_called()

 def test_child_operation_inherits_lock_and_maps_capacity_without_retrying_paid_call(self):
  response=SimpleNamespace(returncode=77,stdout=json.dumps({'capacity_error':'Resource capacity exceeded: single_gpu_workers'}),stderr='')
  with patch.object(admission.subprocess,'run',return_value=response) as run:
   with self.assertRaisesRegex(ValueError,'Resource capacity exceeded'):
    admission.audit_operation(self.args,'launch',123,admission.time.monotonic()+100)
   command=run.call_args.args[0]
   self.assertIn('--recover-control-plane',command)
   self.assertEqual(command[command.index('--operation')+1],'launch')
   self.assertEqual(run.call_args.kwargs['pass_fds'],(123,))

 def test_versioned_config_path_is_bound_and_cannot_reuse_other_cache(self):
  changed='stimuli/calibration_author_format_v2.json'
  with self.assertRaisesRegex(ValueError,'configuration path changed'):
   admission.verify_cached('fixture','author',True,changed)
  cache=json.loads(admission.cache_path('fixture').read_text());cache['config_path']=changed
  admission.cache_path('fixture').write_text(json.dumps(cache))
  self.assertEqual(admission.verify_cached('fixture','author',True,changed)['authorization'],self.authorization)
  with self.assertRaisesRegex(ValueError,'configuration path changed'):
   admission.verify_cached('fixture','author',True)
  for path in ('../outside.json','scripts/other.json','stimuli/sub/other.json','stimuli/other.txt'):
   with self.assertRaises(ValueError):admission.config_relative('author',path)

 def test_versioned_config_propagates_to_modal_command_and_child(self):
  self.args.config_path='stimuli/calibration_author_format_v2.json'
  runner=self.root/'scripts/modal_sandbox_evaluation.py';runner.write_text('fixture source')
  authority=SimpleNamespace(_decorators=lambda source:{'evaluate':{},'coordinate':{}})
  with patch.object(admission,'fresh_launch_authority',return_value=authority):
   command=admission.build_config(self.args)['command']
  self.assertEqual(command[command.index('--config')+1],self.args.config_path)
  response=SimpleNamespace(returncode=0,stdout='{}',stderr='')
  with patch.object(admission.subprocess,'run',return_value=response) as run:
   admission.audit_operation(self.args,'verify',123,admission.time.monotonic()+100)
  child=run.call_args.args[0]
  self.assertEqual(child[child.index('--config-path')+1],self.args.config_path)

 def test_new_preparation_output_preserves_old_config(self):
  import prepare_calibration
  old=self.root/'stimuli/calibration_author.json';before=old.read_bytes()
  new=self.root/'stimuli/calibration_author_format_v2.json'
  value={'trials':[1]*200,'fixture':'mechanical format clarification'}
  args=['prepare_calibration.py','--author-attempt','author','--output',str(new.relative_to(self.root))]
  with patch.object(prepare_calibration,'ROOT',self.root),patch.object(prepare_calibration,'prepare',return_value=value),patch.object(sys,'argv',args):
   prepare_calibration.main();prepare_calibration.main()
  self.assertEqual(old.read_bytes(),before)
  self.assertEqual(json.loads(new.read_text()),value)

 def test_authorize_once_reads_explicit_config_after_multiple_source_bindings(self):
  # Exercise the real authorize_once flow, not only path normalization. Source
  # files are deliberately non-JSON, so a clobbered config variable fails.
  relative='stimuli/calibration_author_format_v2.json'
  value={'author_attempt':'author','fixture':'frozen version two'}
  (self.root/relative).write_text(json.dumps(value))
  seen=[]
  def authorize(root,payload):
   self.assertEqual(root,self.root);self.assertEqual(payload,value)
   seen.append(payload);return self.authorization
  sources=['scripts/science.py','scripts/admit_parallel_calibration.py','scripts/launch_bounded_job.py']
  with patch.object(admission,'SOURCE_FILES',sources), \
       patch.object(admission.sys,'addaudithook'), \
       patch('modal_sandbox_evaluation.authorize_calibration',side_effect=authorize):
   result=admission.authorize_once('fresh_attempt','author',relative)
  cache=json.loads(admission.cache_path('fresh_attempt').read_text())
  self.assertEqual(cache['config_path'],relative)
  self.assertEqual(set(cache['read_prerequisite_sha256']),set(sources+[relative]))
  self.assertEqual(result['authorization'],self.authorization)
  self.assertEqual(len(seen),1)


if __name__=='__main__':unittest.main()
