"""Bounded same-checkpoint refusal-direction projection; all forwards, reversible hooks."""
from pathlib import Path
import os,time,json
import modal
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2] if modal.is_local() else Path('/root')
CAMPAIGN='agent-safety-ablation-20261016'
DEADLINE=1792198579.2516067
app=modal.App(CAMPAIGN+'-h200-v1')
weights=modal.Volume.from_name('curiosity-study-weights')
artifacts=modal.Volume.from_name('curiosity-study-artifacts')
image=(modal.Image.debian_slim(python_version='3.12')
 .uv_sync(str(PROJECT),extras=['inference','cloud'],uv_version='0.12.18')
 .env({'HF_HOME':'/weights/huggingface','HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2'})
 .add_local_dir(str(HERE/'vendor'),'/root/safety_vendor',copy=True)) if modal.is_local() else modal.Image.debian_slim(python_version='3.12')

@app.cls(image=image,gpu='H200',cpu=(2,2),memory=(32768,49152),volumes={'/weights':weights,'/artifacts':artifacts},max_containers=2,min_containers=0,buffer_containers=0,scaledown_window=180,startup_timeout=600,timeout=1200,retries=0,enable_memory_snapshot=False)
class Inference:
 @modal.enter()
 def load(self):
  import sys,threading
  from contextlib import contextmanager
  sys.path.insert(0,'/root/safety_vendor')
  from curiosity_study.batched import BatchedHFModel
  if time.time()>=DEADLINE:raise RuntimeError('Expired before startup')
  def guard():
   time.sleep(max(0,DEADLINE-time.time()));os._exit(72)
  threading.Thread(target=guard,daemon=True).start()
  class CaptureModel(BatchedHFModel):
   @contextmanager
   def batch_hook(self,interventions,captures):
    if any(x is not None for x in interventions):raise ValueError('Use named arm, not legacy additive intervention')
    handles=[]
    try:
     for layer in self.capture_layers:
      if self.capture_site=='block_input':
       def pre(module,args,layer=layer):
        tensor=args[0]
        for pos in self.capture_positions:
         values=tensor[:,pos].float().detach().cpu().tolist()
         for capture,vector in zip(captures,values):capture.setdefault('pre_activations',{}).setdefault(str(layer),{})[str(pos)]=vector
       handles.append(self.layers[layer].register_forward_pre_hook(pre))
      else:
       def post(module,args,out,layer=layer):
        tensor=out[0] if isinstance(out,tuple) else out
        for capture,vector in zip(captures,tensor[:,-1].float().detach().cpu().tolist()):capture.setdefault('activations',{})[str(layer)]=vector
       handles.append(self.layers[layer].register_forward_hook(post))
     self.active_hooks+=len(handles);yield
    finally:
     for h in handles:h.remove()
     self.active_hooks-=len(handles)
  self.model=CaptureModel('Qwen/Qwen3-32B','9216db5781bf21249d130ec9da846c4624c16137')
  self.metadata=self.model.metadata();self.metadata['service_version']='refusal_projection_v1'
  self.metadata['projection']='rank-one FP32 subtraction cast to activation dtype at every decoder input, attention output and MLP output; all positions and generation forwards; no weight mutation'
  self.started=time.time()

 @modal.method()
 def infer(self,job):
  import gzip,hashlib,uuid
  torch=self.model.torch;requests=job['requests']
  if time.time()>=DEADLINE-90:raise RuntimeError('Insufficient lifetime')
  if not 1<=len(requests)<=32:raise ValueError('Batch cap')
  arm=requests[0].get('arm','baseline')
  if arm not in ('baseline','zero_sham','refusal_ablation','random_ablation'):raise ValueError('Unknown arm')
  if any(r.get('arm','baseline')!=arm for r in requests):raise ValueError('Mixed arm')
  for r in requests:
   if r.get('intervention') is not None or not 1<=r.get('max_new_tokens',16)<=4096:raise ValueError('Invalid request')
  self.model.capture_layers=requests[0].get('capture_layers',[])
  self.model.capture_site=requests[0].get('capture_site','block_output')
  self.model.capture_positions=requests[0].get('capture_positions',[-1])
  signature=lambda r:(r.get('capture_layers',[]),r.get('capture_site','block_output'),r.get('capture_positions',[-1]),r.get('max_new_tokens',16),r.get('enable_thinking',False),r.get('direction_key','locked'))
  if any(signature(r)!=signature(requests[0]) for r in requests):raise ValueError('Mixed settings')
  lens=[len(self.model.encode(r['messages'],r.get('enable_thinking',False))[1].input_ids[0]) for r in requests]
  if max(lens)>4096:raise ValueError('Prompt resource cap')
  batch_size=32 if max(lens)<=2048 and requests[0].get('max_new_tokens',16)<=512 else 4
  if len(requests)>batch_size:raise ValueError('Submit correct immutable cohort size')
  spec=job.get('intervention_spec');direction=None
  if arm!='baseline':
   if not spec:raise ValueError('Missing frozen intervention')
   actual_spec_sha=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   if actual_spec_sha!=job.get('intervention_spec_sha256'):raise ValueError('Worker intervention spec hash mismatch')
   direction=torch.tensor(spec['direction'],device=self.model.device,dtype=torch.float32)
   if direction.shape!=(self.model.model.config.hidden_size,) or not torch.isfinite(direction).all():raise ValueError('Invalid direction')
   if abs(float(direction.norm())-1)>1e-5:raise ValueError('Direction must be unit norm')
  inventory=lambda:sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in self.model.model.modules())
  before_hooks=inventory();handles=[];counts={};samples=[];started=time.time()
  def project(tensor,site):
   counts[site]=counts.get(site,0)+1
   if arm=='zero_sham':return tensor
   x=tensor.float();changed=(x-(x@direction).unsqueeze(-1)*direction).to(tensor.dtype)
   if job.get('engineering') and site=='input_28' and counts[site]<=2:
    samples.append({'site':site,'call':counts[site],'shape':list(tensor.shape),'before':tensor[0,-1].float().cpu().tolist(),'after':changed[0,-1].float().cpu().tolist(),'dtype':str(tensor.dtype)})
   return changed
  try:
   if arm!='baseline':
    for index,layer in enumerate(self.model.layers):
     def pre(module,args,index=index):return (project(args[0],f'input_{index}'),*args[1:])
     handles.append(layer.register_forward_pre_hook(pre))
     for kind,module in [('attention',layer.self_attn),('mlp',layer.mlp)]:
      def post(module,args,out,index=index,kind=kind):
       value=out[0] if isinstance(out,tuple) else out;changed=project(value,f'{kind}_{index}')
       return (changed,*out[1:]) if isinstance(out,tuple) else changed
      handles.append(module.register_forward_hook(post))
   rows=self.model.infer_batch(requests,[r['id'] for r in requests])
  finally:
   for h in handles:h.remove()
  if inventory()!=before_hooks or self.model.active_hooks:raise RuntimeError('Hook cleanup failed')
  for r,row in zip(requests,rows):
   row['id']=r['id'];row['arm']=arm;row['intervention_lock_sha256']=job.get('intervention_lock_sha256');row['activations']=row['capture'].pop('activations',{});row['pre_activations']=row['capture'].pop('pre_activations',{})
   row['intervention']={'arm':arm,'spec_sha256':job.get('intervention_spec_sha256'),'direction_key':r.get('direction_key','locked'),'coverage':'all prompt and generation positions; decoder input,attention output,MLP output','coefficient':0 if arm in ('baseline','zero_sham') else 1,'weight_mutation':False}
   row['hooks_remaining']=inventory()-before_hooks
  result={'rows':rows,'metadata':self.metadata,'job':{k:v for k,v in job.items() if k not in ('requests','intervention_spec')},'projection_audit':{'sites':len(counts),'site_forward_counts':counts,'samples':samples,'hooks_before':before_hooks,'hooks_after':inventory(),'zero_sham_identity_implementation':arm=='zero_sham'},'service_started':self.started,'started':started,'finished':time.time(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'container_id':os.environ.get('MODAL_TASK_ID','unknown')}
  root=Path('/artifacts')/CAMPAIGN/'jobs';root.mkdir(parents=True,exist_ok=True)
  target=root/(job['key']+'--'+uuid.uuid4().hex+'.json.gz')
  with gzip.open(target,'wt',compresslevel=1) as f:json.dump(result,f)
  result['remote_artifact']=str(target);artifacts.commit();return result
