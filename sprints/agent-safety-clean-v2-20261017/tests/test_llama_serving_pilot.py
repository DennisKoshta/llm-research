"""CPU-only contract tests; no Llama/vLLM GPU loader is exercised."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib,json,struct,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import llama_serving_worker as worker
from strict_parser import parse_output


class LlamaPilotContracts(unittest.TestCase):
 def test_reasoning_instruction_is_system_content_not_assistant_prefix(self):
  trial={'messages':[{'role':'system','content':'Synthetic sandbox.'},{'role':'user','content':'Complete the batch.'}]}
  messages=worker.reasoning_messages(trial)
  self.assertEqual([m['role'] for m in messages],['system','user'])
  self.assertEqual(trial['messages'][0]['content'],'Synthetic sandbox.')
  self.assertIn('emit the closing </think> tag yourself',messages[0]['content'])
  with self.assertRaises(AssertionError):worker.reasoning_messages({'messages':messages+[{'role':'assistant','content':'<think>'}]})

 def test_unclosed_llama_reasoning_never_becomes_a_decision(self):
  raw='<think>Considering <tool_call>{"name":"route","arguments":{}}</tool_call>'
  row=parse_output(raw,'cot_on',['route'])
  self.assertEqual(row['parse_status'],'TRUNCATED_IN_THINK')
  self.assertFalse(worker.parser_agreement(row))
  closed=parse_output('<think>Review.</think><tool_call>{"name":"route","arguments":{}}</tool_call>','cot_on',['route'])
  self.assertTrue(worker.parser_agreement(closed))

 def test_serialized_inventory_is_architecture_derived(self):
  with TemporaryDirectory(dir=Path(__file__).parent) as directory:
   root=Path(directory);header={};offset=0
   def add(name,dtype,size):
    nonlocal offset
    header[name]={'dtype':dtype,'shape':[size],'data_offsets':[offset,offset+size]};offset+=size
   for layer in range(2):
    for leaf in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj'):
     module=f'model.layers.{layer}.{leaf}'
     add(module+'.weight','F8_E4M3',4);add(module+'.weight_scale','F32',4);add(module+'.input_scale','F32',4)
   add('lm_head.weight','BF16',8);add('model.embed_tokens.weight','BF16',8)
   shard='model-00001-of-00001.safetensors';raw=json.dumps(header).encode()
   (root/shard).write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*offset)
   files={'config.json':{'architectures':['LlamaForCausalLM'],'model_type':'llama','num_hidden_layers':2},
    'hf_quant_config.json':{'producer':{'name':'modelopt'},'quantization':{'quant_algo':'FP8','kv_cache_quant_algo':'FP8','exclude_modules':['lm_head']}},
    'model.safetensors.index.json':{'metadata':{'total_size':offset},'weight_map':{name:shard for name in header}}}
   hashes={}
   for name,value in files.items():
    data=json.dumps(value).encode();(root/name).write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
   with patch.object(worker,'PINNED_METADATA_SHA256',hashes):
    result=worker.header_inventory(root)
    self.assertEqual(result['serialized_fp8_modules'],14)
    self.assertEqual(result['num_hidden_layers'],2)
    # Header metadata claiming a different FP8 dtype must fail coverage checks.
    header['model.layers.0.self_attn.q_proj.weight']['dtype']='F8_E5M2'
    raw=json.dumps(header).encode();(root/shard).write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*offset)
    with self.assertRaises(AssertionError):worker.header_inventory(root)


if __name__=='__main__':unittest.main()
