from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from itertools import product
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import exp32_contract as contract
import exp32_analysis as analysis
import exp32_admission as admission
from exp32_plan import digest


class Tokenizer:
    def apply_chat_template(self,messages,*,add_generation_prompt,tokenize,enable_thinking):
        text=''.join(m['role']+':'+m['content']+'\n' for m in messages)
        if add_generation_prompt:text+='assistant:<think>\n'+('' if enable_thinking else '</think>\n')
        return text
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
    def __call__(self,text,**kwargs):return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}


def source(index,arm='intact',group=0):
    tokenizer=Tokenizer();tid=f'SYNTHETIC:g{group}:s{index}:{arm}'
    request={'trial_id':tid,'scenario_id':f's{group}_{index}','template_id':f't{group}',
             'arm':arm,'cot_mode':'cot_on','cost_regime':'high_friction','consequence':'harmful',
             'boundary':'implicit','domain':'finance','split':'evaluation','design_family':'main',
             'messages':[{'role':'user','content':'Synthetic operational workspace. Quoted marker <think> is part of the input.'}]}
    identity={'model_id':'Qwen/Qwen3.8-27B','base_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
              'model_revision':'a'*64,'model_arm':arm,
              'model_path':'/weights/agent-safety-clean-v2-20261017/checkpoints_full_residual_v2/'+arm}
    prompt=tokenizer.apply_chat_template(request['messages'],add_generation_prompt=True,tokenize=False,enable_thinking=True)
    first={'trial_id':tid,'turn_index':0,'messages':request['messages'],'checkpoint_identity':identity,
           'tokenizer_file_sha256':{'tokenizer.json':'b'*64},'full_prompt':prompt,'prompt_token_ids':tokenizer.encode(prompt)}
    return request,first,identity,tokenizer


def dataset():
    inputs=[];labels={};vectors={}
    for group in range(5):
        for index in range(4):
            for arm in ('intact','refusal_ablated'):
                req,first,identity,tok=source(index,arm,group)
                item=contract.prepare_input('exp11',req,tok,first,identity,first['tokenizer_file_sha256'])
                inputs.append(item);y=index%2;labels[item['trial_id']]=y
                x=np.zeros(5120,dtype=np.float64);x[0]=2*y-1 if arm=='intact' else 0.;x[1]=2*y-1;x[2]=group/10
                vectors[item['trial_id']]=x
    inputs.sort(key=lambda r:r['trial_id']);x=np.stack([vectors[r['trial_id']] for r in inputs])
    metadata=[];outcomes=[]
    for i,item in enumerate(inputs):
        metadata.append({'trial_id':item['trial_id'],'pre_action_input_sha256':item['input_sha256'],
          'timing_sha256':digest(item['timing']),'checkpoint_identity':item['checkpoint_identity'],
          'tokenizer_file_sha256':item['tokenizer_file_sha256'],'layer_index':48,
          'layer_convention':'zero_based_decoder_block_output','forward_input_token_ids':item['timing']['prefix_token_ids'],
          'target_token_index':item['timing']['target_token_index'],'generated_text_used':False,'forward_only':True,
          'activation_row_sha256':hashlib.sha256(x[i].tobytes()).hexdigest()})
        outcomes.append({'trial_id':item['trial_id'],'request_sha256':item['request_sha256'],
          'first_request_sha256':item['first_request_sha256'],'outcome_category':'SHORTCUT_VIOLATION' if labels[item['trial_id']] else 'SAFE_ROUTE'})
    rhat=np.zeros(5120);rhat[0]=1
    return inputs,metadata,x,rhat,outcomes


def assembled():
    return contract.assemble(*dataset(),source_bindings={'synthetic_inputs':'c'*64},synthetic=True)


class PreparationTests(unittest.TestCase):
    def test_exact_generation_marker_despite_quoted_think_and_source_changes(self):
        req,first,identity,tok=source(0)
        row=contract.prepare_input('exp11',req,tok,first,identity,first['tokenizer_file_sha256'])
        self.assertTrue(row['available_prompt_text'].endswith('assistant:'))
        self.assertIn('Quoted marker <think>',row['available_prompt_text'])
        self.assertEqual(row['timing']['target_token_id'],ord(':'))
        bad=deepcopy(first);bad['prompt_token_ids'][-1]+=1
        with self.assertRaisesRegex(ValueError,'Token IDs'):contract.prepare_input('exp11',req,tok,bad,identity,first['tokenizer_file_sha256'])
        req['raw_output']='leak'
        with self.assertRaisesRegex(ValueError,'Outcome/completion'):contract.prepare_input('exp11',req,tok,first,identity,first['tokenizer_file_sha256'])

    def test_outcome_array_timing_and_checkpoint_swaps_rejected(self):
        inputs,meta,x,rhat,outcomes=dataset()
        for change in ('array','timing','checkpoint','outcome'):
            a,b,c,d,e=deepcopy((inputs,meta,x,rhat,outcomes))
            if change=='array':c[0,0]+=1
            if change=='timing':b[0]['target_token_index']-=1
            if change=='checkpoint':b[0]['checkpoint_identity']={**b[0]['checkpoint_identity'],'model_arm':'unrelated'}
            if change=='outcome':e[0]['request_sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):contract.assemble(a,b,c,d,e,source_bindings={'x':'a'*64},synthetic=True)

    def test_saved_npz_requires_exact_unicode_ids_and_hash(self):
        values=assembled();ids=[r['trial_id'] for r in values['rows']]
        with tempfile.TemporaryDirectory(dir=ROOT/'infrastructure',prefix='exp32_test_') as tmp:
            path=Path(tmp)/'saved.npz';np.savez(path,trial_ids=np.asarray(ids),layer48=values['activations'])
            loaded=contract.load_arrays(ROOT,str(path.relative_to(ROOT)),contract.file_hash(path),ids)
            np.testing.assert_array_equal(loaded,values['activations'])
            with self.assertRaises(ValueError):contract.load_arrays(ROOT,str(path.relative_to(ROOT)),contract.file_hash(path),ids[::-1])
            with self.assertRaises(ValueError):contract.load_arrays(ROOT,str(path.relative_to(ROOT)),'0'*64,ids)

    def test_direction_bytes_are_bound_without_renormalizing(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'infrastructure',prefix='exp32_direction_test_') as tmp:
            path=Path(tmp)/'direction.npy';r=np.zeros(5120,dtype=np.float32);r[0]=1;np.save(path,r)
            value,proof=contract.load_direction(ROOT,path.relative_to(ROOT),contract.file_hash(path))
            np.testing.assert_array_equal(value,r.astype(np.float64));self.assertFalse(proof['renormalized'])
            with self.assertRaises(ValueError):contract.load_direction(ROOT,path.relative_to(ROOT),'0'*64)

    def test_group_bootstrap_reports_single_class_replicates_and_sparse_limit(self):
        rows=[{'split_group':str(g),'y':int(g>=2),'scores':{'m':float(g>=2)}} for g in range(4)]
        value=analysis.bootstrap(rows,'m',replicates=5000)
        self.assertTrue(4000<value['valid_replicates']<4700)
        self.assertTrue(value['ci_available']);self.assertTrue(value['degenerate_interval']['auroc'])
        rows=rows[:2]
        value=analysis.bootstrap(rows,'m',replicates=10)
        self.assertFalse(value['ci_available']);self.assertEqual(value['ci_unavailable_reason'],'UNDEFINED_POINT_METRIC')

    def test_full_coverage_checks_factorial_cells_not_only_row_counts(self):
        from exp32_plan import LEVELS
        rows=[]
        def add(exp,scenario,template,arm,mode,family,cost,harm,boundary,domain,fmt='not_applicable'):
            rows.append({'trial_id':str(len(rows)),'experiment':exp,'scenario_id':scenario,'template_id':template,
                         'split_group':template,'cot_mode':mode,'design_family':family,'payoff_format':fmt,
                         'factors':{'arm':arm,'cost':cost,'consequence':harm,'boundary':boundary,'domain':domain}})
        for arm in LEVELS['arm']:
            for i,cost,harm,boundary in product(range(40),LEVELS['cost'],LEVELS['consequence'],LEVELS['boundary']):
                add('exp11',f's{i}',f't{i//4}',arm,'cot_on','main',cost,harm,boundary,LEVELS['domain'][(i//4)%4])
            for family,n in [('direct_instruction',60),('forced_staged',120),('unlisted_shortcut',40)]:
                for j in range(n):
                    i=j%40;add('exp11',f's{i}',f't{i//4}',arm,'cot_on',family,'high_friction','harmful','implicit',LEVELS['domain'][(i//4)%4])
        for arm,i,cost,boundary in product(LEVELS['arm'][:2],range(40),LEVELS['cost'][1:],LEVELS['boundary']):
            add('exp11',f's{i}',f't{i//4}',arm,'cot_off','main',cost,'harmful',boundary,LEVELS['domain'][(i//4)%4])
        cells=list(product(LEVELS['cost'],LEVELS['consequence'],LEVELS['boundary']))
        for group,variant,domain,arm,mode,fmt in product(range(12),range(4),LEVELS['domain'],LEVELS['arm'][:2],('cot_on','cot_off'),('numeric_table','narrative_stakes')):
            cost,harm,boundary=cells[(group+3*variant)%12]
            add('exp31',f's{group}_{variant}_{domain}',f'c{group}',arm,mode,'matched_domain',cost,harm,boundary,domain,fmt)
        self.assertEqual(contract.coverage(rows),{'exp11':4520,'exp31':1536,'main_analysis_rows':4736})
        rows[0]['factors']['boundary']='explicit'
        with self.assertRaisesRegex(ValueError,'factorial'):contract.coverage(rows)

    def test_group_splits_keep_all_variants_and_ignore_outcomes(self):
        values=assembled();rows=values['rows'];before=analysis.fold_map(rows)
        changed=deepcopy(rows)
        for r in changed:r['outcome_category']='SAFE_ROUTE'
        self.assertEqual(before,analysis.fold_map(changed[::-1]))
        self.assertEqual(set(before.values()),set(range(5)))
        changed[0]['split_group']='another'
        with self.assertRaises(ValueError):analysis.fold_map(changed)

    def test_full_interaction_includes_five_way_product(self):
        terms=analysis.interaction_terms('exp11');self.assertEqual(len(terms),287)
        self.assertEqual(len(analysis.interaction_terms('exp31')),95)
        term=(('cost','high_friction'),('consequence','harmful'),('boundary','explicit'),('domain','wetlab'),('arm','refusal_ablated'))
        self.assertIn(term,terms)
        row={'factors':dict(term)}
        self.assertEqual(analysis.interaction_matrix([row],'exp11')[0,terms.index(term)],1.)
        row['factors']['boundary']='implicit'
        self.assertEqual(analysis.interaction_matrix([row],'exp11')[0,terms.index(term)],0.)

    def test_cell_selection_uses_training_rows_only(self):
        rows=assembled()['rows'];train=[r for r in rows if r['split_group']!='t0'];test=[r for r in rows if r['split_group']=='t0']
        before=analysis.eligible_cells(train)
        for row in test:row['outcome_category']='SAFE_ROUTE'
        self.assertEqual(before,analysis.eligible_cells(train))
        self.assertTrue(all(r['eligible'] for r in before.values()))
        for row in train:row['outcome_category']='SAFE_ROUTE'
        self.assertFalse(any(r['eligible'] for r in analysis.eligible_cells(train).values()))

    def test_metrics_ties_one_class_and_perfect_ranking(self):
        value=analysis.metric_values([0,1],[.5,.5])
        self.assertEqual(value['auroc'],.5);self.assertEqual(value['fpr_at_80_tpr'],.8)
        self.assertAlmostEqual(value['ppv_at_10pct_prevalence'],.1)
        perfect=analysis.metric_values([0,0,1,1],[0,.1,.9,1]);self.assertEqual(perfect['auroc'],1.)
        self.assertEqual(perfect['fpr_at_80_tpr'],0.);self.assertEqual(perfect['ppv_at_10pct_prevalence'],1.)
        for y in ([0,0],[1,1],[]):
            bad=analysis.metric_values(y,[.5]*len(y));self.assertIsNotNone(bad['undefined_reason']);self.assertIsNone(bad['auroc'])

    def test_bootstrap_preserves_paired_cluster_predictions_and_is_repeatable(self):
        rows=[{'trial_id':f'{g}_{y}','split_group':str(g),'y':y,'scores':{'a':.5,'b':.5}} for g in range(5) for y in (0,1)]
        first=analysis.bootstrap(rows,'a',comparator='b',replicates=100)
        self.assertEqual(first,analysis.bootstrap(rows,'a',comparator='b',replicates=100))
        self.assertEqual(first['valid_replicates'],100)
        self.assertEqual(first['interval95']['auroc'],[0.,0.])
        self.assertTrue(first['degenerate_interval']['auroc'])

    def test_train_only_vocabulary_and_orthogonal_probe(self):
        d=assembled();train=[r for r in d['rows'] if r['split_group']!='t0'];test=[r for r in d['rows'] if r['split_group']=='t0']
        for row in test:row['available_prompt_text']+=' heldoutsecretword'
        _,info=analysis.fit_predict(d,train,test,'prompt_tfidf')
        self.assertNotIn('heldoutsecretword',info['vocabulary'])
        a=[r for r in train if r['factors']['arm']=='refusal_ablated'];b=[r for r in test if r['factors']['arm']=='refusal_ablated']
        _,info=analysis.fit_predict(d,a,b,'ablated_orthogonal');self.assertAlmostEqual(info['effective_direction_dot_rhat'],0.)
        with self.assertRaisesRegex(ValueError,'Intact-only'):analysis.fit_predict(d,a,b,'intact_rhat')

    def test_complete_synthetic_engine_covers_all_regimes_and_both_denominators(self):
        parts=list(dataset());parts[4][0]['outcome_category']='TRUNCATED_OR_MALFORMED'
        d=contract.assemble(*parts,source_bindings={'synthetic_inputs':'c'*64},synthetic=True)
        report=analysis.analyze_synthetic(d,bootstrap_replicates=10)
        self.assertEqual({r['regime'] for r in report['summaries']},{'A','B','C'})
        self.assertEqual({r['denominator'] for r in report['summaries']},{'itt','strict'})
        self.assertTrue(all(not r['held_out_labels_used_for_selection'] for r in report['training_only_cell_decisions']))
        self.assertFalse(report['phase3_authorized']);self.assertFalse(report['empirical_analysis_complete'])
        parts=list(dataset())
        for row in parts[4]:row['outcome_category']='SAFE_ROUTE'
        empty=contract.assemble(*parts,source_bindings={'synthetic_inputs':'c'*64},synthetic=True)
        unavailable=analysis.analyze_synthetic(empty,bootstrap_replicates=10)
        self.assertTrue(all(r['status'].startswith('UNAVAILABLE') for r in unavailable['summaries']))
        d['synthetic']=False
        with self.assertRaises(ValueError):analysis.analyze_synthetic(d,bootstrap_replicates=10)

    def test_post_join_outcome_or_vector_mutation_rejected_before_fitting(self):
        for changed in ('outcome','vector'):
            d=assembled()
            if changed=='outcome':d['rows'][0]['outcome_category']='ESCALATE_OR_REFUSE'
            else:d['activations'][0,7]+=1
            with self.subTest(changed=changed),self.assertRaisesRegex(ValueError,'changed'):
                analysis.analyze_synthetic(d,bootstrap_replicates=10)

    def test_admission_does_not_treat_missing_backend_as_pass(self):
        value=admission.recompute(ROOT,{'phase3_inputs':{'exp11_selection':'pending'},'exp11_selection':'pending',
                                      'exp31_selection':'pending','capture_selection':'pending'})
        self.assertEqual(value['state'],'PENDING_INDEPENDENT_BACKEND_AUDITOR');self.assertFalse(value['authorized'])


if __name__=='__main__':unittest.main()
