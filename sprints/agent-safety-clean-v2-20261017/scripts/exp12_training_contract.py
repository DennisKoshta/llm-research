"""Offline serializer/adapter contracts; no optimizer or training backend.

The intended backend accumulates all turns of each complete episode. These
checks explicitly reject clipping an overlong example or omitting any turn.
"""
from copy import deepcopy
import json

from exp12_design import (NS, LORA, PRIMARY_MODEL, PRIMARY_REVISION, digest, require,
                         fingerprint, source_reference, training_grid)
from exp12_corpus_contract import complete_target


def serialize_episode(request, turns, tokenizer, *, context_limit):
    """One complete trajectory, all turn sequences, exact unpadded token accounting.

Training EOS is appended after every complete assistant turn. Original prompts
and outputs remain untouched in raw evidence. Prompt tokens have no loss; all
completion tokens (including EOS) have loss. No sequence packing or truncation.
"""
    require(type(context_limit) is int and context_limit > 0, 'Concrete model context limit required')
    require(turns and len(turns) <= 10, 'Every complete source trajectory turn required')
    history = deepcopy(request['messages']); sequences = []
    for index, turn in enumerate(turns):
        require(turn['turn_index'] == index, 'Source turn sequence omitted, duplicated or reordered')
        source_reference(turn['source'])
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        require(turn['full_prompt'] == prompt, 'SFT source prompts must independently replay, including rewritten history')
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        require(turn['prompt_token_ids'] == ids, 'Exact complete prompt IDs required')
        target = complete_target(turn['raw_output'], tokenizer, request['declared_names'])
        joined = ids + target['target_token_ids']
        require(len(joined) <= context_limit, 'INFEASIBLE_CONTEXT: full prompt plus target exceeds context; slicing is prohibited')
        terminal = turn['observation'].get('terminal') is True
        require(terminal == (index == len(turns) - 1), 'Only complete trajectories ending at their terminal observation may enter SFT')
        sequences.append({'input_ids': joined, 'labels': [-100] * len(ids) + target['target_token_ids'],
                          'prompt_token_count': len(ids), 'target': target, 'source': deepcopy(turn['source'])})
        history.extend([{'role': 'assistant', 'content': turn['raw_output']},
                        {'role': 'tool', 'content': json.dumps(turn['observation'], sort_keys=True)}])
    return {'schema': 'exp12_complete_training_example_v1', 'example_id': request['trial_id'],
            'prompt_id': request.get('source_request_id', request['trial_id']), 'model_arm': request['model_arm'],
            'corpus': request['corpus'], 'sft_valid': True, 'source_sha256': digest([request, turns]),
            'sequences': sequences, 'training_tokens': sum(len(s['input_ids']) for s in sequences),
            'supervised_tokens': sum(len(s['target']['target_token_ids']) for s in sequences),
            'context_limit': context_limit, 'execution_provenance_verified': False}


def validate_serialization(example, request, source_turns, tokenizer):
    require(example == serialize_episode(request, source_turns, tokenizer, context_limit=example['context_limit']),
            'Training example differs from whole source replay; no padding, slicing or hidden turns')
    return True


def validate_lora_inventory(configuration, inventory, selected_modules):
    """Concrete backend inventory is required; no guessed architecture suffixes.

Each row must come from an independently audited loaded-model traversal. The
contract includes every attention linear module, including hybrid-attention
variants, and excludes the output head from the all-linear arm.
"""
    require(configuration in LORA and inventory and len({r['name'] for r in inventory}) == len(inventory), 'Unique concrete module inventory required')
    require(all(type(r['is_linear']) is bool and type(r['is_attention']) is bool and type(r['is_output_head']) is bool
                for r in inventory), 'Typed architecture inventory required')
    expected = sorted(r['name'] for r in inventory if r['is_linear'] and not r['is_output_head'] and
                      (configuration == 'r32_all_linear' or r['is_attention']))
    require(expected and selected_modules == expected, 'LoRA targets must exactly cover the registered loaded architecture policy')
    return {'configuration': configuration, 'target_modules': expected, 'inventory_sha256': digest(inventory)}


def adapter_contract(student, *, intact_manifest_sha256, tokenizer_sha256, selection_sha256,
                     module_inventory_sha256, training_tokens, optimizer_steps):
    """A required future manifest definition, explicitly not a saved adapter."""
    require(student in training_grid(student['student_id'].split(':')[1]), 'Student differs from the fixed 72-cell grid')
    require(all(fingerprint(s) for s in (intact_manifest_sha256, tokenizer_sha256, selection_sha256, module_inventory_sha256)),
            'Base, tokenizer, complete corpus and loaded module inventory must be frozen')
    require(type(training_tokens) is int and training_tokens > 0 and type(optimizer_steps) is int and optimizer_steps > 0,
            'Exact matched actual compute targets required')
    return {'schema': 'exp12_adapter_manifest_contract_v1', 'student': deepcopy(student),
            'base': {'model_id': PRIMARY_MODEL, 'base_revision': PRIMARY_REVISION, 'model_arm': 'intact',
                     'materialization_sha256': intact_manifest_sha256,
                     'path': f'/weights/{NS}/checkpoints_full_residual_v2/intact'},
            'tokenizer_sha256': tokenizer_sha256, 'training_selection_sha256': selection_sha256,
            'module_inventory_sha256': module_inventory_sha256, 'expected_training_tokens': training_tokens,
            'expected_optimizer_steps': optimizer_steps, 'adapter_path': student['adapter_path'],
            'required_future_files': ['adapter_config.json', 'adapter_model.safetensors', 'training_journal.jsonl',
                                      'optimizer_step_token_audit.json', 'source_and_backend_provenance.json'],
            'required_checks': ['Immutable adapter file hashes and persistent-volume reload',
                                'Actual optimizer steps and total consumed nonpadding tokens equal selection',
                                'No repeated or omitted episode; fixed seed ordering each of one epoch',
                                'Training returns SUCCESS and started identity matches independently audited durable call',
                                'Pretraining FP8/BF16 or quantized-LoRA compatibility evidence',
                                'Inference adapter inventory and fresh five-trial pilot before full readouts'],
            'trained': False, 'persisted': False, 'execution_authorized': False}
