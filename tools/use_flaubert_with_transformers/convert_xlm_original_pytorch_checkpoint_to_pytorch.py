# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert OpenAI GPT checkpoint."""

from __future__ import absolute_import, division, print_function

import os
import argparse
import json
from io import open

import torch
import numpy

from transformers import CONFIG_NAME, WEIGHTS_NAME
from transformers.tokenization_xlm import VOCAB_FILES_NAMES

import logging
logging.basicConfig(level=logging.INFO)

def convert_xlm_checkpoint_to_pytorch(xlm_checkpoint_path, pytorch_dump_folder_path):
    # Load checkpoint
    chkpt = torch.load(xlm_checkpoint_path, map_location='cpu')

    state_dict = chkpt['model']

    # We have the base model one level deeper than the original XLM repository
    two_levels_state_dict = {}
    for k, v in state_dict.items():
        if 'pred_layer' in k:
            two_levels_state_dict[k] = v
        else:
            two_levels_state_dict['transformer.' + k] = v

    config = chkpt['params']
    config = dict((n, v) for n, v in config.items() if not isinstance(v, (torch.FloatTensor, numpy.ndarray)))

    # Remove keys in config file
    keys_to_remove = ['command', 'data_path', 'dump_path', 'exp_id', 'exp_name', 'hyp_path', 'mono_dataset', 
                    'reload_checkpoint', 'master_addr', 'sinusoidal_embeddings', 'n_gpu_per_node', 
                    'lambda_clm_config', 'debug', 'lambda_pc', 'lambda_mlm_config', 'lambda_ae_config',
                    'epoch_size', 'validation_metrics', 'global_rank', 'accumulate_gradients', 'para_dataset',
                    'lambda_mt_config', 'lambda_bt_config', 'max_len', 'max_epoch', 'eval_only', 'split_data',
                    'pc_steps', 'bt_src_langs', 'node_id', 'time_limit', 'debug_slurm', 'reload_model', 'world_size',
                    'bt_steps', 'debug_train', 'local_rank', 'multi_gpu', 'beam_size', 'lambda_pc_config', 'ref_paths',
                    'is_slurm_job', 'master_port', 'length_penalty', 'is_master', 'stopping_criterion', 'n_nodes',
                    'eval_bleu', 'batch_size', 'save_periodic', 'multi_node', 'early_stopping', 'ae_steps', 'optimizer',
                    'use_memory', 'reload_emb', 'lambda_mt', 'lambda_bt', 'asm', 'mt_steps', 'lambda_clm', 'context_size',
                    'lambda_ae', 'lambda_mlm', 'clm_steps', 'min_count', 'layerdrop']
    for k in keys_to_remove:
        if k in config:
            del config[k]
    # config['layerdrop'] = 0.0
    
    vocab = chkpt['dico_word2id']
    vocab = dict((s + '</w>' if s.find('@@') == -1 and i > 13 else s.replace('@@', ''), i) for s, i in vocab.items())

    # Save pytorch-model
    pytorch_weights_dump_path = pytorch_dump_folder_path + '/' + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + '/' + CONFIG_NAME
    pytorch_vocab_dump_path = pytorch_dump_folder_path + '/' +  VOCAB_FILES_NAMES['vocab_file']

    print("Save PyTorch model to {}".format(pytorch_weights_dump_path))
    torch.save(two_levels_state_dict, pytorch_weights_dump_path)

    print("Save configuration file to {}".format(pytorch_config_dump_path))
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2, sort_keys=True) + "\n")


    print("Save vocab file to {}".format(pytorch_vocab_dump_path))
    with open(pytorch_vocab_dump_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(vocab, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--xlm_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path the official PyTorch dump.")
    parser.add_argument("--pytorch_dump_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_xlm_checkpoint_to_pytorch(args.xlm_checkpoint_path, args.pytorch_dump_folder_path)
