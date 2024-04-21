# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer

from factscore.utils import convert_model_to_int8_on_gpu
from factscore.lm import LM

class CLM(LM):
    def __init__(self, model_name, model_dir, cache_file=None):
        self.model_name = model_name
        self.model_dir = model_dir
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        self.model = convert_model_to_int8_on_gpu(self.model, device='cuda')
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_dir, use_fast=True)
        # time.sleep(2)

    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False, verbose=False):
        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]

        input_ids = self.tokenizer(prompts).input_ids
        if verbose:
            input_ids = tqdm(input_ids)

        generations = []
        scores = []

        for _i, curr_input_ids in enumerate(input_ids):
            if len(curr_input_ids) > max_sequence_length - max_output_length:
                curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
            curr_input_ids = torch.LongTensor([curr_input_ids]).cuda()
            gen_outputs = self.model.generate(
                curr_input_ids,
                max_length=curr_input_ids.shape[1]+max_output_length,
                return_dict_in_generate=True,
                output_scores=True
            )

            gen_tokens = gen_outputs["sequences"]
            # saving the logits for the very first token
            gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
            gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()

            if verbose and len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)

            if self.model_name.startswith("llama-sni"):
                gen = gen.split("</s>")[0]
                
            generations.append(gen)
            scores.append(gen_scores)

        if verbose:
            print("generations:", generations)
            print("scores:", scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]
        
        return generations, scores

    # this *_batch function aims to generate a batch of generations for speedup
    # not tested/used currently 
    def _generate_batch(self, prompts, max_sequence_length=2048, max_output_length=128,
                        end_if_newline=False, end_if_second_newline=False, verbose=False):
        

        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]
        
        # print(self.tokenizer.pad_token_id)
        # print("truncation side", self.tokenizer.truncation_side)
        # print("padding side", self.tokenizer.padding_side)
        # print("pad token", self.tokenizer.pad_token)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length-max_output_length).input_ids.cuda()
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        # print("input ids", input_ids.shape)
        # tensor dtype is int64
        # print("model dtype", self.model.dtype)
        
        if verbose:
            input_ids = tqdm(input_ids)
            print("model dtype", self.model.dtype)
            print("input ids dtype", input_ids.dtype)
            print("tokenizer pad token", self.tokenizer.pad_token)

        gen_outputs = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1]+max_output_length,
            return_dict_in_generate=True,
            output_scores=True
        )
        generations = self.tokenizer.batch_decode(gen_outputs["sequences"][:, input_ids.shape[-1]:])
        scores = gen_outputs["scores"][0].detach().cpu().numpy()
        scores = [score for score in scores]

        if end_if_newline:
            generations = [gen.split("\n")[0].strip() for gen in generations]
        elif end_if_second_newline:
            generations = ["\n".join(gen.split("\n")[:2]).strip() for gen in generations]
        if self.model_name.startswith("llama-sni"):
            generations = [gen.split("</s>")[0] for gen in generations]

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]
        
        return generations, scores