#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from argparse import Namespace

import onmt
from prior_sampling import PriorSampling

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

import pickle
import torchtext
import torch
import codecs
import random
import numpy as np

from model_builder_custom import load_test_model_with_projection_layer

class VerseGenerator:
    def __init__(self, modelFile, entropy_threshold):

        
        opt = Namespace(models=[modelFile], data_type='text',
                        fp32=False, batch_size=1)

        self.fields, self.model, self.model_opt = \
            load_test_model_with_projection_layer(opt)

        self.vocab = self.fields["tgt"].base_field.vocab
        
        self.batch_size_encoder = opt.batch_size
        self.n_batches_decoder = 16
        self.batch_size_decoder = 16
        self.max_length = 30
        self.sampling_temp = 0.8
        self.entropy_threshold = entropy_threshold

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def generateCandidates(self, previous, rhymePrior, nmfPrior):
        if rhymePrior is not None:
            rhymePrior = torch.from_numpy(rhymePrior).float().to(self.device)
            # print("rhymePrior")
            # print(rhymePrior) #eg tensor([2.6316e-22, 2.6316e-22, 2.6316e-22,  ..., 2.6316e-22, 2.6316e-22, 2.6316e-22])
        if nmfPrior is not None:
            nmfPrior = torch.from_numpy(nmfPrior).float().to(self.device)

            
        if previous is None:
            # when no previous verse is defined (first verse of
            # the poem), encode the phrase "unk unk unk" - this
            # works better for initialization of the decoder than                
            # an all-zero or random hidden encoder state
            src = torch.tensor([0, 0, 0])
        else:
            src = torch.tensor([self.vocab.stoi[w] for w in previous])

        src = src.view(-1,1,1).to(self.device)
        src_lengths = torch.tensor([src.size(0)]).to(self.device)

        #run encoder
        enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths)
        results = {
            "predictions": [],
            "scores": [],
        }

        #variables to restart with each batch
        src_init = src
        src_lengths_init = src_lengths
        memory_bank_init = memory_bank
        enc_states_init = enc_states

        for n_batch in range(self.n_batches_decoder):
            #initialize decoder with encoder states
            self.model.decoder.init_state(src_init, memory_bank_init, enc_states_init)


            decode_strategy = PriorSampling(
                batch_size=self.batch_size_encoder,
                pad=self.vocab.stoi[self.fields["tgt"].base_field.pad_token],
                bos=self.vocab.stoi[self.fields["tgt"].base_field.init_token],
                eos=self.vocab.stoi[self.fields["tgt"].base_field.eos_token],
                sample_size=self.batch_size_decoder,
                min_length=0,
                max_length=self.max_length,
                return_attention=False,
                block_ngram_repeat=1,
                exclusion_tokens={},
                sampling_temp=self.sampling_temp,
                keep_topk=-1,
                entropy_threshold=self.entropy_threshold,
            )

            #initialize sampler
            src_map = None
            fn_map_state, memory_bank, memory_lengths, src_map = \
                decode_strategy.initialize(memory_bank_init, src_lengths_init, src_map)
            if fn_map_state is not None:
                self.model.decoder.map_state(fn_map_state)

                #at beginning repeat both priors, size of decoder batch
                if rhymePrior is not None:
                    rhymePrior_batch = rhymePrior.repeat(self.batch_size_decoder, 1)
                    # print('rhymePrior_batch')
                    # print(rhymePrior_batch) #eg tensor([[6.2500e-22, 6.2500e-22, 6.2500e-22,  ..., 6.2500e-22, 6.2500e-22,
                                                        #  6.2500e-22],
                                                        # [6.2500e-22, 6.2500e-22, 6.2500e-22,  ..., 6.2500e-22, 6.2500e-22,
                                                        #  6.2500e-22],
                                                        # [6.2500e-22, 6.2500e-22, 6.2500e-22,  ..., 6.2500e-22, 6.2500e-22,
                                                        #  6.2500e-22],
                                                        # ...,
                                                        # [6.2500e-22, 6.2500e-22, 6.2500e-22,  ..., 6.2500e-22, 6.2500e-22,
                                                        #  6.2500e-22],
                                                        # [6.2500e-22, 6.2500e-22, 6.2500e-22,  ..., 6.2500e-22, 6.2500e-22,
                                                        #  6.2500e-22],
                                                        # [6.2500e-22, 6.2500e-22, 6.2500e-22,  ..., 6.2500e-22, 6.2500e-22,
                                                        #  6.2500e-22]])
                if nmfPrior is not None:
                    nmfPrior_batch = nmfPrior.repeat(self.batch_size_decoder, 1)
                    # print("nmfPrior_batch")
                    # print(nmfPrior_batch)
            

            for step in range(self.max_length):
                decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
                dec_out, dec_attn = self.model.decoder(
                    decoder_input, memory_bank, memory_lengths=memory_lengths, step=step
                )
                if "std" in dec_attn:
                    attn = dec_attn["std"]
                else:
                    attn = None
                log_probs = self.model.generator(dec_out.squeeze(0))

                if step == 0 and rhymePrior is not None:
                    decode_strategy.advance(log_probs, attn, prior=rhymePrior_batch)
                elif nmfPrior is not None:
                    decode_strategy.advance(log_probs, attn, prior=nmfPrior_batch)
                else:
                    decode_strategy.advance(log_probs, attn)

                any_finished = decode_strategy.is_finished.any()
                if any_finished:
                    decode_strategy.update_finished()

                    if decode_strategy.done:
                        break
                        
                select_indices = decode_strategy.select_indices

                if any_finished:
                    if isinstance(memory_bank, tuple):
                        memory_bank = tuple(x.index_select(1, select_indices)
                                            for x in memory_bank)
                    else:
                        memory_bank = memory_bank.index_select(1, select_indices)

                    memory_lengths = memory_lengths.index_select(0, select_indices)
            

                    #if any finished need to update nmfprior
                    if nmfPrior is not None:
                        nmfPrior_batch = nmfPrior.repeat(len(select_indices), 1)
                    

                    self.model.decoder.map_state(
                        lambda state, dim: state.index_select(dim, select_indices))

            results["scores"].extend(decode_strategy.scores[0])
            results["predictions"].extend(decode_strategy.predictions[0])
            # print('decode_strategy.predictions[0]')
            # print(decode_strategy.predictions[0]) #eg [tensor([2804,    9,  777, 1074,  194,    9,    3]), tensor([1233,   63,   12,   65,  346,   13,   83,    3]), tensor([2804,   63,   22,  131,   13,   40,   42,    3]), tensor([8928,    6,   43,   59, 5037,   13,   60,   80,    3]), tensor([8928,   37,    7,  815,    7,  131,   13,   40,  273,   42,    3]), tensor([268,   9,   7,  65, 484,  34, 125,   5,  34,  49,   3]), tensor([1233,   13,   56, 6002, 2043,   63,   12, 1380,   13,   40,   42,    3]), tensor([268,   7, 131,  41,  67,   5,  44,  65, 213,  13,  49,   3]), tensor([ 1233,     6,    27, 11901,    22,    97,  1044,     8,    89,    44,...]
        

        allSents = []
        # print("results['predictions']")
        # print(results['predictions']) #eg[[tensor([ 416,  511,   17,    5, 2524,   69, 7248,    5, 9478,    3]), tensor([10386,    21,  6028,     6,     7,  1404,     8, 12066,  6390,     3]), tensor([2646,    7,   78,    7,   35,   38,    5,   40,   38,  436,    3]), tensor([ 416, 1435,   69, 1016,  109,   35,   17,  159,   34,   29,    3]), tensor([2646, 2421,    9,   33,   53,   22, 4245,    6,   21, 5308,    3]), tensor([ 416,  973,    9,   70, 6018,   62,  706,  119,    8, 2905,   37,    3]), tensor([ 416,  113,  511,   14,    8,    5, 1493, 1618,    9,   33,   37,    3]), tensor([4605,    6,   11, 2169,  108,    6,   43, 1937,   22,   13,  417,    3]), tensor([ 2646,  4669,     8,  2609,     9,    19,     5, 13270,  2468,     9,
        #                                   #  34,    33,    28,     3]), tensor([ 416,  117,    7,  568,   12,  656,    6,   21, 4706, 3949,    7,  131,
        #                                   # 58,    3]), tensor([ 2646,    11, 11860,     6,    19,   970,    12,   947,     6,    11,
        #                                   # 596,     6,    21,   294,     3]), tensor([6497,    8, 4777,   11,  131,   12,    8,  373,   69,  373,   33, 8072,
        #                                   #  6,  335,    3]), tensor([ 2646,    11,   429,    79,   799,     5,  1311,    11,   562,    79,
        #                                   # 668,   140,  9222, 12947,     3])]]
        for sent in results['predictions']:
            #print('sent:')
            #print(sent) #tensor([ 868,   18,   11, 1047,    6,    7, 1801,   33, 1162,   33,  297,   56,  33,  396,    9,   56,  368,    6,    3])

            wsent = [self.vocab.itos[i] for i in sent[:-1]]
            # print(sent[1])
            # print("self.vocab.itos[sent[1]]:")
            # print(self.vocab.itos[sent[1]])
            # print("self.vocab.itos[sent[1]+1]:")
            # print(self.vocab.itos[sent[1]+1])
            # print("self.vocab.itos[sent[1]+2]:")
            # print(self.vocab.itos[sent[1]+2])
            wsent.reverse()
            allSents.append(wsent)
        allScores = list(results['scores'])
        return allSents, allScores
