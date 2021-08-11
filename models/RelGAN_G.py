# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : RelGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.generator import LSTMGenerator
from models.relational_rnn_general import RelationalMemory


class RelGAN_G(LSTMGenerator):
    def __init__(self, mem_slots, num_heads, head_size, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,
                 gpu=False):
        super(RelGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'relgan'

        self.temperature = 1.0  # init value is 1.0

        if cfg.if_linear_embedding:
            self.embeddings = nn.Linear(vocab_size, embedding_dim)
        # # previous standard solution
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # bing: for the generator training, we use linear layer instead


        if cfg.model_type == 'LSTM':
            # LSTM
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, batch_first=True)
            self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)
        else:
            # RMC
            self.hidden_dim = mem_slots * num_heads * head_size
            if cfg.if_use_context_attention_aware:
                # ==== approach 0 ====:
                # self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=(embedding_dim+self.hidden_dim),
                #                              num_heads=num_heads, return_all_outputs=True)
                # ==== approach 1 ====:
                self.hidden2embed = nn.Linear(self.hidden_dim, embedding_dim) # we have this, we omit th
                self.layer_norm = nn.LayerNorm(embedding_dim)
                self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=(embedding_dim),
                                             num_heads=num_heads, return_all_outputs=True)

            else:
                self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=embedding_dim,
                                         num_heads=num_heads, return_all_outputs=True)
            self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        self.init_params()
        pass
    
    
    

    def init_hidden(self, batch_size=cfg.batch_size):
        # lstm init: by zero
        if cfg.model_type == 'LSTM':
            h = torch.zeros(1, batch_size, self.hidden_dim)
            c = torch.zeros(1, batch_size, self.hidden_dim)

            if self.gpu:
                return h.cuda(), c.cuda()
            else:
                return h, c
        else:
            """init RMC memory"""
            memory = self.lstm.initial_state(batch_size)
            memory = self.lstm.repackage_hidden(memory)  # detch memory at first
            return memory.cuda() if self.gpu else memory

    def step(self, inp, hidden, next_token_requires_grad=False, context=None):
        """
        RelGAN step forward
        :param inp: [batch_size]
        :param hidden: memory size
        :return: pred, hidden, next_token, next_token_onehot, next_o
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: [batch_size], next sentence token
            - next_token_onehot: batch_size * vocab_size, not used yet
            - next_o: batch_size * vocab_size, not used yet
        """
        # inp.shape [8], it is like the batch size
        emb = self.embeddings(inp).unsqueeze(1) # original embedding shape: batch_size * len[value: 1] * embedding_dim

        emb = self.stance_aware_setting(context, emb)


        out, hidden = self.lstm(emb, hidden)
        gumbel_t = self.add_gumbel(self.lstm2out(out.squeeze(1)))
        if next_token_requires_grad:
            next_token = torch.argmax(gumbel_t, dim=1)
        else:
            next_token = torch.argmax(gumbel_t, dim=1).detach()
        # next_token_onehot = F.one_hot(next_token, cfg.vocab_size).float()  # not used yet
        next_token_onehot = None

        pred = F.softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * vocab_size
        # next_o = torch.sum(next_token_onehot * pred, dim=1)  # not used yet
        next_o = None

        return pred, hidden, next_token, next_token_onehot, next_o


    def sample(self, num_samples, batch_size, one_hot=False, start_letter=cfg.start_letter,
               hidden=None,
               next_token_requires_grad = False,
               attention_context = None):
        """
        Sample from RelGAN Generator
        - one_hot: if return pred of RelGAN, used for adversarial training
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        # bing: all_preds: the full prediction of the word with max-length and vocabulary size
        # todo: here, if num_batch>=2, it will cause some problems for the inported hidden state setting
        global all_preds
        # add by bing for samples
        global samples

        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        if hidden != None:
            # print("hidden is not None") # testing passed for the generation
            # only test it when we input the specific hidden state rather than the random generation in pretraining
            assert num_batch == 1 # for replaced hidden state setting!!
        # print(f"the self.max_seq_len is {self.max_seq_len}") # it is from the real-data setting
        if next_token_requires_grad:
            # RuntimeError: only Tensors of floating point and complex dtype can require gradients
            samples = torch.zeros(num_batch * batch_size, self.max_seq_len).to(cfg.device)
            samples.requires_grad = True # after set this, the grad is recorded! to my memory, we should also remove the long setting
        else:
            samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long().to(cfg.device) # add the transfer to device by bing, remove the .long()
        # todo: .long() will make the samples not required grads any more?? check the variable in the sample: be areful! very important
        if one_hot:
            all_preds = torch.zeros(batch_size, self.max_seq_len, self.vocab_size)
            if self.gpu:
                all_preds = all_preds.cuda()
                # print(f" all_preds 1: {all_preds.requires_grad}")
        for b in range(num_batch):
            # added by bing for the related setting!
            # bing, todo: for conditional text generation, we change this part!
            if hidden == None:
                hidden = self.init_hidden(batch_size)
            else:
                pass

            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()


            for i in range(self.max_seq_len):

                if cfg.if_linear_embedding:
                    inp = F.one_hot(inp, cfg.extend_vocab_size).float()  # shape: 8, 26685: batch_size*1 -> batch_size*vocab_size

                # pred size: batch_size * vocab_size: just for one position and the value should be the prob
                # attention_context: # batch_size * seq-len[value: 1] * (hidden_dim)
                pred, hidden, next_token, _, _ = self.step(inp, hidden, next_token_requires_grad, attention_context)
                # samples are the output setting: where the txt file exist
                # next token: the token for the next postion with the whole batch setting!
                # print(pred.requires_grad) # true
                # print(next_token.requires_grad) # true
                # print(f" in the inner part: {samples.requires_grad}")
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                # print(f" in the inner part: {samples.requires_grad}") # True: if the sample is not long and requires_grad=True
                if one_hot:
                    all_preds[:, i] = pred
                    # print(f" in the all_preds: {all_preds.requires_grad}")
                inp = next_token # batch_size*1
        # print(samples.requires_grad) # True
        samples = samples[:num_samples]  # num_samples * seq_len, return the needed sentences
        # print(samples.requires_grad) # True
        # exit(0)
        if one_hot:
            return all_preds  # batch_size * seq_len * vocab_size
        return samples

    @staticmethod
    def add_gumbel(o_t, eps=1e-10, gpu=cfg.CUDA):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.zeros(o_t.size())
        if gpu:
            u = u.cuda()

        u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t
