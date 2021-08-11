# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import numpy as np
import torch
import torch.nn as nn

import config as cfg
from metrics.bleu import BLEU
from metrics.clas_acc import ACC
from metrics.nll import NLL
from metrics.ppl import PPL
from utils.cat_data_loader import CatClasDataIter
from utils.data_loader import GenDataIter
from utils.helpers import Signal, create_logger, get_fixed_temperature
from utils.text_process import load_dict, write_tokens, tensor_to_tokens
import torch.nn.functional as F

class BasicInstructor:
    def __init__(self, opt):
        self.log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=cfg.log_filename if cfg.if_test
                                 else [cfg.log_filename, cfg.save_root + 'log.txt'])
        self.sig = Signal(cfg.signal_file)
        self.opt = opt
        self.show_config()

        self.clas = None

        # load dictionary
        self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset)
        # print("start self.train_data")
        # Dataloader
        try:
            self.train_data = GenDataIter(cfg.train_data)
            # print("finish self.train_data")
            # by bing, add the if_context for later update!
            self.test_data = GenDataIter(cfg.test_data, if_test_data=True, if_context=cfg.if_context)
            # print("finish self.test_data")
        except RuntimeError:
            print("error in the self.train_data and self.test_data building")
            pass

        try:
            self.train_data_list = [GenDataIter(cfg.cat_train_data.format(i)) for i in range(cfg.k_label)]
            self.test_data_list = [GenDataIter(cfg.cat_test_data.format(i), if_test_data=True) for i in
                                   range(cfg.k_label)]
            self.clas_data_list = [GenDataIter(cfg.cat_test_data.format(str(i)), if_test_data=True) for i in
                                   range(cfg.k_label)]

            self.train_samples_list = [self.train_data_list[i].target for i in range(cfg.k_label)]
            self.clas_samples_list = [self.clas_data_list[i].target for i in range(cfg.k_label)]
        except:
            pass

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.dis_criterion = nn.CrossEntropyLoss()
        self.clas_criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.clas_opt = None

        # Metrics
        self.bleu = BLEU('BLEU', gram=[2, 3, 4, 5], if_use=cfg.use_bleu)
        self.nll_gen = NLL('NLL_gen', if_use=cfg.use_nll_gen, gpu=cfg.CUDA)
        self.nll_div = NLL('NLL_div', if_use=cfg.use_nll_div, gpu=cfg.CUDA)
        self.self_bleu = BLEU('Self-BLEU', gram=[2, 3, 4], if_use=cfg.use_self_bleu)
        self.clas_acc = ACC(if_use=cfg.use_clas_acc)
        self.ppl = PPL(self.train_data, self.test_data, n_gram=5, if_use=cfg.use_ppl)
        self.all_metrics = [self.bleu, self.nll_gen, self.nll_div, self.self_bleu, self.ppl]

    def _run(self):
        print('Nothing to run in Basic Instructor!')
        pass

    def _test(self):
        pass

    def init_model(self):
        if cfg.dis_pretrain:
            self.log.info(
                'Load pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))
            self.dis.load_state_dict(torch.load(cfg.pretrained_dis_path, map_location='cuda:{}'.format(cfg.device)))
        # revised by bing
        # if cfg.gen_pretrain: previous
        if cfg.if_use_saved_gen:
            # comment (not cfg.if_pretrain_mle) since sometimes, we want to cotinue training!
            self.log.info('Load MLE pre-trained generator: {}'.format(cfg.pretrained_gen_path_used))
            if cfg.device == torch.device("cpu"):
                self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path_used))
            else:
                self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path_used, map_location='cuda:{}'.format(cfg.device)))

        if cfg.CUDA:
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()

    @staticmethod
    def compute_hiddens(model, seqs, if_detach=True, if_list2tensor_quick_computation=False):
        if isinstance(seqs, list):
            # bing: debug setting, we have to ensure there are more than one seq in seqs
            if if_list2tensor_quick_computation and len(seqs) > 1:
                # ==== condition 0 ====: from list input, where in the seq-context-one-long setting, we pad the different seq and train it
                # previously, we set: 1+; but, after thinking, we should omit it and set our own plans: be careful of it and set it well! --- be careful!
                seq_max_len = max([len (i) for i in seqs]) # list of m(length_of_one_long_seq) or m*vocab if linear embedding and get m
                # torch.ones --> torch.float32, we should use long to get the required dtype
                padded_front_idx = [(torch.ones(seq_max_len - len(i))*cfg.padding_idx).long().to(cfg.device) for i in seqs]
                if cfg.if_linear_embedding:
                    padded_front_idx = [F.one_hot(one_seq, cfg.extend_vocab_size).float() for one_seq in
                            padded_front_idx]  # [54,vocab_size]-<[54]
                # unsqueeze to add the 1*batch dimension for the concatenation!
                padded_seqs = [torch.cat([padded_front_idx[i], seqs[i]], dim=0).unsqueeze(0) for i in range(len(seqs))]
                seqs = torch.cat(padded_seqs, dim=0)
                # repeat the torch processing
                batch_size = seqs.shape[0]
                hidden = model.init_hidden(batch_size)  # batch_size*1*512
                # print(seqs.shape, hidden.shape)
                _, hiddens = model.forward(seqs, hidden, need_hidden=True)
            else:
                # ===== condition 1 ====: from list input, we do not pad, and train it one by one
                hiddens = []
                for seq in seqs:
                    # seq shape, when linear embedding with one-hot, dimension is origina-seq-len*vocab_size * [545, 26685]
                    # seq shape, when default embedding without one-hot, dim share is [545]
                    batch_size = 1  # seq_one_long.shape[0], it should be (1*dynamic-edits)
                    hidden = model.init_hidden(batch_size) # 1, 1, 512
                    # print(seq.view(batch_size, -1).shape) # shape: 1, 545 # preivous we use seq=seq.view()
                    # set it when we have the stance training with only one post: 1*max_seq_len*vocab
                    if seq.shape[0] == 1:
                        seq_unsqueezed = seq
                    else:
                        seq_unsqueezed = seq.unsqueeze(0) # a universal solution for the w/ and w/o conditions for batch=1 # (1, a) from [a] or (1, a, b) from (a, b): linear embed
                    _, hidden = model.forward(seq_unsqueezed, hidden, need_hidden=True)

                    # if if_detach:
                    #     hiddens.append(hidden.detach())
                    # else:
                    #     hiddens.append(hidden)
                    hiddens.append(hidden)
                hiddens = torch.cat(hiddens, dim=0)  # batch_size*hidden_state
        else:
            # ==== condition 2 ====: directly compute from the tensor
            assert isinstance(seqs, torch.Tensor) == True # shape: num_edit*max_len*vocab_size
            batch_size = seqs.shape[0]
            hidden = model.init_hidden(batch_size) # batch_size*1*hidden_dimension
            _, hiddens = model.forward(seqs, hidden, need_hidden=True) # batch_size*1*hidden_dimension

        if if_detach:
            return hiddens.detach()
        else:
            return hiddens

    def train_gen_epoch(self, model, data_loader, criterion, optimizer, compute_hidden=False, if_list2tensor_quick_computation=False,
                        if_relevancy_attention_aware_context = False,
                        if_one_hot_in_one_batch_in_malcom_exp = False
                        ):
        """

        :param model:
        :param data_loader:
        :param criterion:
        :param optimizer:
        :param compute_hidden:
        :param if_list2tensor_quick_computation: when we have different length seq, we do the padding and unification
        :return:
        """
        total_loss = 0
        num_batches = 0
        # ==== condition 0 ====: we compute the hidden from previous one long tensor
        if compute_hidden and not if_relevancy_attention_aware_context:
            for index in range(0, len(data_loader), cfg.batch_size):
                # print(index)
                # ensure the valid indexing
                if (index + cfg.batch_size) <= len(data_loader):
                    num_batches += 1
                    one_data_batch = data_loader[index: index+cfg.batch_size]

                    # note, once you get the flatten 1-dimensional tensor, be careful of view(1, -1) setting
                    # prev
                    # inp = torch.cat([one_dict['input'].view(1, -1) for one_dict in one_data_batch], dim=0)
                    # present
                    # inp = []
                    # for one_dict in one_data_batch:
                    #     if len(one_dict['input'].shape) == 1:
                    #         inp.append(one_dict['input'].view(1, -1))
                    #     else:
                    #         print()
                    #         inp.append(one_dict['input']) # torch.Size([248, 4322]) torch.Size([8, 1, 512]) # error
                    # inp = torch.cat(inp, dim=0)
                    # present 2: more universal solution
                    if if_one_hot_in_one_batch_in_malcom_exp:
                        # F.one_hot(i, cfg.extend_vocab_size).float()
                        inp = torch.cat([F.one_hot(one_dict['input'], cfg.extend_vocab_size).float().unsqueeze(0) for one_dict in one_data_batch], dim=0)
                        one_long_context = [F.one_hot(one_dict['one_long_context'], cfg.extend_vocab_size).float() for one_dict in one_data_batch]
                    else:
                        inp = torch.cat([one_dict['input'].unsqueeze(0) for one_dict in one_data_batch], dim=0)
                        one_long_context = [one_dict['one_long_context'] for one_dict in one_data_batch]
                    target = torch.cat([one_dict['target'].view(1, -1) for one_dict in one_data_batch], dim=0)

                    if cfg.CUDA:
                        inp, target = inp.cuda(), target.cuda()

                    # ==== compute the hiddens ====:
                    # every seq should already be on device
                    hidden = self.compute_hiddens(model, one_long_context, if_list2tensor_quick_computation=if_list2tensor_quick_computation) # batch_size*1*hidden
                    # ==== end ====

                    # todo for bing, a little different from the step function?
                    # by checking, the forward comes from the basic relational_rnn_general generator setting!
                    # print(inp.shape, hidden.shape) # batch_size*seq_len*[vocab_sizeInLinearEmbedding]
                    pred = model.forward(inp, hidden)  # (seq_len*batch)*vocab_size
                    # print(inp.shape, hidden.shape, target.shape, pred.shape)
                    # target.view(-1) shape: (batch_size*max_len_seq)
                    loss = criterion(pred, target.view(-1))
                    # self.log.info(f"one loss in the inner train_gen_epoch mle training is {loss}") # tested pass, we can get the loss
                    self.optimize(optimizer, loss, model)
                    total_loss += loss
                    # self.log.info("finish one dataloader in pretraining generator")

        # ==== condition 1 ====: we do not use hiiden, the same as the relgan paper
        elif not compute_hidden and not if_relevancy_attention_aware_context:
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target'] # batch_size*max_len_seq
                assert 'one_long_context' not in data
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()

                hidden = model.init_hidden(data_loader.batch_size) # batch_size*1*hidden_state
                # todo for bing, a little different from the step function?
                # by checking, the forward comes from the basic relational_rnn_general generator setting!
                pred = model.forward(inp, hidden) # (seq_len*batch)*vocab_size
                # print(inp.shape, hidden.shape, target.shape, pred.shape)
                # target.view(-1) shape: (batch_size*max_len_seq)
                loss = criterion(pred, target.view(-1))
                # print("get the loss")
                # exit(0)
                self.optimize(optimizer, loss, model)
                total_loss += loss
                # self.log.info("finish one dataloader in pretraining generator")
        # ==== condition 2 ====: we compute the attention-aware context
        elif if_relevancy_attention_aware_context:
            for index in range(0, len(data_loader), cfg.batch_size):
                # ensure the valid indexing
                if (index + cfg.batch_size) <= len(data_loader):
                    num_batches += 1
                    one_data_batch = data_loader[index: index+cfg.batch_size]

                    inp = torch.cat([one_dict['input'].unsqueeze(0) for one_dict in one_data_batch], dim=0) # batch_size*max_seq_len*[vocab_size]
                    target = torch.cat([one_dict['target'].view(1, -1) for one_dict in one_data_batch], dim=0) # batch_size*max_seq_len

                    if cfg.CUDA:
                        inp, target = inp.cuda(), target.cuda()

                    prev_train_seq = [one_dict['prev_train_seq'] for one_dict in one_data_batch] # batch_size length list of tensor: (num_edit-1)*max_seq_len
                    matched_context = [one_dict['matched_context'] for one_dict in
                                      one_data_batch]  # batch_size length list of tensor: 1*max_seq_len
                    match_relevancy_score = [one_dict['match_relevancy_score'] for one_dict in
                                      one_data_batch]  # batch_size length list of tensor: 1*(num_edit-1)

                    weighted_hiddens = self.weighted_context_computation(model, prev_train_seq, matched_context, match_relevancy_score)
                    # context_aware generation: # batch_size * seq-len * (hidden_dim)
                    pred = model.forward(inp, weighted_hiddens, context=weighted_hiddens.repeat(1, cfg.max_seq_len, 1))
                    loss = criterion(pred, target.view(-1))
                    self.optimize(optimizer, loss, model)
                    total_loss += loss

        else:
            self.log.info("**** no correct settting for training generator and exit ****")
        return total_loss / len(data_loader)

    def weighted_context_computation(self, model, prev_train_seq, matched_context, match_relevancy_score):
        """
        we compute the attention-aware weighted sum for the testing!
        :param model:
        :param prev_train_seq:
        :param matched_context:
        :param match_relevancy_score:
        :return:
        """
        matched_prev_train_seq_and_context = [torch.cat([prev_train_seq[i], matched_context[i]], dim=0) for i in
                                              range(len(prev_train_seq))]  # batch_size list of tensor: num_edit*max_seq
        matched_attention_score = [torch.cat([one_relevancy_score.view(1, -1), torch.ones(1, 1).to(cfg.device)], dim=1)
                                   for one_relevancy_score in match_relevancy_score]  # batch_size list: 1*(num_edit)
        weighted_hiddens = []  # batch_size list of: 1*hidden
        for i in range(len(matched_prev_train_seq_and_context)):
            one_prev_train_seq_and_context = matched_prev_train_seq_and_context[i]  # num_edit*max_seq*[vocab_size]
            seq_hiddens = self.compute_hiddens(model, one_prev_train_seq_and_context).squeeze(1)  # num_edit*1*hidden_dimension->num_edit*hidden_state

            one_full_attention_score = matched_attention_score[i]  # 1*(num_edit)
            attention_weights = F.softmax(one_full_attention_score, dim=1)  # 1*(num_edit)
            weighted_hidden_state = torch.matmul(attention_weights, seq_hiddens)  # 1*hidden_state
            weighted_hiddens.append(weighted_hidden_state)
        weighted_hiddens = torch.cat([one_hidden.unsqueeze(0) for one_hidden in weighted_hiddens],
                                     dim=0)  # batch_size*1*hidden
        return weighted_hiddens

    def train_dis_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        total_acc = 0
        total_num = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()

            pred = model.forward(inp)
            loss = criterion(pred, target)
            self.optimize(optimizer, loss, model)

            total_loss += loss.item()
            total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
            total_num += inp.size(0)

        total_loss /= len(data_loader)
        total_acc /= total_num
        return total_loss, total_acc

    def train_classifier(self, epochs):
        """
        Classifier for calculating the classification accuracy metric of category text generation.

        Note: the train and test data for the classifier is opposite to the generator.
        Because the classifier is to calculate the classification accuracy of the generated samples
        where are trained on self.train_samples_list.

        Since there's no test data in synthetic data (oracle data), the synthetic data experiments
        doesn't need a classifier.
        """
        import copy

        # Prepare data for Classifier
        clas_data = CatClasDataIter(self.clas_samples_list)
        eval_clas_data = CatClasDataIter(self.train_samples_list)

        max_acc = 0
        best_clas = None
        for epoch in range(epochs):
            c_loss, c_acc = self.train_dis_epoch(self.clas, clas_data.loader, self.clas_criterion,
                                                 self.clas_opt)
            _, eval_acc = self.eval_dis(self.clas, eval_clas_data.loader, self.clas_criterion)
            if eval_acc > max_acc:
                best_clas = copy.deepcopy(self.clas.state_dict())  # save the best classifier
                max_acc = eval_acc
            self.log.info('[PRE-CLAS] epoch %d: c_loss = %.4f, c_acc = %.4f, eval_acc = %.4f, max_eval_acc = %.4f',
                          epoch, c_loss, c_acc, eval_acc, max_acc)
        self.clas.load_state_dict(copy.deepcopy(best_clas))  # Reload the best classifier

    @staticmethod
    def eval_dis(model, data_loader, criterion):
        total_loss = 0
        total_acc = 0
        total_num = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()

                pred = model.forward(inp)
                loss = criterion(pred, target)
                total_loss += loss.item()
                total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
                total_num += inp.size(0)
            total_loss /= len(data_loader)
            total_acc /= total_num
        return total_loss, total_acc

    @staticmethod
    def optimize_multi(opts, losses):
        for i, (opt, loss) in enumerate(zip(opts, losses)):
            opt.zero_grad()
            loss.backward(retain_graph=True if i < len(opts) - 1 else False)
            opt.step()

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()

    def show_config(self):
        self.log.info(100 * '=')
        self.log.info('> training arguments:')
        for arg in vars(self.opt):
            self.log.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
        self.log.info(100 * '=')

    def cal_metrics(self, fmt_str=False, dictionary = None, eval_samples=None):
        """
        Calculate metrics
        :param fmt_str: if return format string for logging
        """
        # print(f"in the call mestric, the original dictionary size is {len(self.idx2word_dict)}")
        if dictionary == None:
            dictionary = self.idx2word_dict
            # print(f"in the call mestric, the dictionary size is {len(dictionary)}")
        else:
            # print("use the new dictionay in the cal_metrics computation")
            # print(f"in the call mestric, the dictionary size is {len(dictionary)}")
            pass

        with torch.no_grad():
            # Prepare data for evaluation
            # added by bing: get it from random setting or the variable transfer!
            if eval_samples == None:
                eval_samples = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size)
                # not sure why 200, 200 in this case
                gen_tokens_s = tensor_to_tokens(self.gen.sample(200, 200), dictionary)
            else:
                gen_tokens_s = tensor_to_tokens(eval_samples, dictionary)
            gen_data = GenDataIter(eval_samples)
            gen_tokens = tensor_to_tokens(eval_samples, dictionary)

            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=self.test_data.tokens)
            self.nll_gen.reset(self.gen, self.train_data.loader)
            self.nll_div.reset(self.gen, gen_data.loader)
            self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
            self.ppl.reset(gen_tokens)

        if fmt_str:
            return ', '.join(['%s = %s' % (metric.get_name(), metric.get_score()) for metric in self.all_metrics])
        else:
            return [metric.get_score() for metric in self.all_metrics]

    def cal_metrics_with_label(self, label_i):
        assert type(label_i) == int, 'missing label'

        with torch.no_grad():
            # Prepare data for evaluation
            eval_samples = self.gen.sample(cfg.samples_num, 8 * cfg.batch_size, label_i=label_i)
            gen_data = GenDataIter(eval_samples)
            gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
            gen_tokens_s = tensor_to_tokens(self.gen.sample(200, 200, label_i=label_i), self.idx2word_dict)
            clas_data = CatClasDataIter([eval_samples], label_i)

            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=self.test_data_list[label_i].tokens)
            self.nll_gen.reset(self.gen, self.train_data_list[label_i].loader, label_i)
            self.nll_div.reset(self.gen, gen_data.loader, label_i)
            self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
            self.clas_acc.reset(self.clas, clas_data.loader)
            self.ppl.reset(gen_tokens)

        return [metric.get_score() for metric in self.all_metrics]

    def comb_metrics(self, fmt_str=False):
        all_scores = [self.cal_metrics_with_label(label_i) for label_i in range(cfg.k_label)]
        all_scores = np.array(all_scores).T.tolist()  # each row for each metric

        if fmt_str:
            return ', '.join(['%s = %s' % (metric.get_name(), score)
                              for (metric, score) in zip(self.all_metrics, all_scores)])
        return all_scores

    def _save(self, phase, epoch, dictionary=None, prev_hiddens=None):
        """Save model state dict and generator's samples"""

        # TODO: why do we set no ADV? proposed by bing?
        if phase != 'ADV':
            torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phase, epoch))

        if dictionary == None:
            dictionary = self.idx2word_dict
        else:
            pass

        save_sample_path = cfg.save_samples_root + 'samples_{}_{:05d}.txt'.format(phase, epoch)
        if prev_hiddens == None:
            samples = self.gen.sample(cfg.batch_size, cfg.batch_size)
        else:
            samples = self.gen.sample(cfg.batch_size, cfg.batch_size,
                                                    hidden=prev_hiddens)  # batch_size*max_len_sent*vocab_size
        write_tokens(save_sample_path, tensor_to_tokens(samples, dictionary))

    def update_temperature(self, i, N):
        self.gen.temperature.data = torch.Tensor([get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)])
        if cfg.CUDA:
            self.gen.temperature.data = self.gen.temperature.data.cuda()
