import torch
import torch.nn.functional as F
import torch.optim as optim

import config as cfg
from instructor.real_data.relgan_instructor import RelGANInstructor
from models.RelGAN_D import RelGAN_D
from models.RelGAN_G import RelGAN_G
from utils.helpers import get_fixed_temperature, get_losses
from torch import nn
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import nltk
from utils.text_process import tokens_to_tensor
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, accuracy_score



from instructor.real_data.g_classifier import TwoLevelLstmClassifier
from instructor.real_data.g_classifier import TIES_V2
from instructor.real_data.g_classifier import CNN_Clf

from sklearn.model_selection import train_test_split
import numpy as np
import time
from utils.data_loader import GenDataIter
from torch.utils.data import DataLoader
from instructor.real_data.mmd_loss import MMD_loss
import random

from metrics.bleu import BLEU
from utils.text_process import tensor_to_tokens
from utils.text_process import write_tokens
import os
from scipy.spatial.distance import cosine


class RelGANInstructorRevised(RelGANInstructor):
    # vocab_size extend_vocab_size
    def __init__(self, opt, vocab_size=cfg.extend_vocab_size):
        # the previous varialbes and init_model are conducted
        super(RelGANInstructorRevised, self).__init__(opt, vocab_size)
        # ==== parameter logging ====
        self.log.info(f"we finally use device {cfg.device}")
        # ==== end ====

        self.gen_paired = RelGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.log.info(f"whether the used text generation model is stance-based attention-aware: {cfg.if_use_context_attention_aware}")

        # print(f"in the init {vocab_size}")
        # # exit(0)
        # # ==== part 0 ====: some data initiliazation
        # # ==== part 0.0 ===: dataset loading
        # self.sents2label = pickle.load(open('dataset/testdata/' + opt.dataset + '_used.pkl', "rb"))
        # self.sents2context = pickle.load(open('dataset/testdata/' + opt.dataset + '_context_used.pkl', "rb"))
        # self.sents2relevancy_score = pickle.load(open('dataset/testdata/' + opt.dataset + '_relevancy_used.pkl', "rb"))

        self.sents2label = pickle.load(open('dataset/testdata/' + opt.dataset + '.pkl', "rb"))
        self.sents2context = pickle.load(open('dataset/testdata/' + opt.dataset + '_context.pkl', "rb"))
        self.sents2relevancy_score = pickle.load(open('dataset/testdata/' + opt.dataset + '_relevancy.pkl', "rb"))

        # ==== part 0.0 ====: ending
        self.labels = []
        self.seqs_without_final_sent = []
        # todo: update it later, for function text_process with the same inputs and different output! modular setting
        self.seqs_one_long_each_seq = []
        self.final_sents = []

        self.contexts = []
        self.seqs_one_long_each_seq_with_contex = []
        self.relevancy_scores = []

        # for the bleu score computation!
        self.final_sents_tokens = []

        print(f"the required length of sequence in lstm level (maximal edits) is {cfg.max_len_seq_lstm-1}")
        print(f"whether we are in the testing mode: {cfg.if_test}")
        # print(" we store the dictionary")
        # pickle.dump(self.test_data.word2idx_dict, open("yelpEquLargeDict.pkl", "wb"))
        # exit(0)
        for index, (sents, label) in enumerate(self.sents2label.items()):
            # self.log.info(index)
            # adapted from def load_data(self, filename) in data_loader.py
            # self.log.info(sents, label)

            # ==== part 1 ====: do the tokenization for the sentence, padding in the tokens to tensor
            tokenlized = []
            tokenlized_one_long = []
            for sent in sents[:-1]:
                text = nltk.word_tokenize(sent.lower())
                tokenlized.append(text)
                tokenlized_one_long.extend(text)
            relevancy_score = self.sents2relevancy_score[sents] # list format
            if len(relevancy_score) == len(tokenlized):
                pass
            else:
                assert len(relevancy_score)-1 == len(tokenlized) # align the relevancy score with the posts
                relevancy_score = relevancy_score[:-1]
            # ==== padding ====: set the padding for lstm-classifier with same number of edits/posts
            # TODO: why -1? not quite sure, the historical n-1 posts
            if len(tokenlized) < (cfg.max_len_seq_lstm-1):
                tokenlized = [[cfg.padding_token] * cfg.max_seq_len] * (cfg.max_len_seq_lstm-1 - len(tokenlized)) + tokenlized
                # common mistake: you update tokenized, then, you use len(tokenlized!, be careful of this errors!)
                relevancy_score = [0] * (cfg.max_len_seq_lstm - 1 - len(relevancy_score)) + relevancy_score
            else:
                # this condition exists with too many edits!
                # todo: double check the wikipedia notebook, I should already set the max-num-edits;
                tokenlized = tokenlized[-(cfg.max_len_seq_lstm-1):]
                relevancy_score = relevancy_score[-(cfg.max_len_seq_lstm - 1):]
            # ==== end ====
            final_sent = nltk.word_tokenize(sents[-1].lower())
            self.final_sents_tokens.append(final_sent)
            if isinstance(self.sents2context[sents], list):
                # used in ankur's setting
                context = nltk.word_tokenize(self.sents2context[sents][-1].lower())
            else:
                context = nltk.word_tokenize(self.sents2context[sents].lower())
            # note, we include the global test dataset for the setting, we need it at this training
            # self.word2idx_dict is equal to self.train_data.word2idx_dict
            # self.log.info(len(tokenlized))
            tokenlized_tf = tokens_to_tensor(tokenlized, self.test_data.word2idx_dict).to(cfg.device)
            tokenlized_tf_one_long = tokens_to_tensor(tokenlized_one_long, self.test_data.word2idx_dict,
                                                      one_long=True).to(cfg.device)
            tokenlized_tf_final_sent = tokens_to_tensor(final_sent, self.test_data.word2idx_dict,
                                                      one_long=True,
                                                      one_long_if_pad = True).to(cfg.device)

            # print(context)
            tokenlized_tf_context = tokens_to_tensor(context, self.test_data.word2idx_dict,
                                                      one_long=True,
                                                      one_long_if_pad = True).to(cfg.device)

            tokenlized_tf_one_long_with_context = tokens_to_tensor(tokenlized_one_long+context, self.test_data.word2idx_dict,
                                                      one_long=True).to(cfg.device)

            # TODO: the one long should ahve the view(1, -1) setting
            self.seqs_without_final_sent.append(tokenlized_tf)
            self.seqs_one_long_each_seq.append(tokenlized_tf_one_long)
            self.final_sents.append(tokenlized_tf_final_sent.view(1, -1))
            self.labels.append(torch.Tensor([label]).long().to(cfg.device))
            self.relevancy_scores.append(torch.Tensor(relevancy_score).view(1, -1).to(cfg.device))

            self.contexts.append(tokenlized_tf_context.view(1, -1))
            self.seqs_one_long_each_seq_with_contex.append(tokenlized_tf_one_long_with_context)
        # print(max([len (i) for i in self.seqs_one_long_each_seq_with_contex])) # 567 for the setting
        self.log.info(f"the classifier label distribution is : {Counter(torch.cat(self.labels).cpu().numpy())}")
        self.seqs = [torch.cat((self.seqs_without_final_sent[i], self.final_sents[i]), dim=0) for i in range(len(self.seqs_without_final_sent))]
        # training & testing data split, like 80:20 split
        # todo: this should be in the adv_train_generator function for the related setting!

        # random state 0 for reproducibility, it is enough even with shuffle and stratify setting
        # this code is revised for the later cross-validation
        train_idx, test_idx = train_test_split(np.arange(len(self.labels)), test_size=0.2, random_state=0,
                                                shuffle=True, stratify=self.labels)

        # we can assume, seqs = seqs_without_final_sent + final_sents
        # zip for the quick packing and unpacking
        self.train_seqs, self.train_seqs_without_final_sent, self.train_final_sents, self.train_seqs_one_long_each_seq, \
            self.train_labels = zip(*([(self.seqs[idx], self.seqs_without_final_sent[idx], self.final_sents[idx], self.seqs_one_long_each_seq[idx],
            self.labels[idx]) for idx in train_idx]))
        self.test_seqs, self.test_seqs_without_final_sent, self.test_final_sents, self.test_seqs_one_long_each_seq, \
            self.test_labels = zip(*([(self.seqs[idx], self.seqs_without_final_sent[idx], self.final_sents[idx], self.seqs_one_long_each_seq[idx],
            self.labels[idx]) for idx in test_idx]))
        self.log.info(f"the classifier label distribution is : {Counter(torch.cat(self.train_labels).cpu().numpy()), Counter(torch.cat(self.test_labels).cpu().numpy())}")

        # ==== added 2020 dec. 20 ==== consider the context
        self.train_seqs_one_long_each_seq_with_contex = [ self.seqs_one_long_each_seq_with_contex[idx] for idx in train_idx]
        self.test_seqs_one_long_each_seq_with_contex = [ self.seqs_one_long_each_seq_with_contex[idx] for idx in test_idx]
        # ==== consider the relevancy score ====:
        self.train_relevancy_scores = [ self.relevancy_scores[idx] for idx in train_idx]
        self.test_relevancy_scores = [ self.relevancy_scores[idx] for idx in test_idx]

        self.train_contexts = [ self.contexts[idx] for idx in train_idx]
        self.test_contexts = [ self.contexts[idx] for idx in test_idx]

        # exit(0)
        # ==== shreeshaa's data transfer only ====
        test_seqs_all = []
        for one_seq in  self.test_seqs_without_final_sent:
            one_seq_tokens = tensor_to_tokens(one_seq, dictionary=self.test_data.idx2word_dict)
            test_seqs_all.append(one_seq_tokens)
        # ==== dump file for S's transfer only purpose ====
        # pickle.dump(test_seqs_all, open("./test_seqs.pkl", "wb"))
        # pickle.dump(self.test_labels, open("./test_labels.pkl", "wb"))
        # ==== end ====

        context_user_history, target_context = [], []
        for i in range(len(self.test_contexts)):
            one_target_context = self.test_contexts[i]
            one_context_user_history = self.test_seqs_one_long_each_seq_with_contex[i]

            # print(one_target_context)
            one_target_context = tensor_to_tokens(one_target_context, dictionary=self.test_data.idx2word_dict)
            # print(one_context_user_history)
            # one_context_user_history = tensor_to_tokens(one_context_user_history, dictionary=self.test_data.idx2word_dict)

            target_context.append(one_target_context)
            # context_user_history.append(one_context_user_history)
        # ==== dump for S's transfer only purpose ====
        # save the file
        # pickle.dump(context_user_history, open("./context_user_history.pkl", "wb"))
        # pickle.dump(target_context, open("./target_context.pkl", "wb"))
        # ==== end ====

        if cfg.if_have_clf:
            self.log.info(f"the training classifier is: cfg.clf_by_rnn: {cfg.clf_by_rnn}, cfg.clf_by_cnn: {cfg.clf_by_cnn}, cfg.clf_by_ties:{cfg.clf_by_ties}")
            self.classifier_init()

        #######################
        # build the global ldas as the class field
        contexts_token_list = torch.cat(self.contexts, dim=0).tolist()
        contexts_matrix = self.word_token_list2feature_vector(contexts_token_list, len(self.test_data.word2idx_dict))
        self.ldas = []
        for num_clusters in range(2, 20, 5):
            lda = LatentDirichletAllocation(n_components=num_clusters)
            lda.fit(contexts_matrix)
            self.ldas.append(lda)
        #######################
        # ==== end ====



        if cfg.if_pretrain_mle:
            self.log.info(f"the mle optimizer is: gen_lr:{cfg.gen_lr}")

        if cfg.if_adv_training:
            # optimizer logs
            self.log.info(f"the optimizer setting is: cfg.attack_lr: {cfg.attack_lr}, cfg.relevancy_lr: {cfg.relevancy_lr}, cfg.recency_lr: {cfg.recency_lr}, recent posts: {cfg.num_recent_posts}")
            self.log.info(f"the other setting: cfg.if_pretrain_mle_in_adv: {cfg.if_pretrain_mle_in_adv}")
            # optimizers
            self.attack_opt = optim.Adam(self.gen.parameters(), lr=cfg.attack_lr)
            self.relevancy_opt = optim.Adam(self.gen.parameters(), lr=cfg.relevancy_lr)
            self.recency_opt = optim.Adam(self.gen.parameters(), lr=cfg.recency_lr)

        self.log.info(f"the size of the dictionary is {len(self.test_data.word2idx_dict)}")

    def classifier_init(self):

        # ==== part 1 ====: for the classification module
        EMBEDDING_DIM = 256 * 2  # # LSTM_LAYERS = 1
        num_label = 2
        SEQ_HIDDEN_DIM = 128 * 2
        SEQS_HIDDEN_DIM = 64 * 2

        if cfg.clf_by_rnn:
            self.clf_seq = TwoLevelLstmClassifier(EMBEDDING_DIM, SEQ_HIDDEN_DIM, SEQS_HIDDEN_DIM,
                                           len(self.test_data.word2idx_dict), num_label,
                                            max_len_seq_lstm=cfg.max_len_seq_lstm, max_len_sent=cfg.max_seq_len).to(cfg.device)
        elif cfg.clf_by_cnn:
            self.clf_seq = CNN_Clf(embed_dim = EMBEDDING_DIM, vocab_size=len(self.test_data.word2idx_dict), num_classes=num_label,
                 max_len_seq_lstm=cfg.max_len_seq_lstm, max_len_sent=cfg.max_seq_len).to(cfg.device)
        elif cfg.clf_by_ties:
            # hrere, the batch_size is the number
            # self.clf_seq = TIES(batch_size=cfg.max_len_seq_lstm, output_size=num_label, hidden_size=SEQ_HIDDEN_DIM, vocab_size=len(self.test_data.word2idx_dict), embedding_length=EMBEDDING_DIM).to(cfg.device)
            self.clf_seq = TIES_V2(embedding_dim=EMBEDDING_DIM, seq_hidden_dim=SEQ_HIDDEN_DIM, vocab_size=len(self.test_data.word2idx_dict), label_size=num_label).to(cfg.device)

        # ==== used in the black-box testing ====
        if cfg.if_black_box_test_rnn:
            self.clf_seq_another = TwoLevelLstmClassifier(EMBEDDING_DIM, SEQ_HIDDEN_DIM, SEQS_HIDDEN_DIM,
                                           len(self.test_data.word2idx_dict), num_label,
                                            max_len_seq_lstm=cfg.max_len_seq_lstm, max_len_sent=cfg.max_seq_len).to(cfg.device)
            self.clf_seq_another.load_state_dict(torch.load(cfg.pretrained_rnn_clf_another, map_location=f"cuda:{cfg.device}"))
            self.log.info(f"the path of the clf_seq_another is RNN in black-box attack: {cfg.pretrained_rnn_clf_another}")
        elif cfg.if_black_box_test_ties:
            self.clf_seq_another = TIES_V2(embedding_dim=EMBEDDING_DIM, seq_hidden_dim=SEQ_HIDDEN_DIM,
                                   vocab_size=len(self.test_data.word2idx_dict), label_size=num_label).to(cfg.device)
            self.clf_seq_another.load_state_dict(
                torch.load(cfg.pretrained_ties_clf, map_location=f"cuda:{cfg.device}"))
            self.log.info(f"the path of the clf_seq_another is TIES in black-box attack: {cfg.pretrained_ties_clf}")
        # ========================================


        if cfg.if_unbalance:
            self.log.info(f"the class weight for the unbalanced data distribution is {cfg.weights}")
            class_weight = torch.Tensor(cfg.weights).to(cfg.device)
            self.clf_loss = nn.CrossEntropyLoss(weight=class_weight).to(cfg.device)
        else:
            self.clf_loss = nn.CrossEntropyLoss().to(cfg.device)



        if cfg.use_saved_clf:
            self.log.info(f"use_saved_clf in directory: {cfg.pretrained_clf_path}")
            self.clf_seq.load_state_dict(torch.load(cfg.pretrained_clf_path, map_location=f"cuda:{cfg.device}")) # torch.load(cfg.pretrained_clf_path)
        elif cfg.device != torch.device("cpu"):
            # : add cfg.device is not torch.device("cpu") in the later time and avoid long time training
            self.log.info("Train clf")
            self.log.info(f"the learning rate of clf is: {cfg.clf_lr}")
            # train the clf and save it!
            self.clf_optim = optim.SGD(self.clf_seq.parameters(), lr=cfg.clf_lr)
            self.log.info("**** start clf training ****")
            self.train_classifier_lstm(cfg.clf_epoches)
        self.log.info("**** deployed classifier performance ****")
        self.test_classifier_lstm(self.train_seqs, self.train_labels)
        self.test_classifier_lstm()

        # freeze the clf parameters:
        # by checking the GAN module: the similar training, the author does not set the requires grad(image, we only train gen)
        # search pytorch free gradient for some layers
        # check two posts: https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088
        # https://discuss.pytorch.org/t/freezing-intermediate-layers-while-training-top-and-bottom-layers/39776
        # for param in self.clf_seq.parameters():
        #     param.requires_grad = False
        #     # input -> clf_seq-> loss
        #     # add by , error in the attack mode, cudnn can only be in the training mode
        #     # self.clf_seq.train() # by default, we set it as train and ignore the possible dropout layer now!

    def train_classifier_lstm(self, epoches):
        avg_train_losses = []
        for epoch in tqdm(range(epoches)):
            # TODO: we use the batch size of 1 rather than cfg.batch_size
            # para: batch_size: cfg.batch_size vs 1
            avg_train_loss = self.clf_seq.train_classifier_lstm(self.clf_seq,
                                self.train_seqs, self.train_labels, cfg.batch_size,
                                  self.clf_loss, self.clf_optim,
                                  vocab_size=len(self.test_data.word2idx_dict))
            self.log.info(f"after epoch {epoch}, the avg train loss is: {avg_train_loss}")
            # #### tqdm monitoring ####
            if epoch % 10 == 0 or epoch == (epoches-1):
                if cfg.if_sav_clf:
                    self.save_model("CLF", epoch)
                self.test_classifier_lstm()
                self.log.info("**** test on the baseline to see the difference ****")
            # #### tqdm monitoring ####
            avg_train_losses.append(avg_train_loss)

        return avg_train_losses

    def test_classifier_lstm(self, tested_seqs=None, tested_labels=None):
        if tested_seqs is None and tested_labels is None:
            tested_seqs, tested_labels = self.test_seqs, self.test_labels
            self.log.info("test the classifier by the test data")
        else:
            self.log.info("test the clf by the assigned data")
        avg_test_loss, y_pred_probs, y_tests = self.clf_seq.test_classifier_lstm(self.clf_seq,
                                tested_seqs, tested_labels, cfg.batch_size, self.clf_loss,
                                vocab_size=len(self.test_data.word2idx_dict))
        # print(f"the result is {y_pred_probs, y_tests}")
        results, _, _ = self.clf_seq.compute_performance(y_pred_probs,y_tests)
        self.log.info(f"the avg loss is {avg_test_loss} and performance is {results}")
        return avg_test_loss, results

    # override
    def _run(self):
        # ===PRE-TRAINING (GENERATOR)===
        if cfg.if_pretrain_mle:
            self.log.info('Starting Generator MLE Training...')
            start_time = time.time()
            if cfg.if_previous_pretrain_model:
                self.log.info("use the preivous MLE module")
                self.pretrain_generator(cfg.MLE_train_epoch)
            else:
                self.log.info("use the posterior MLE module")
                for epoch in range(cfg.MLE_train_epoch):
                    mle_loss = self.general_adv_train_both(step_number=1, mode="mle")
                    self.log.info(f"the MLE loss in epoch {epoch} is {mle_loss}")
                    if epoch % cfg.pre_log_step == 0 or epoch == cfg.MLE_train_epoch - 1:
                        with torch.no_grad():
                            self.test_generator_performance()
                        if cfg.if_sav_pretrain:
                            self.save_model("MLE", epoch)

            # ####  #### : comment this: and not cfg.if_test: since we will test on the situation
            # ==== comment it at forst on Jan. 18th, due to the override problem ====
            # if cfg.if_sav_pretrain:
            #     torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
            #     self.log.info('Save pretrain_generator: {}'.format(cfg.pretrained_gen_path))
            self.log.info(f"The whole MLE took: {time.time() - start_time}")
            self.log.info('Ending Generator MLE Training...')

        if cfg.if_adv_training:
            progress = tqdm(range(cfg.ADV_train_epoch))
            for adv_epoch in progress:
                start_time = time.time()
                self.sig.update()
                if self.sig.adv_sig:

                    if cfg.if_adv_gan:
                        self.log.info('Starting GAN Training...')
                        g_loss = self.general_adv_train_both(step_number=cfg.ADV_g_step, mode="gen")
                        d_loss = self.general_adv_train_both(step_number=cfg.ADV_d_step, mode="dis") # d_loss = 0
                        self.log.info(f"the GAN loss is {g_loss, d_loss}")

                    if cfg.if_adv_attack:
                        self.log.info("start attack training")
                        attack_loss = self.general_adv_train_both(step_number=cfg.ADV_g_step, mode="attack")
                        self.log.info(f"the attack loss is {attack_loss}")

                    if cfg.if_adv_recency:
                        self.log.info("start recency training")
                        recency_loss = self.general_adv_train_both(step_number=cfg.ADV_g_step, mode="recency")
                        self.log.info(f"the recency loss is {recency_loss}")

                    if cfg.if_adv_relevancy:
                        self.log.info("start relevancy training")
                        relevancy_loss = self.general_adv_train_both(step_number=cfg.ADV_g_step, mode="relevancy")
                        self.log.info(f"the relevancy loss is {relevancy_loss}")

                    if cfg.if_adv_stance:
                        self.log.info("start stance training")
                        relevancy_loss = self.general_adv_train_both(step_number=cfg.ADV_g_step, mode="stance")
                        self.log.info(f"the stance loss is {relevancy_loss}")

                    if cfg.if_pretrain_mle_in_adv:
                        self.log.info("start mle training in adv")
                        mle_loss = self.general_adv_train_both(step_number=1, mode="mle")
                        self.log.info(f"the stance loss is {mle_loss}")

                    self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature

                    # performance on the testing data
                    if cfg.if_have_clf and adv_epoch % cfg.adv_log_step == 0:
                        with torch.no_grad():
                            self.test_generator_performance()
                    # SAVE
                    if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                        if cfg.if_sav_adv:
                            self.save_model("ADV", adv_epoch)
                    # progress.set_description(
                    #     'g_loss: %.4f, d_loss: %.4f, temperature: %.4f' % (g_loss, d_loss, self.gen.temperature))
                    # comment first by  and check the previous setting
                    # TEST
                    # if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                    #     self.log.info('[ADV] epoch %d: g_loss: %.4f, d_loss: %.4f, %s' % (
                    #         adv_epoch, g_loss, d_loss, self.cal_metrics(fmt_str=True,
                    #                                                     dictionary = self.test_data.idx2word_dict)))
                        # self.log.info(": show if_test".format(cfg.if_test))
                        # : TODO: to change, and save text from the specific conditional text
                        # if cfg.if_save and not cfg.if_test:
                        #     self._save('ADV', adv_epoch, dictionary = self.test_data.idx2word_dict)

                else:
                    self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
                    progress.close()
                    break
                self.log.info(f"epoch {adv_epoch} took: {time.time() - start_time}")
            self.log.info("the whole training in the run mode is done")


    def save_model(self, mode, epoch):
        import os
        if not os.path.exists(cfg.save_model_root):
            os.makedirs(cfg.save_model_root)
            #TODO: why this happens?? in the past, I do not encouter the non-existance problem!!
            self.log.info(f"the directory does not exist {cfg.save_model_root}")
        if mode == "CLF":
            save_model_path = cfg.save_model_root + f"{mode}_model{cfg.clf_model_name}_epoch{epoch}.pt"
            torch.save(self.clf_seq.state_dict(), save_model_path)
        else:
            # MLE, ADV
            save_model_path = cfg.save_model_root + f"{mode}_model{cfg.model_name}_epoch{epoch}.pt"
            torch.save(self.gen.state_dict(), save_model_path) #  'gen_{}_{:05d}.pt'.format(phase, epoch)
        self.log.info(f"the saved model path is: {save_model_path}")

    # override by adding the dictionary
    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        # ===Train===
        # ==== rebuild the context-aware training data ====
        # data loader, inp, tgt setting
        # ==== Note ====: we cannot do the batch for variant length of sequences
        # todo: the difference length of the contexts cannot be loaded into a tensor by torch.cat()
        one_long_with_context = self.train_seqs_one_long_each_seq_with_contex

        # one_long_with_context = one_long_with_context[:cfg.batch_size] # [one_long_with_context[0]]
        # print(one_long_with_context)

        input, target = GenDataIter.prepare(torch.cat(self.train_final_sents, dim=0))
        # input, target = input[:cfg.batch_size], target[:cfg.batch_size]
        # print(input, target)
        # prev: self.batch_size
        if cfg.if_linear_embedding:
            # added during zero mle loss debugging, we first recall the previous method
            train_paired_data = [{'one_long_context': F.one_hot(i, cfg.extend_vocab_size).float(),
                                  'input': F.one_hot(j, cfg.extend_vocab_size).float(), 'target': k} for (i, j, k) in
                                 zip(one_long_with_context, input, target)]
        else:
            train_paired_data = [{'one_long_context': i, 'input': j, 'target': k} for (i, j, k) in
                                 zip(one_long_with_context, input, target)]
        # print(train_paired_data[0]['input'].shape)

        for epoch in range(epochs):
            start_time = time.time()
            self.sig.update()
            if self.sig.pre_sig:
                # random.shuffle(train_paired_data) # commet it for the debug checking!

                # train_paired_data = torch.load("./test_mle.pt", map_location=torch.device(cfg.device))

                pre_loss = self.train_gen_epoch(self.gen, train_paired_data, self.mle_criterion, self.gen_opt,
                                                compute_hidden=True)
                print(f"the pretraining loss in epoch {epoch} the previous module is {pre_loss}")
                # torch.save(train_paired_data, "./test_mle_prev.pt")
                # print("test the tensor from posterior mle")
                # exit(0)
                # # ===Tes            # if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                #     self.log.info('[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (
                #         epoch, pre_loss, self.cal_metrics(fmt_str=True, dictionary = self.test_data.idx2word_dict)))
                #     # comment by  and no save now
                #     # if cfg.if_save and not cfg.if_test:
                #     #     self._save('MLE', epoch, dictionary = self.test_data.idx2word_dict)
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break

            self.log.info(f"one epoch took: {time.time() - start_time}")



    def general_adv_train_both(self, step_number, mode:str, stage="train", clf_for_pred=None):
        """
        seq_fmt_flag=None
        :param step_number: or epoches for each gen, dis, attack training
        :param mode: "gen", "dis":return:
        """
        if stage == "train":
            # w/o context
            # seqs_fmt_selected = self.train_seqs_one_long_each_seq
            # w context
            seqs_fmt_selected = self.train_seqs_one_long_each_seq_with_contex
            paired_seqs_fmt_selected_prev_without_final_sent = self.train_seqs_without_final_sent
            paired_seqs_fmt_selected_final = self.train_final_sents
            matched_labels = self.train_labels
            matched_relevancyscores = self.train_relevancy_scores
            matched_context = self.train_contexts

        else:
            assert stage == "test"
            # w/o context
            # seqs_fmt_selected = self.test_seqs_one_long_each_seq
            # w context
            seqs_fmt_selected = self.test_seqs_one_long_each_seq_with_contex
            paired_seqs_fmt_selected_prev_without_final_sent = self.test_seqs_without_final_sent
            paired_seqs_fmt_selected_final = self.test_final_sents
            matched_labels = self.test_labels
            matched_relevancyscores = self.test_relevancy_scores
            matched_context = self.test_contexts

        total_loss = 0
        # different metrics setting
        y_pred_probs_all = []
        y_tests_all = []
        y_generated_texts_in_token = []

        # step_number = epoch_number in the understanding!
        for step in range(step_number):
            start_time = time.time()
            # todo: udpate it later, here, we only use_one_long

            # input: hidden-state
            # output: the generate token for different matrics
            # cfg.batch_size = 1 # for initial testing and debugging
            for num_batch, index in enumerate(range(0, len(seqs_fmt_selected), cfg.batch_size)):
                if (index + cfg.batch_size) <= len(seqs_fmt_selected):
                    # ==== approach 0 ==== : do the parallel embedding like model(input, hidden_state)
                    # refer to forward in generator.py
                    # but, hard to transfer multiple edits -> one hidden, (similar to 2-level lstm classification)
                    # reffered function x = self.train_gen_epoch(self.gen, self.train_data.loader, self.mle_criterion, self.gen_opt)
                    # model = self.gen
                    # batch_size = seq.shape[0]
                    # hidden = model.init_hidden(batch_size)
                    # _, hidden = model.forward(seq, hidden, need_hidden=True) # hiden.shape: batch_size*hidden-dim
                    # # next hidden to one hidden
                    # ==== approach 1 ==== : do the sequential embedding with the whole sentence
                    # padding exists, it will affect the general performance due to the zero setting
                    # seq_in_one_series = torch.cat(seq, dim=1)
                    # possible solution 0: we change the tokens_to_tensor with additional flags: if pad and length setting
                    # possible solution 1: in the above-tokenlized.append(text), we ues extend to have the whole things
                    # h_n of shape (num_layers * num_directions, batch, hidden_size)
                    # todo: () will generate hidden as the generator, --- yes, tuple init is a little different!
                    # avoid the previous inner state update, hidden is a tuple and cannot be detached directly
                    seqs = seqs_fmt_selected[index: index+cfg.batch_size]

                    # : end
                    real_samples = paired_seqs_fmt_selected_final[index: index + cfg.batch_size]
                    real_samples = torch.cat(real_samples, dim=0)  # batch_size*max_len_sent

                    prev_train_seqs = paired_seqs_fmt_selected_prev_without_final_sent[
                                      index: index + cfg.batch_size]  # bathch-length list of (max_edit-1)*max_len_sent
                    matched_context_one_batch = matched_context[index: index + cfg.batch_size]
                    match_relevancyscores_one_batch = torch.cat(matched_relevancyscores[index: index + cfg.batch_size],
                                                                dim=0)  # batch_size*(max_len_seq_lstm-1)
                    # : add the vocab_size dimention to be compatible with linear embedding
                    if cfg.if_linear_embedding:
                        seqs = [F.one_hot(one_seq, cfg.extend_vocab_size).float() for one_seq in seqs] # [54,vocab_size]-<[54]
                        prev_train_seqs = [F.one_hot(one_seq, cfg.extend_vocab_size).float() for one_seq in prev_train_seqs]
                        matched_context_one_batch = [F.one_hot(one_seq, cfg.extend_vocab_size).float() for one_seq in matched_context_one_batch]
                    # #### end ####

                    if mode != "mle" and not cfg.if_use_context_attention_aware:
                        # #### even if it is not used in MLE, for simplicity, we uncomment it ####
                        prev_hiddens = self.compute_hiddens(self.gen, seqs,
                                                            if_list2tensor_quick_computation=True)  # False True
                        # copy in case: https://stackoverflow.com/questions/55266154
                        gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True,
                                                  hidden=prev_hiddens.clone().detach())  # batch_size*max_len_sent*vocab_size by one_hot
                        gen_samples_in_tokens = self.gen.sample(cfg.batch_size, cfg.batch_size,
                                                            hidden=prev_hiddens.clone().detach())  # batch_size*max_len_sent*vocab_size by one_hot
                    elif mode != "mle" and cfg.if_use_context_attention_aware:
                        weighted_hiddens = self.weighted_context_computation(self.gen, prev_train_seqs, matched_context_one_batch, match_relevancyscores_one_batch)
                        gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True,
                                        hidden=weighted_hiddens.clone().detach(),
                                        attention_context = weighted_hiddens.clone().detach()) # weighted_hidden: batch_size*1*hidden
                        gen_samples_in_tokens = self.gen.sample(cfg.batch_size, cfg.batch_size,
                                                            hidden=weighted_hiddens.clone().detach(),
                                                                attention_context = weighted_hiddens.clone().detach())
                    # ==== loss ====
                    # train gen by mle
                    if mode == "mle":
                        self.sig.update()
                        if self.sig.pre_sig:
                            # data loader, inp, tgt setting
                            # ==== Note ====: we cannot do the batch for variant length of sequences
                            # todo: the difference length of the contexts cannot be loaded into a tensor by torch.cat()
                            input, target = GenDataIter.prepare(real_samples) # batch_size*max_len_sent
                            # input, target = input[0], target[0]
                            # print(input, target)
                            # exit(0)

                            if cfg.if_linear_embedding:
                                # one_long_context = [F.one_hot(one_context, cfg.extend_vocab_size).float() for one_context in one_long_context]
                                input = F.one_hot(input, cfg.extend_vocab_size).float()

                            # prev: self.batch_size
                            train_paired_data = [{'one_long_context': i, 'input': j, 'target': k,
                                                  "prev_train_seq":m, "matched_context":n, "match_relevancy_score":l} for (i, j, k, m, n, l) in
                                                 zip(seqs, input, target,
                                                     prev_train_seqs, matched_context_one_batch, match_relevancyscores_one_batch)]
                            # torch.save(train_paired_data, "./test_mle.pt")
                            # exit(0)
                            # print(train_paired_data[0]['input'].shape)
                            # random.shuffle(train_paired_data)
                            # train_paired_data = torch.load("./test_mle.pt", map_location=torch.device(cfg.device))
                            # train_paired_data = torch.load("./test_mle_prev.pt", map_location=torch.device(cfg.device))

                            mle_loss = self.train_gen_epoch(self.gen, train_paired_data, self.mle_criterion, self.gen_opt,
                                                            compute_hidden=True,
                                                            if_list2tensor_quick_computation=True,
                                                            if_relevancy_attention_aware_context=cfg.if_use_context_attention_aware)
                            # print(f"mle_loss in the mle inner part (later universal part) is {mle_loss}")
                            # exit(0)
                            # TODO: check it in later time!
                            # why we can use w/o item()
                            total_loss += mle_loss
                    # ==== loss =====
                    # train gen and dis
                    elif mode == "gen" or mode == "dis":
                        # self.log.info(gen_samples.requires_grad) # it is true
                        #  checking, it should use cfg.extend_vocab_size, since
                        # previous it is cfg.vocab_size
                        real_samples = F.one_hot(real_samples, cfg.extend_vocab_size).float() # batch_size*max_len_sent*vocab_size
                        # ===Train===
                        d_out_fake = self.dis(gen_samples) # correct at the pass
                        d_out_real = self.dis(real_samples) # wrong at first
                        g_loss, d_loss = get_losses(d_out_real, d_out_fake, cfg.loss_type)

                        if mode == "gen":
                            self.optimize(self.gen_adv_opt, g_loss, self.gen)
                            total_loss += g_loss.item()
                        else:
                            assert mode == "dis"
                            self.optimize(self.dis_opt, d_loss, self.dis)
                            total_loss += d_loss.item()
                    # ==== loss ====
                    # train text generators
                    elif mode == "attack":
                        # previously, I use set the no grad argument in the sample, check it in the later time, if possible
                        # gen_train_seqs shape: list with batch_size*seqs(i.e., tensor: (max_len_seq)*max_len_sent)
                        # preivous type setting: prev_train_seqs:long; gen_final_sents[i]: float
                        # self.log.info(gen_final_sents[0].shape) # 30*3779
                        # self.log.info((F.one_hot(prev_train_seqs[0], len(self.test_data.word2idx_dict)).float().shape)) # 19*30*3779
                        # NOTE: if we use the batch_size*max_len_seq format, we may not be able to do the back-propagation,
                        if stage == "train":
                            # in the training mode, we have to use the token probability, thus, we do not need onehot
                            # prev shape: (max_num_edit-1)*seq_len; gen_samples: seq_len*vocab_size
                            gen_train_seqs = [torch.cat((prev_train_seqs[i],
                                                     gen_samples[i].unsqueeze(0)), dim=0) for i in range(cfg.batch_size)]
                        else:
                            # in testing, we just use the token to do the result generation, thus, we use onehot to the vo
                            gen_train_seqs = [torch.cat((prev_train_seqs[i],
                                                     F.one_hot(gen_samples_in_tokens[i].unsqueeze(0), cfg.extend_vocab_size).float()), dim=0) for i in range(cfg.batch_size)]

                        match_labels_one_batch = matched_labels[index: index + cfg.batch_size]
                        if clf_for_pred == None:
                            y_pred_probs = [self.clf_seq(one_seq) for one_seq in gen_train_seqs]
                        else:
                            y_pred_probs = [clf_for_pred(one_seq) for one_seq in gen_train_seqs]

                        y_pred_probs = torch.cat(y_pred_probs, dim=0) # to be tensor
                        if stage == "train":
                            # referenced: https://stackoverflow.com/questions/4260280/if-else-in-a-list-comprehension
                            flipped_labels = [label-1 if label.item() > 0 else label+1 for label in match_labels_one_batch]
                            flipped_labels = torch.cat(flipped_labels) # to be tensor, and the flipping result is tested

                            # loss function building
                            # be careful of the parameter update: the parameters for clf should not change
                            # #### approach 0 ####: by the weight freezee
                            # #### approach 1 ####: by the generated value detach, but, it fails
                            attack_loss_criteria = nn.CrossEntropyLoss().to(cfg.device)
                            # _, y_pred_probs, y_tests = self._test_classifier_lstm(clf_seq, gen_train_seqs, flipped_labels, cfg.batch_size, flipped_criteria_loss)
                            attack_loss = attack_loss_criteria(y_pred_probs, flipped_labels)

                            # print(f"the computated avg loss is: {attack_loss}")
                            # attack_loss = max(random.randint(-7,+7)/10. + attack_loss, 0)
                            # print(f"the computated avg loss with random additon is: {attack_loss}")

                            # todo: we can set another optimizer in the attack mode
                            self.optimize(self.attack_opt, attack_loss, self.gen)

                            total_loss += attack_loss.item()
                        else:
                            # #### testing mode ####:
                            y_pred_probs_all.append(y_pred_probs.detach().cpu())
                            y_tests_all.extend([each_label.detach().cpu() for each_label in match_labels_one_batch])
                            y_generated_texts_in_token.append(gen_samples_in_tokens)
                            # #### testing mode ####:

                    # ==== loss ====
                    # train text generator
                    elif mode == "relevancy" or mode == "recency" or mode == "stance":

                        # ==== approach 1 ====: batch_size*max_len_seq*vocabulary setting
                        # ==== approach 0 ====: we should use the batch_size*max_len_seq for back propagation by hidden state
                        # print(prev_train_seqs[0].requires_grad)
                        # print(prev_train_seqs[0].shape, gen_samples[0].unsqueeze(0).shape) # torch.Size([19, 31]) torch.Size([1, 31])
                        # ==== approach 2 ====: we directly use the hidden state from the gen.sample()

                        # ==== start loss computation ====:
                        batch_level_loss = 0
                        # ==== mmd by hidden ====: avoid OOM error
                        self.gen_paired.load_state_dict(self.gen.state_dict())
                        # approach 0: freeze the gradient or
                        # approach 1: we do not optimize that model
                        # for param in self.gen_paired.parameters():
                        #     param.requires_grad = False
                        self.gen_paired.to(cfg.device) # TODO: check we need to .to(cfg.device), to my understanding, I have set it!
                        # ==== end ====
                        # for new computation usage only in the relevancy


                        stance_loss_save = [[], []]
                        mmd_loss_criteria = MMD_loss()
                        for i in range(cfg.batch_size):
                            one_seq_loss = 0
                            # ==== loss 1 ====: by the original text distributio
                            # error of moved leafy varaibles: https://stackoverflow.com/questions/59696406
                            matched_prev_train_seqs = prev_train_seqs[i] # (max_edit-1)*max_len_seq -> # (max_edit-1)*max_len_seq*vocab_size
                            # matched_prev_train_seqs = F.one_hot(matched_prev_train_seqs, cfg.extend_vocab_size).float()

                            # ==== loss direction 0 ====: iterate edit one by one
                            matched_gen_samples = gen_samples[i] # max_len_seq*vocab_size;
                            # ==== approach 1 ====: for batch-level computation
                            # all_hiddens: num_edit*1*hidden_state

                            # all_hiddens = self.compute_hiddens(self.gen_paired, torch.cat([matched_prev_train_seqs, matched_gen_samples.unsqueeze(0)], dim=0))
                            # print(all_hiddens.requires_grad)
                            # all_hiddens = all_hiddens.squeeze(1)
                            # print(all_hiddens.requires_grad)
                            # matched_gen_samples_transferred, matched_prev_train_seqs_one_edit = all_hiddens[-1].unsqueeze(0), all_hiddens[0:-1, :]
                            # print(matched_gen_samples_transferred.requires_grad)
                            if mode == "relevancy":
                                matched_weight = match_relevancyscores_one_batch[i] # shape: [max_len_seq_lstm-1]
                                # previously, we judge the weight by is not, then, the value may be different, we should use >. a better solution!
                                # print(matched_weight)
                                matched_prev_train_seqs = [ matched_prev_train_seqs[j] for j in range(len(matched_weight)) if matched_weight[j] > 0 ]
                            elif mode == "recency":
                                matched_prev_train_seqs = matched_prev_train_seqs[-cfg.num_recent_posts:] # num_recent_posts*max_len_seq*vocab_size
                            elif mode == "stance":
                                # matched_context_one_batch: one context tensor
                                # TODO: currently, for stance, we just use one context for the comparison, updated it in the later time.
                                matched_prev_train_seqs = [F.one_hot(matched_context_one_batch[i], cfg.extend_vocab_size).float()] # 1*max_seq_len*vocab_size


                            if len(matched_prev_train_seqs) == 0:
                                # #### special conditions when we do not have any weight in the relevancy
                                batch_level_loss += 0
                            else:
                                # when running the relevancy, it can be very slow, thus, we compute it with parallel setting
                                # TODO: add the cocurrent setting
                                # here the hidden state computation should not depend on the hidden state and context
                                matched_prev_train_seqs_one_edit = self.compute_hiddens(self.gen_paired, matched_prev_train_seqs, if_list2tensor_quick_computation=True).squeeze(1) # num_edit-1*1*hidden
                                matched_gen_samples_transferred = self.compute_hiddens(self.gen_paired, matched_gen_samples.unsqueeze(0), if_detach=False).squeeze(1) # 1*hidden

                                matched_gen_samples_transferred = torch.cat([matched_gen_samples_transferred]*matched_prev_train_seqs_one_edit.shape[0], dim=0)
                                # print(matched_gen_samples_transferred.shape, matched_gen_samples_transferred.requires_grad)

                                if mode == "stance":
                                    stance_loss_save[0].append(matched_gen_samples_transferred), stance_loss_save[1].append(matched_prev_train_seqs_one_edit)
                                    continue

                                current_loss = mmd_loss_criteria(matched_gen_samples_transferred, matched_prev_train_seqs_one_edit)
                                one_seq_loss = current_loss
                                # print(f"one example loss in one batch loss is {one_seq_loss}")
                                # # ==== approach 0 ====: loss one edit by one edit
                                # for j in range(matched_prev_train_seqs.shape[0]):
                                #     matched_prev_train_seqs_one_edit = matched_prev_train_seqs[j, :] # max_len_seq*vocab_size
                                #     # TODO: we add weight at this part! w=[expoential decay setting!]
                                #     # large testing: 31, 26685
                                #     # print(matched_gen_samples.shape, matched_prev_train_seqs_one_edit.shape) # 31*4068(vocab_size,4k pass)
                                #     # ==== mode loss by incremental update ====
                                #
                                #     matched_gen_samples_transferred, matched_prev_train_seqs_one_edit = self.compute_hiddens(self.gen_paired, [matched_gen_samples, matched_prev_train_seqs_one_edit], if_detach=False)
                                #     # matched_gen_samples_transferred dimension: 1*512, maybe, we can use the batch computation! check it later
                                #
                                #     current_loss = mmd_loss_criteria(matched_gen_samples_transferred, matched_prev_train_seqs_one_edit)
                                #
                                #     if mode == "recency":
                                #         base_weight = 1 / 2
                                #         one_seq_loss = one_seq_loss*base_weight + (1-base_weight)* current_loss
                                #     else:
                                #         match_relevancyscores_one_batch = matched_relevancyscores[index: index + cfg.batch_size]
                                #         matched_weight = torch.cat(match_relevancyscores_one_batch,
                                #                                    dim=0)[i].view(1, -1)  # batch_size*(max_len_seq_lstm-1) # 8*19
                                #         # computation error from the zero value
                                #         if matched_weight[:, j] == 0:
                                #             continue
                                #         if j == 0:
                                #             one_seq_loss = current_loss
                                #         else:
                                #             one_seq_loss = (one_seq_loss*(torch.sum(matched_weight[:, :j])) + current_loss*matched_weight[:, j])/torch.sum(matched_weight[:, :j+1])

                                batch_level_loss += one_seq_loss

                        if mode == "stance":
                            batch_level_loss = mmd_loss_criteria(torch.cat(stance_loss_save[0], dim=0), torch.cat(stance_loss_save[1], dim=0))
                        else:
                            batch_level_loss /= cfg.batch_size  # one_seq_loss/matched_prev_train_seqs.shape[0]
                        # ==== loss direction 1 ====: compute the matrx-level loss function
                        # matched_gen_samples = gen_samples[i].unsqueeze(0) # 1*max_len_seq*vocab_size
                        # matched_gen_samples = torch.cat([matched_gen_samples] * matched_prev_train_seqs.shape[0], dim=0) # (max_edit-1)*max_len_seq*vocab_size

                        # RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed.
                        # https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795/4
                        # : my thinking, we only use one matched_gen_samples for one time and it is deleted in the bp setting
                        # print("start optimization")
                        # TODO: we should have different optimzers in the experiment, be careful of the checking!
                        self.optimize(self.relevancy_opt, batch_level_loss, self.gen) # , retain_graph=True
                        # print("end optimization")
                        total_loss += batch_level_loss.item()
                        # ==== loss 0 ====: by the hidden state
                        # gen_train_seqs = torch.cat((prev_train_seqs[i], gen_samples[i].unsqueeze(0)), dim=0)
                        # hidden = self.gen.init_hidden(gen_train_seqs.shape[0])
                        # # previous error: RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long
                        # _, hidden = self.gen.forward(gen_train_seqs, hidden, need_hidden=True)

                    # break  # only use one epoch, thus, we should break
            self.log.info(f" mode:{mode}; step:{step}; stage:{stage} took: {time.time()-start_time}")
        if stage == "train":
            # when step_number is greater than 1, then, we have:  return total_loss/((num_batch+1)*step_num)
            avg_loss = total_loss/(num_batch+1) if step_number != 0 else 0
            return avg_loss
        else:
            # from attack-test only
            return torch.cat(y_pred_probs_all, dim=0), torch.cat(y_tests_all), torch.cat(y_generated_texts_in_token, dim = 0)

    @staticmethod
    def word_token_list2feature_vector(list_of_list_of_tokens, vocab_size):
        contexts_matrix = np.zeros((len(list_of_list_of_tokens), vocab_size))
        for i, context in enumerate(list_of_list_of_tokens):
            for word_id in context:
                contexts_matrix[i, word_id] += 1
        return contexts_matrix

    # given the input of generated texts and compute the performance
    def perform_compute(self, y_pred_probs_all, y_tests_all, y_generated_texts_in_token, if_just_transfer_no_write=True,
                        y_hat_star_preds = None):
        results, _, y_preds = self.clf_seq.compute_performance(y_pred_probs_all, y_tests_all)
        self.log.info(f"the testing performance of the attack sceneario is {results}")
        # return # temporary stop
        if isinstance(y_generated_texts_in_token, list):
            y_generated_texts_in_token = torch.cat(y_generated_texts_in_token, dim=0)
        self.log.info(
            f"the y_generated_texts_in_token shape is {y_generated_texts_in_token.shape}")  # [batch_size]*max_len_seq
        # # ==== test 0 ====: different metrics
        # # save the whole batch results and finally, compute bleu scores and nll probabilitys
        # first, we can compute bleu scores by yourself alone, the previous computation is too complexs
        self.bleu = BLEU('BLEU', gram=[2, 3, 4, 5], if_use=cfg.use_bleu)
        y_generated_texts_in_token_word_format = tensor_to_tokens(y_generated_texts_in_token,
                                                                  dictionary=self.test_data.idx2word_dict)
        self.bleu.reset(test_text=y_generated_texts_in_token_word_format, real_text=self.final_sents_tokens)
        self.log.info(f"the bleu name is {self.bleu.get_name()} and score is {self.bleu.get_score()}")
        save_sample_path = cfg.save_samples_root + "sample.txt" # 'samples_{}_{}.txt'.format(None, None)  # phase, epoch {:05d}
        if if_just_transfer_no_write == False:
            self.log.info(f"the model is {cfg.model_selection}; the saved directory is {cfg.save_samples_root}")
            if not os.path.exists(cfg.save_samples_root):
                os.makedirs(cfg.save_samples_root)
                write_tokens(cfg.save_samples_root + "test.txt", self.final_sents_tokens,
                             if_just_transfer_no_write=if_just_transfer_no_write)
        written_tokens = write_tokens(save_sample_path, y_generated_texts_in_token_word_format,
                                      if_just_transfer_no_write=if_just_transfer_no_write)
        # self.log.info(f"the generated text is: \n {written_tokens}")

        y_generated_texts_in_feature_vectors = self.word_token_list2feature_vector(y_generated_texts_in_token.tolist(),
                                                                                   len(self.test_data.word2idx_dict))
        # #### compared text to the generated text ####
        test_contexts_feature_vectors = self.word_token_list2feature_vector(
            torch.cat(self.test_contexts, dim=0).tolist(), len(self.test_data.word2idx_dict))
        # #### contruct the recency metrics from the latest posts ####
        test_recency_feature_vectors = [seq[-1] for seq in
                                        self.test_seqs_without_final_sent]  # list of tensor of shape [max_seq_len]
        test_recency_feature_vectors = self.word_token_list2feature_vector(test_recency_feature_vectors,
                                                                           len(self.test_data.word2idx_dict))
        # #### for stance text comparison and computation ####
        test_stance_feature_vectors = []
        for j, seq in enumerate(self.test_relevancy_scores): # # num_seqs*max_seq_len
            # print(seq)
            matched_idx = [x[0] for x in enumerate(seq.flatten()) if x[1]>0]
            if len(matched_idx) == 0:
                matched_idx = [-1]
            test_stance_feature_vectors.append(self.test_seqs_without_final_sent[j][random.choice(matched_idx)])
        test_stance_feature_vectors = self.word_token_list2feature_vector(test_stance_feature_vectors,
                                                                           len(self.test_data.word2idx_dict))

        # self.log.info(f"{y_generated_texts_in_feature_vectors.shape, test_contexts_feature_vectors.shape}")
        relevancy_scores = 0
        recency_scores = 0
        stance_scores = 0
        for i in range(len(y_generated_texts_in_feature_vectors)):
            one_relevancy_score = 0
            one_recency_score = 0
            one_stance_score = 0
            for index, lda in enumerate(self.ldas):
                # self.log.info(y_generated_texts_in_feature_vectors[i].shape)
                # self.log.info(lda.transform(y_generated_texts_in_feature_vectors[i].reshape(1, -1)).shape)
                # error debug, lda.transform expected 2D array rather than the flatterned 1D
                one_relevancy_score += 1- cosine(lda.transform(y_generated_texts_in_feature_vectors[i].reshape(1, -1)),
                                              lda.transform(test_contexts_feature_vectors[i].reshape(1, -1)))
                one_recency_score += 1- cosine(lda.transform(y_generated_texts_in_feature_vectors[i].reshape(1, -1)),
                                              lda.transform(test_recency_feature_vectors[i].reshape(1, -1)))
                one_stance_score += 1- cosine(lda.transform(y_generated_texts_in_feature_vectors[i].reshape(1, -1)),
                                            lda.transform(test_stance_feature_vectors[i].reshape(1, -1)))
            relevancy_scores += one_relevancy_score / (len(self.ldas))
            recency_scores += one_recency_score / (len(self.ldas))
            stance_scores += one_stance_score / (len(self.ldas))
        relevancy_scores /= (len(y_generated_texts_in_feature_vectors))
        recency_scores /= (len(y_generated_texts_in_feature_vectors))
        stance_scores /= (len(y_generated_texts_in_feature_vectors))

        if y_hat_star_preds is not None:
            # ==== attack rated computation ====:
            def attack_rate_computation(y_tests_all, y_preds, y_hat_star_preds):
                attack_cnt = 0
                for i in range(len(y_tests_all)):
                    if y_tests_all[i] == y_hat_star_preds[i]:
                        if y_preds[i] != y_hat_star_preds[i]:
                            attack_cnt += 1
                attack_rate = attack_cnt/len(y_tests_all)
                return attack_rate
            attack_rate = attack_rate_computation(y_tests_all, y_preds, y_hat_star_preds)
            self.log.info(f"the attack_rate is {attack_rate}")

        self.log.info(f"the relevancy score is {relevancy_scores}")
        self.log.info(f"the recency score is {recency_scores}")
        self.log.info(f"the stance score is {stance_scores}")



    def test_generator_performance(self, clf_for_pred=None, y_hat_star_preds=None, if_just_transfer_no_write=True):
        y_pred_probs_all, y_tests_all, y_generated_texts_in_token = self.general_adv_train_both(step_number=1, mode="attack", stage="test", clf_for_pred=clf_for_pred)
        self.perform_compute(y_pred_probs_all, y_tests_all, y_generated_texts_in_token, y_hat_star_preds=y_hat_star_preds,
                             if_just_transfer_no_write=if_just_transfer_no_write)

    ###########################




