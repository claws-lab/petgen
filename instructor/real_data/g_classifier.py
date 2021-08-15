import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np

from torch.autograd import Variable
import config as cfg

class Generic_Clf(nn.Module):

    @staticmethod
    # padding_seq = True, max_len_seq_lstm = 20
    def train_classifier_lstm(model, seqs, labels, batch_size,
                              loss_function, optimizer,
                               vocab_size,
                              if_malcom_special_condition=False):
        """
        input: self.train_seqs, self.train_labels, self.test_seqs, self.test_labels
        :return:
        """
        avg_train_loss = 0
        model.train()
        for number_batch, idx in enumerate(range(0, len(seqs)-batch_size, batch_size)):
            optimizer.zero_grad()
            seqs_batch = seqs[idx:idx+batch_size]
            labels_batch = labels[idx:idx+batch_size]
            pred_probs = []
            for i in range(batch_size):
                seq = seqs_batch[i] # tensor: num_sents*max_sent_len
                if cfg.if_linear_embedding and not if_malcom_special_condition:
                    # print(seq.shape)
                    seq = F.one_hot(seq, vocab_size).float() # seq_len*sent_len*vocab_size
                elif cfg.if_linear_embedding and if_malcom_special_condition:
                    assert len(seq) == 2 #
                    assert isinstance(seq, list) or isinstance(seq, tuple)
                    seq = [ F.one_hot(i, vocab_size).float() for i in seq ]
                pred_prob = model(seq)
                pred_probs.append(pred_prob)
            # compute the loss
            pred_probs = torch.cat(pred_probs, dim=0)
            labels_batch = torch.cat(labels_batch)
            loss = loss_function(pred_probs, labels_batch)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item() # get the loss for the whole batch samples
        return avg_train_loss/((number_batch+1))

    @staticmethod
    def test_classifier_lstm(model, seqs, labels, batch_size, loss_function,
                              vocab_size,
                             if_malcom_special_condition=False):
        y_pred_probs = []
        y_tests = []


        # model.eval()
        with torch.no_grad():
            # print(f"{len(seqs) - batch_size, batch_size}")
            for times, idx in enumerate(range(0, len(seqs) - batch_size, batch_size)):
                seqs_batch = seqs[idx:idx + batch_size]
                labels_batch = labels[idx:idx + batch_size]

                for i in range(batch_size):
                    seq = seqs_batch[i]  # tensor: num_sents*max_sent_len
                    if cfg.if_linear_embedding and not if_malcom_special_condition:
                        seq = F.one_hot(seq, vocab_size).float()
                    elif cfg.if_linear_embedding and if_malcom_special_condition:
                        seq = [F.one_hot(element, vocab_size).float() for element in seq ]
                    pred_prob = model(seq)
                    y_pred_probs.append(pred_prob)
                y_tests.extend(labels_batch)
            y_pred_probs = torch.cat(y_pred_probs, dim=0)
            y_tests = torch.cat(y_tests)
            avg_test_loss = loss_function(y_pred_probs, y_tests)/(times + 1)
        return avg_test_loss, y_pred_probs, y_tests

    @staticmethod
    def compute_performance(y_pred_probs, y_tests, show_pred_label=False):
        y_preds = torch.argmax(y_pred_probs, dim=1)
        y_tests, y_preds = y_tests.detach().cpu(), y_preds.detach().cpu()
        results = {
            'precision': precision_score(y_tests, y_preds),
            'recall': recall_score(y_tests, y_preds),
            'f1': f1_score(y_tests, y_preds),
            'accuracy': accuracy_score(y_tests, y_preds)
        }
        # ==== for the debugging setting ====
        # if show_pred_label:
        #     indices = torch.randperm(len(y_preds))[:40]
        #     print(y_preds[indices])
        # print(results)
        return results, y_tests, y_preds


class TwoLevelLstmClassifier(Generic_Clf):
    def __init__(self, embedding_dim, seq_hidden_dim, seqs_hidden_dim, vocab_size, label_size,
                 max_len_seq_lstm, max_len_sent):
        super(TwoLevelLstmClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.seq_hidden_dim = seq_hidden_dim
        self.seqs_hidden_dim = seqs_hidden_dim
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.max_len_seq_lstm = max_len_seq_lstm
        self.max_len_sent = max_len_sent

        if cfg.if_linear_embedding:
            # ==== approach 1 ====: by linear layer, for text generation purpose, learn from discriminator setting
            self.emb = nn.Linear(vocab_size, embedding_dim, bias=False)
        else:
            # ==== approach 0 ====: by embedding layer
            self.emb = nn.Embedding(vocab_size, embedding_dim)

        # assert cfg.if_linear_embedding == True # we have to use the linear embedding due to the attack module

        self.seq_lstm = nn.LSTM(embedding_dim, seq_hidden_dim)
        self.seqs_lstm = nn.LSTM(seq_hidden_dim, seqs_hidden_dim)
        self.linear = nn.Linear(max_len_seq_lstm * seq_hidden_dim + 1 * seqs_hidden_dim, label_size)

    def forward(self, sequences):
        """
        :param sequences: a seq of sentence, shape [max_len_sent]*max_len_seq; i.e. [max_len_seq]*max_len_sent
        :return:
        """
        sents_emb = self.emb(sequences)  # [max_len_seq]*max_len_sent*embed_dim
        # by change axis, or view-- how to decide the index ordering is correct
        _, (h_n, _) = self.seq_lstm(sents_emb.view(self.max_len_sent, len(sequences), -1))  #
        # in some LSTM h_n, we also just pick the final
        # **h_n** of shape `(1, len(sequences), hidden_size)
        _, (hn_seqs, _) = self.seqs_lstm(
            h_n.view(len(sequences), 1, self.seq_hidden_dim))  # shape (MAX_NUM_SEQ, 1, hidden_dimmension)
        # print(hn_seqs.shape) # (1, 1, hidden_dim)
        feature_vec = torch.cat((h_n.view(1, len(sequences) * self.seq_hidden_dim),
                                 hn_seqs.view(1, self.seqs_hidden_dim)), dim=1) # (1, all_feature_dimension)
        pred_label = self.linear(feature_vec)  # 1*label_size
        # pred_label = pred_label.view(1, pred_label.shape[-1]) # torch.squeeze(pred_label) to ([2]) not good for
        # following cross entroy, we should have the batch number!
        return pred_label

# ==== the initial classifier for interface purpose ====
class TIES_V2(TwoLevelLstmClassifier):
    def __init__(self, embedding_dim, seq_hidden_dim, seqs_hidden_dim, vocab_size, label_size,
                 max_len_seq_lstm, max_len_sent):
        super(TIES_V2, self).__init__(embedding_dim, seq_hidden_dim, seqs_hidden_dim, vocab_size, label_size,
                 max_len_seq_lstm, max_len_sent)

class CNN_Clf(TwoLevelLstmClassifier):
    def __init__(self, embedding_dim, seq_hidden_dim, seqs_hidden_dim, vocab_size, label_size,
                 max_len_seq_lstm, max_len_sent):
        super(CNN_Clf, self).__init__(embedding_dim, seq_hidden_dim, seqs_hidden_dim, vocab_size, label_size,
                 max_len_seq_lstm, max_len_sent)

if __name__ == "main":
    generic_classifier = Generic_Clf()