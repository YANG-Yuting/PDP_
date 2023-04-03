import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from BERT.tokenization import BertTokenizer
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
import os
import random
from numpy.random import choice
from collections import deque
from torch.nn import functional as F
# from transformers import RobertaTokenizer


class Dataset_LSTM(Dataset):
    """ Dataset for mr/imdb-LSTM """

    def __init__(self, args):
        self.args = args

    def convert_examples_to_features(self, examples, pad_token='<pad>'):
        """Pad & word2id & Get syn"""
        features = []
        oov_id = self.args.word2id['<oov>']
        for text_a in examples:
            """padding on the left with '<pad>'"""
            pad_sequences = []
            if len(text_a) > self.args.max_seq_length:
                pad_sequences.append(text_a[:self.args.max_seq_length])
            else:
                pad_sequences.append([pad_token] * (self.args.max_seq_length - len(text_a)) + text_a)
            pad_sequences = [self.args.word2id.get(w, oov_id) for seq in pad_sequences for w in seq]

            features.append(pad_sequences)
        return features

    def transform_text(self, data, labels):
        """Build Dataloader"""
        eval_features = self.convert_examples_to_features(data)
        all_input_ids = torch.tensor(eval_features, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        all_data = TensorDataset(all_input_ids, labels)

        return all_data
        # if torch.cuda.device_count() > 1:
        #     sampler = torch.utils.data.distributed.DistributedSampler(all_data)
        #     nw = min([os.cpu_count(), self.args.batch_size if self.args.batch_size > 1 else 0, 8])
        #     # nw = 0
        #     dataloader = DataLoader(all_data, sampler=sampler, pin_memory=True, num_workers=nw,
        #                             batch_size=self.args.batch_size) # batch_
        # else:
        #     sampler = SequentialSampler(all_data)
        #     dataloader = DataLoader(all_data, sampler=sampler, batch_size=self.args.batch_size)

        # return dataloader


class Dataset_LSTM_ascc(Dataset):
    """ Dataset for mr/imdb-LSTM while train/test ascc """

    def __init__(self, args):
        self.args = args

    def __len__(self):
        return self.nSamples - 1

    def convert_examples_to_features(self, examples, pad_token='<pad>'):
        """Pad & word2id & Get syn"""
        features = []
        oov_id = self.args.word2id['<oov>']
        for text_a in examples:
            """padding on the left with '<pad>'"""
            pad_sequences = []
            if len(text_a) > self.args.max_seq_length:
                pad_sequences.append(text_a[:self.args.max_seq_length])
            else:
                pad_sequences.append([pad_token] * (self.args.max_seq_length - len(text_a)) + text_a)
            pad_sequences = [self.args.word2id.get(w, oov_id) for seq in pad_sequences for w in seq]
            """get syn"""
            if ' '.join(text_a) in self.args.candidate_bags.keys():
                # 对原数据集中文本获取同义词
                syns = []
                candidate_bag = self.args.candidate_bags[' '.join(text_a)]
                for idx in range(len(text_a)):
                    neghs = candidate_bag[text_a[idx]].copy()
                    # neghs.remove(text_a[idx])  # remove self in candidates
                    if len(neghs) > self.args.syn_num:  # max num of syn: 20
                        neghs = neghs[:self.args.syn_num]
                    else:
                        neghs = [pad_token] * (self.args.syn_num - len(neghs)) + neghs  # pad for syn
                    neghs = [self.args.word2id.get(n, oov_id) for n in neghs]
                    syns.append(neghs)
                if len(syns) > self.args.max_seq_length:
                    syns = syns[:self.args.max_seq_length]
                else:
                    pad_syn = [self.args.word2id[pad_token]] * self.args.syn_num
                    syns = [pad_syn for i in range(self.args.max_seq_length - len(syns))] + syns

                syns_valid = np.where(np.array(syns)!= self.args.word2id[pad_token], 1, 0).tolist() # keep syn positions

            assert len(pad_sequences) == self.args.max_seq_length
            assert len(syns) == self.args.max_seq_length
            assert len(syns_valid) == self.args.max_seq_length
            features.append([pad_sequences, syns, syns_valid])
        return features

    def transform_text(self, data, labels):
        """Build Dataloader"""
        eval_features = self.convert_examples_to_features(data)
        all_input_ids = torch.tensor([f[0] for f in eval_features], dtype=torch.long)
        all_input_syns = torch.tensor([f[1] for f in eval_features], dtype=torch.long)
        all_input_syn_valids = torch.tensor([f[2] for f in eval_features], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        all_data = TensorDataset(all_input_ids, all_input_syns, all_input_syn_valids, labels)
        # all_data = TensorDataset(all_input_ids, labels)

        return all_data
        # if torch.cuda.device_count() > 1:
        #     sampler = torch.utils.data.distributed.DistributedSampler(all_data)
        #     nw = min([os.cpu_count(), self.args.batch_size if self.args.batch_size > 1 else 0, 8])
        #     # nw = 0
        #     dataloader = DataLoader(all_data, sampler=sampler, pin_memory=True, num_workers=nw,
        #                             batch_size=self.args.batch_size) # batch_
        # else:
        #     sampler = SequentialSampler(all_data)
        #     dataloader = DataLoader(all_data, sampler=sampler, batch_size=self.args.batch_size)

        # return dataloader


class Dataset_LSTM_ascc_imdb(Dataset):
    """
    Dataset for imdb-lstm while load model trained by ASCC project(main differences):
        (1) Filter strange symbols;
        (2) word2id is generated by ASCC;
        (3) Pad on right with 0 and skip oov words.
    """

    def __init__(self, args):
        self.args = args
        with open('/home/huangpei/ASCC-main/temp/imdb_word2id.pickle', 'rb') as f:
            args.word2id = pickle.load(f)
        args.word2id['<pad>'] = 0

    """Pad & word2id"""
    def convert_examples_to_features(self, examples):
        features = []
        for text_a in examples:
            # padding on the right with 0; skip oov words
            text_a = ' '.join(text_a)
            filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            for c in filters:
                text_a = text_a.replace(c, ' ')
            text_a = text_a.split(' ')
            text_ids = [self.args.word2id[w] for w in text_a if w in self.args.word2id.keys()]
            if len(text_ids) > self.args.max_seq_length:
                pad_sequences = text_ids[:self.args.max_seq_length]
            else:
                pad_sequences = text_ids + [0] * (self.args.max_seq_length - len(text_ids))  # 右边填充
            features.append(pad_sequences)
        return features

    """Build Dataloader (data: list of list of word)"""
    def transform_text(self, data, labels):
        eval_features = self.convert_examples_to_features(data)
        all_input_ids = torch.tensor(eval_features, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        all_data = TensorDataset(all_input_ids, labels)

        if torch.cuda.device_count() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(all_data)
            nw = min([os.cpu_count(), self.args.batch_size if self.args.batch_size > 1 else 0, 8])
            dataloader = DataLoader(all_data, batch_sampler=sampler, pin_memory=True, num_workers=nw)
        else:
            sampler = SequentialSampler(all_data)
            dataloader = DataLoader(all_data, sampler=sampler, batch_size=self.args.batch_size)

        return dataloader


class Dataset_LSTM_snli(Dataset):
    """ Dataset for mr/imdb-LSTM
        Mind: For snli-lstm(ESIM), input is word embedding"""

    def __init__(self, args):
        self.args = args
        self.word_emb_dim = 300
        self.word_vec = pickle.load(open(args.data_path+'snli/word_vec.pkl', 'rb'))
        self.word_vec[self.args.full_dict['<oov>']] = pickle.load(open(args.data_path + 'snli/glove_unk.pkl', 'rb'))


    def convert_examples_to_features(self, examples):  # examples: [s1 [SEP] s2, s1 [SEP] s2,...]
        """Pad & word2vector"""

        features = []
        for i in range(len(examples)):
            s1, s2 = ' '.join(examples[i]).split(' [SEP] ')
            s1 = s1.split(' ')
            s2 = s2.split(' ')
            """pad on word vector with 0"""
            s1_len = len(s1)
            if s1_len > self.args.max_seq_length:
                s1 = s1[:self.args.max_seq_length]
            s1_embs = np.zeros((self.args.max_seq_length, self.word_emb_dim))
            for (j, w) in enumerate(s1):
                id = self.args.full_dict[w] if w in self.args.full_dict.keys() else self.args.full_dict['<oov>']  # 42391: <oov>
                s1_embs[j] = self.word_vec[id]
            """load word vec"""
            s2_len = len(s2)
            if s2_len > self.args.max_seq_length:
                s2 = s2[:self.args.max_seq_length]
            s2_embs = np.zeros((self.args.max_seq_length, self.word_emb_dim))
            for (j, w) in enumerate(s2):
                id = self.args.full_dict[w] if w in self.args.full_dict.keys() else self.args.full_dict['<oov>']
                s2_embs[j] = self.word_vec[id]

            features.append([s1_embs, s1_len, s2_embs, s2_len])
        return features

    def transform_text(self, data, labels):  # data: [s1 [SEP] s2, s1 [SEP] s2,...]
        """Build Dataloader"""
        eval_features = self.convert_examples_to_features(data)
        s1_ids = torch.tensor([f[0] for f in eval_features], dtype=torch.float)
        s1_lens = torch.tensor([f[1] for f in eval_features], dtype=torch.float)
        s2_ids = torch.tensor([f[2] for f in eval_features], dtype=torch.float)
        s2_lens = torch.tensor([f[3] for f in eval_features], dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        all_data = TensorDataset(s1_ids, s1_lens, s2_ids, s2_lens, labels)

        return all_data


class Dataset_LSTM_snli_ascc(Dataset):
    """ Dataset for mr/imdb-LSTM
        Mind: For snli-lstm(ESIM), input is word embedding"""

    def __init__(self, args):
        self.args = args
        self.word_emb_dim = 300
        self.word_vec = pickle.load(open(args.data_path+'snli/word_vec.pkl', 'rb'))
        self.word_vec[args.full_dict['<oov>']] = pickle.load(open(args.data_path + 'snli/glove_unk.pkl', 'rb'))


    def convert_examples_to_features(self, examples, pad_token='<pad>'):  # examples: [s1 [SEP] s2, s1 [SEP] s2,...]
        """Pad & word2vector"""

        features = []
        s1_embs_all, s1_len_all, s2_embs_all, s2_len_all, s2_syns_all, s2_syn_embs_all, s2_syns_valid_all = [],[],[],[],[],[],[]
        for i in range(len(examples)):
            # print(i)
            s1, s2 = ' '.join(examples[i]).split(' [SEP] ')
            s1 = s1.split(' ')
            s2 = s2.split(' ')
            """pad on word vector with 0"""
            s1_len = len(s1)
            if s1_len > self.args.max_seq_length:
                s1 = s1[:self.args.max_seq_length]
            s1_embs = np.zeros((self.args.max_seq_length, self.word_emb_dim))
            for (j, w) in enumerate(s1):
                id = self.args.full_dict[w] if w in self.args.full_dict.keys() else self.args.full_dict['<oov>']  # 42391: <oov>
                s1_embs[j] = self.word_vec[id]
            """load word vec"""
            s2_len = len(s2)
            if s2_len > self.args.max_seq_length:
                s2 = s2[:self.args.max_seq_length]
            s2_embs = np.zeros((self.args.max_seq_length, self.word_emb_dim))
            for (j, w) in enumerate(s2):
                id = self.args.full_dict[w] if w in self.args.full_dict.keys() else self.args.full_dict['<oov>']
                s2_embs[j] = self.word_vec[id]

            """get syn"""
            if ' '.join(s2) in self.args.candidate_bags.keys():
                # 对原数据集中文本获取同义词
                s2_syns = []
                s2_syn_embs = np.zeros((self.args.max_seq_length, self.args.syn_num, self.word_emb_dim))
                candidate_bag = self.args.candidate_bags[' '.join(s2)]
                for idx in range(len(s2)):
                    neghs = candidate_bag[s2[idx]].copy()
                    if len(neghs) > self.args.syn_num:
                        neghs = neghs[:self.args.syn_num]
                    neghs = [self.args.full_dict[w] if w in self.args.full_dict.keys() else self.args.full_dict['<oov>'] for w in neghs ] # word2id
                    for k in range(min(self.args.syn_num, len(neghs))):
                        s2_syn_embs[j, k] = self.word_vec[neghs[k]]

                    if len(neghs) < self.args.syn_num:
                        neghs = [self.args.full_dict[pad_token]] * (self.args.syn_num - len(neghs)) + neghs  # pad for syn
                    s2_syns.append(neghs)

                if len(s2_syns) > self.args.max_seq_length:
                    s2_syns = s2_syns[:self.args.max_seq_length]
                else:
                    pad_syn = [self.args.full_dict[pad_token]] * self.args.syn_num
                    s2_syns = [pad_syn for i in range(self.args.max_seq_length - len(s2_syns))] + s2_syns

                s2_syns_valid = np.where(np.array(s2_syns) != self.args.full_dict[pad_token], 1, 0).tolist()  # keep syn positions
            s1_embs_all.append(s1_embs)
            s1_len_all.append(s1_len)
            s2_embs_all.append(s2_embs)
            s2_len_all.append(s2_len)
            s2_syns_all.append(s2_syns)
            s2_syn_embs_all.append(s2_syn_embs)
            s2_syns_valid_all.append(s2_syns_valid)

            # features.append([s1_embs, s1_len, s2_embs, s2_len, s2_syns, s2_syn_embs, s2_syns_valid])
        s1_embs_all = torch.tensor(s1_embs_all, dtype=torch.float)
        s1_len_all = torch.tensor(s1_len_all, dtype=torch.float)
        s2_embs_all = torch.tensor(s2_embs_all, dtype=torch.float)
        s2_len_all = torch.tensor(s2_len_all, dtype=torch.float)
        s2_syns_all = torch.tensor(s2_syns_all, dtype=torch.float)
        s2_syn_embs_all = torch.tensor(s2_syn_embs_all, dtype=torch.float)
        s2_syns_valid_all = torch.tensor(s2_syns_valid_all, dtype=torch.float)
        return s1_embs_all, s1_len_all, s2_embs_all, s2_len_all, s2_syns_all, s2_syn_embs_all, s2_syns_valid_all

    def transform_text(self, data, labels):  # data: [s1 [SEP] s2, s1 [SEP] s2,...]
        """Build Dataloader"""
        s1_embs_all, s1_len_all, s2_embs_all, s2_len_all, s2_syns_all, s2_syn_embs_all, s2_syns_valid_all = self.convert_examples_to_features(data)
        labels = torch.tensor(labels, dtype=torch.long)
        all_data = TensorDataset(s1_embs_all, s1_len_all, s2_embs_all, s2_len_all, s2_syns_all, s2_syn_embs_all, s2_syns_valid_all, labels)

        return all_data


class InputFeatures(object):
    """A single set of features of data for BERT"""
    def __init__(self, inputs):
        self.input_ids = inputs[0]
        self.input_mask = inputs[1]
        self.segment_ids = inputs[2]
        try:
            self.syns = inputs[3]
            self.syns_valid = inputs[4]
        except:
            pass


class Dataset_BERT(Dataset):
    """ Dataset for mr/imdb/snli-BERT
        Mind: for snli, s2 is concatenated with s1
    """

    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def convert_examples_to_features(self, examples, labels):
        input_ids_all, input_mask_all, segment_ids_all, model_nos_all, labels_all = [], [], [], [], []
        for (ex_index, text_a) in enumerate(examples):  # inputs: CLS text_a SEP
            # If SNLI, text_a should be: s1 "SEP" s2
            tokens_a = self.tokenizer.tokenize(' '.join(text_a))  # 109

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.args.max_seq_length - 2:
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.args.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            segment_ids_all.append(segment_ids)
            model_nos_all.append([0])  # 无意义
            labels_all.append(labels[ex_index])

            # for m in range(self.args.num_models):
            #     input_ids_all.append(input_ids)
            #     input_mask_all.append(input_mask)
            #     segment_ids_all.append(segment_ids)
            #     model_no = [m]
            #     # model_no = F.one_hot(torch.tensor([m]), num_classes=self.args.num_models)
            #     model_nos_all.append(model_no)
            #     labels_all.append(labels[ex_index])
        return input_ids_all, input_mask_all, segment_ids_all, model_nos_all, labels_all

    def get_token_num(self, text_a):
        tokens_a = self.tokenizer.tokenize(' '.join(text_a))
        # print('tokens_a', tokens_a)
        token_num_a = [0] * len(text_a)
        ii, jj = 0, 0
        while jj < len(tokens_a):
            # print(ii, jj, text_a[ii], tokens_a[jj])
            if tokens_a[jj] == text_a[ii]:
                token_num_a[ii] += 1
                ii += 1
                jj += 1
                # print('count')
            elif '##' in tokens_a[jj] or tokens_a[jj] in text_a[ii]:
                token_num_a[ii] += 1
                jj += 1
                # print('count')
            else:
                ii += 1
        #     print('token_num_a', token_num_a)
        # print('token_num_a', token_num_a)
        return token_num_a

    """Pad & tokenize"""

    def get_syn(self, examples):
        negbs_all = []
        for (ex_index, text_a) in enumerate(examples):  # inputs: CLS text_a SEP
            # print('text_a', len(text_a), text_a)
            if ' '.join(text_a) in self.args.candidate_bags.keys():
                # 对原数据集中文本获取同义词
                candidate_bag = self.args.candidate_bags[' '.join(text_a)]
                # print('candidate_bag', candidate_bag)
                negbs_a = [text_a.copy() for i in range(self.args.syn_num)]  # 第一条一定是自己？
                # print('negbs_a', negbs_a)
                # valid_a = [[0]*len(text_a) for i in range(self.args.syn_num)]
                # all_negbs = [['<pad>']*len(text_a) for i in range(self.args.syn_num)]  # 第一条一定是自己？
                for idx in range(len(text_a)):
                    negbs = candidate_bag[text_a[idx]].copy()  # 包含自己?
                    if len(negbs) > self.args.syn_num:
                        negbs = negbs[:self.args.syn_num]
                    for j in range(len(negbs)):
                        negbs_a[j][idx] = negbs[j]
                negbs_all.extend(negbs_a)
        #         print('negbs_a', negbs_a)
        # print('negbs_all', negbs_all)

        """syn2id"""
        # input_ids_all, input_mask_all, segment_ids_all = self.convert_examples_to_features(negbs_all)
        syn_ids_all, syn_input_mask_all, syn_segment_ids_all, valid_all = [], [], [], []
        """pad for different token of syns"""
        for (ex_index, text_a) in enumerate(examples):
            # orig
            # print('-------ex_index', ex_index)
            text_a = ["[CLS]"] + text_a + ["[SEP]"]
            # print('text_a', len(text_a), text_a)
            try:
                token_num_a = self.get_token_num(text_a)
            except:
                continue
            tokens_a = self.tokenizer.tokenize(' '.join(text_a))
            # print('tokens_a', len(tokens_a), tokens_a)
            # print('token_num_a', token_num_a)

            text_a_syn = negbs_all[ex_index*self.args.syn_num: (ex_index+1)*self.args.syn_num].copy()
            for syn in text_a_syn:
                syn = ["[CLS]"] + syn + ["[SEP]"]
                # print('syn', syn)
                try:
                    token_num_a_syn = self.get_token_num(syn)
                except:
                    continue
                # print('token_num_a_syn', token_num_a_syn)
                tokens_a_syn = self.tokenizer.tokenize(' '.join(syn))
                id_a_syn = self.tokenizer.convert_tokens_to_ids(tokens_a_syn)
                # print('id_a_syn', id_a_syn)
                id_a_syn_pad = id_a_syn.copy()
                ii, jj = 0, 0
                while ii < len(token_num_a):
                    # print('ii, jj', ii, jj)
                    if token_num_a[ii]==token_num_a_syn[ii]:
                        jj += 1
                    else:
                        pad_num = token_num_a[ii] - token_num_a_syn[ii]
                        # print('pad_num', pad_num)
                        if pad_num < 0:
                            id_a_syn_pad = id_a_syn_pad[:jj+token_num_a[ii]] + id_a_syn_pad[jj+token_num_a[ii]+(-pad_num):]
                        else:
                            id_a_syn_pad = id_a_syn_pad[:jj+token_num_a_syn[ii]] + [0] * pad_num + id_a_syn_pad[jj+token_num_a_syn[ii]:]
                        # print('id_a_syn_pad', id_a_syn_pad)
                        jj += token_num_a[ii]
                    ii += 1
                assert len(tokens_a) == len(id_a_syn_pad)


                segment_ids = [0] * len(id_a_syn_pad)
                input_mask = [1] * len(id_a_syn_pad)
                padding = [0] * (self.args.max_seq_length - len(id_a_syn_pad))
                id_a_syn_pad += padding
                segment_ids += padding
                input_mask += padding
                syn_ids_all.append(id_a_syn_pad)
                syn_input_mask_all.append(input_mask)
                syn_segment_ids_all.append(segment_ids)

                # idx = 0
                # jdx = 1
                # while jdx < len(tokens)-2:  # 除去[CLS]和[SEP]位置
                #     neghs = candidate_bag[text_a[idx]].copy()
                #     if len(neghs) > self.args.syn_num:
                #         neghs = neghs[:self.args.syn_num]
                #     else:
                #         neghs += ['<pad>'] * (self.args.syn_num - len(neghs))
                #     neghs_ids = []  # syn_num, id_num
                #     for n in neghs:
                #         n_ids = self.tokenizer.convert_tokens_to_ids(n)
                #         if len(n_ids) > sharp_num[idx]:
                #             n_ids = n_ids[:sharp_num[idx]]
                #         else:
                #             n_ids += [0] * (sharp_num[idx]-len(n_ids))
                #         neghs_ids.append(n_ids)
                #     neghs_ids = np.array(neghs_ids).reshape([sharp_num[idx], self.args.syn_num])
                #     syns[jdx:jdx+sharp_num[idx]] = neghs_ids
                #     jdx += sharp_num[idx]
                #     idx += 1
                
            """syn_valid 只保留是同义词集合中替换得到的"""
            cls_valid = [1]+[0]*(self.args.syn_num-1)
            sep_valid = [1]+[0]*(self.args.syn_num-1)
            syn_valid = [cls_valid]
            # print('syn_ids_all', syn_ids_all)
            # print('token_num_a', len(token_num_a), token_num_a)
            temp_a = text_a[1:-1].copy()
            candidate_bag = self.args.candidate_bags[' '.join(temp_a)]
            negbs_num = [1]  # cls
            for idx in range(len(temp_a)):
                negbs_num.append(len(candidate_bag[temp_a[idx]]))
            negbs_num.append(1)  # sep
            # print('negbs_num', len(negbs_num), negbs_num)
            for i in range(len(token_num_a)):
                num_neg = negbs_num[i]
                token_num = token_num_a[i]
                # print(i, num_neg, token_num)
                if token_num == 1:
                    syn_valid.append([1]*min(num_neg, self.args.syn_num) + [0]*(self.args.syn_num-min(num_neg, self.args.syn_num)))
                else:
                    while token_num > 0:
                        # print(syn_valid)
                        syn_valid.append([1] * min(num_neg, self.args.syn_num) + [0] * (self.args.syn_num - min(num_neg, self.args.syn_num)))
                        token_num = token_num - 1
            syn_valid.append(sep_valid)
            if len(syn_valid) >= self.args.max_seq_length:
                syn_valid = syn_valid[:self.args.max_seq_length]
            else:
                for j in range(self.args.max_seq_length - len(syn_valid)):
                    syn_valid.append([0]*self.args.syn_num)
                # print('id_a_syn_pad', id_a_syn_pad)
                # print('syn_valid', syn_valid)
                        
                # syn_valid = np.where(np.array(id_a_syn_pad) != 0, 1, 0).tolist()  # keep syn positions
            syn_valid = np.array(syn_valid).T.tolist()
            valid_all.extend(syn_valid)

        # print(len(valid_all), len(valid_all[0]))
        # print(len(syn_ids_all))
        return syn_ids_all, syn_input_mask_all, syn_segment_ids_all, valid_all

    """Build Dataloader"""
    def transform_text(self, data, labels):
        input_ids_all, input_mask_all, segment_ids_all, model_nos_all, labels_all = self.convert_examples_to_features(data, labels)
        all_input_ids = torch.tensor(input_ids_all, dtype=torch.long)
        all_input_mask = torch.tensor(input_mask_all, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids_all, dtype=torch.long)
        all_model_nos = torch.tensor(model_nos_all, dtype=torch.long)
        all_labels = torch.tensor(labels_all, dtype=torch.long)
        all_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_model_nos, all_labels)

        if self.args.ascc:
            syn_ids_all, syn_input_mask_all, syn_segment_ids_all, valid_all = self.get_syn(data)
            syn_ids_all = torch.tensor(syn_ids_all, dtype=torch.long)
            syn_input_mask_all = torch.tensor(syn_input_mask_all, dtype=torch.long)
            syn_segment_ids_all = torch.tensor(syn_segment_ids_all, dtype=torch.long)
            valid_all = torch.tensor(valid_all, dtype=torch.long)
            all_data_syn = TensorDataset(syn_ids_all, syn_input_mask_all, syn_segment_ids_all, valid_all)

            assert len(syn_ids_all) == self.args.syn_num*len(input_ids_all)
            return all_data, all_data_syn
        else:
            return all_data, None
        # Run prediction for full data
        # if torch.cuda.device_count() > 1:
        #     sampler = torch.utils.data.distributed.DistributedSampler(all_data)
        #     sampler = torch.utils.data.BatchSampler(sampler, self.args.batch_size, drop_last=True)
        #     nw = min([os.cpu_count(), self.args.batch_size if self.args.batch_size > 1 else 0, 8])
        #     dataloader = DataLoader(all_data, batch_sampler=sampler, pin_memory=True, num_workers=nw)
        # else:
        #     sampler = SequentialSampler(all_data)
        #     dataloader = DataLoader(all_data, sampler=sampler, batch_size=self.args.batch_size)
        #
        # return dataloader


class Dataset_ROBERTA(Dataset):
    def __init__(self, args):
        self.args = args
        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_path+'roberta-base')

    def convert_examples_to_features(self, examples):
        input_ids_all, input_mask_all, segment_ids_all = [], [], []
        for (ex_index, text_a) in enumerate(examples):  # inputs:  text_a
            # print('text_a', text_a)
            input_ids = self.tokenizer(' '.join(text_a))['input_ids']  # 包含了cls/sep/<s>？
            input_mask = self.tokenizer(' '.join(text_a))['attention_mask']
            assert len(input_ids) > 2

            if len(input_ids) > self.args.max_seq_length:
                input_ids = input_ids[:self.args.max_seq_length]
                input_mask = input_mask[:self.args.max_seq_length]
            segment_ids = [0] * len(input_ids)

            # left-pad up to the sequence length.
            padding = [self.args.pad_token_id] * (self.args.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += [0] * len(padding)
            segment_ids += [0] * len(padding)

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            segment_ids_all.append(segment_ids)


        return input_ids_all, input_mask_all, segment_ids_all

    def get_token_num(self, text_a):
        tokens_a = self.tokenizer.tokenize(' '.join(text_a))
        # print('text_a', text_a)
        # print('tokens_a', tokens_a)
        token_num_a = [0] * len(text_a)
        ii, jj = 0, 0
        while jj < len(tokens_a):
            if ii >= len(text_a):
                ii = ii-1
            # print(ii, jj, text_a[ii], tokens_a[jj])

            if text_a[ii] in tokens_a[jj] or tokens_a[jj].replace('Ġ', '') in text_a[ii]:
                token_num_a[ii] += 1
                ii += 1
                # print('count')
            elif 'Ġ' not in tokens_a[jj]:
                token_num_a[ii - 1] += 1
                # print('count')
            jj += 1
        #     print('token_num_a', token_num_a)
        # print('token_num_a', token_num_a)
        return token_num_a

    def get_syn(self, examples):
        negbs_all, valid_all = [], []
        for (ex_index, text_a) in enumerate(examples):  # inputs: CLS text_a SEP
            # print('text_a', text_a)
            if ' '.join(text_a) in self.args.candidate_bags.keys():
                # 对原数据集中文本获取同义词
                candidate_bag = self.args.candidate_bags[' '.join(text_a)]
                # print('candidate_bag', candidate_bag)
                negbs_a = [text_a.copy() for i in range(self.args.syn_num)]  # 第一条一定是自己？
                # print('negbs_a', negbs_a)
                # valid_a = [[0]*len(text_a) for i in range(self.args.syn_num)]
                # all_negbs = [['<pad>']*len(text_a) for i in range(self.args.syn_num)]  # 第一条一定是自己？
                for idx in range(len(text_a)):
                    negbs = candidate_bag[text_a[idx]].copy()  # 包含自己?
                    if len(negbs) > self.args.syn_num:
                        negbs = negbs[:self.args.syn_num]
                    for j in range(len(negbs)):
                        negbs_a[j][idx] = negbs[j]
                negbs_all.extend(negbs_a)
                # print('negbs_a', negbs_a)
        # print('negbs_all', negbs_all)

        """syn2id"""
        # input_ids_all, input_mask_all, segment_ids_all = self.convert_examples_to_features(negbs_all)
        syn_ids_all, syn_input_mask_all, syn_segment_ids_all = [], [], []
        """pad for different token of syns"""
        for (ex_index, text_a) in enumerate(examples):
            # orig
            # text_a = ["[CLS]"] + text_a + ["[SEP]"]
            token_num_a = [1] + self.get_token_num(text_a) + [1]
            tokens_a = self.tokenizer(' '.join(text_a))['input_ids']
            # print('-------ex_index', ex_index)
            # print('text_a', text_a)
            # print('tokens_a', tokens_a)
            # print('token_num_a', token_num_a)

            text_a_syn = negbs_all[ex_index*self.args.syn_num: (ex_index+1)*self.args.syn_num].copy()
            for syn in text_a_syn:
                # syn = ["[CLS]"] + syn + ["[SEP]"]
                # print('syn', syn)
                token_num_a_syn = [1] + self.get_token_num(syn) + [1]
                # print('token_num_a_syn', token_num_a_syn)
                # id_a_syn = self.tokenizer.convert_tokens_to_ids(tokens_a_syn)
                id_a_syn = self.tokenizer(' '.join(syn))['input_ids']  # 包含了cls/sep/<s>？
                input_mask = self.tokenizer(' '.join(syn))['attention_mask']
                # print('id_a_syn', id_a_syn)
                id_a_syn_pad = id_a_syn.copy()
                ii, jj = 0, 0
                while ii < len(token_num_a):
                    # print('ii, jj', ii, jj)
                    if token_num_a[ii] == token_num_a_syn[ii]:
                        jj += 1
                    else:
                        pad_num = token_num_a[ii] - token_num_a_syn[ii]
                        # print('pad_num', pad_num)
                        if pad_num < 0:
                            id_a_syn_pad = id_a_syn_pad[:jj+token_num_a[ii]] + id_a_syn_pad[jj+token_num_a[ii]+(-pad_num):]
                        else:
                            id_a_syn_pad = id_a_syn_pad[:jj+token_num_a_syn[ii]] + [0] * pad_num + id_a_syn_pad[jj+token_num_a_syn[ii]:]
                        # print('id_a_syn_pad', id_a_syn_pad)
                        jj += token_num_a[ii]
                    ii += 1
                # print('tokens_a', tokens_a)
                # print('id_a_syn_pad', id_a_syn_pad)

                # assert len(tokens_a) == len(id_a_syn_pad)

                segment_ids = [0] * len(id_a_syn_pad)
                input_mask = [1] * len(id_a_syn_pad)
                padding = [self.args.pad_token_id] * (self.args.max_seq_length - len(id_a_syn_pad))
                id_a_syn_pad += padding
                segment_ids += padding
                input_mask += padding
                syn_ids_all.append(id_a_syn_pad)
                syn_input_mask_all.append(input_mask)
                syn_segment_ids_all.append(segment_ids)

                # idx = 0
                # jdx = 1
                # while jdx < len(tokens)-2:  # 除去[CLS]和[SEP]位置
                #     neghs = candidate_bag[text_a[idx]].copy()
                #     if len(neghs) > self.args.syn_num:
                #         neghs = neghs[:self.args.syn_num]
                #     else:
                #         neghs += ['<pad>'] * (self.args.syn_num - len(neghs))
                #     neghs_ids = []  # syn_num, id_num
                #     for n in neghs:
                #         n_ids = self.tokenizer.convert_tokens_to_ids(n)
                #         if len(n_ids) > sharp_num[idx]:
                #             n_ids = n_ids[:sharp_num[idx]]
                #         else:
                #             n_ids += [0] * (sharp_num[idx]-len(n_ids))
                #         neghs_ids.append(n_ids)
                #     neghs_ids = np.array(neghs_ids).reshape([sharp_num[idx], self.args.syn_num])
                #     syns[jdx:jdx+sharp_num[idx]] = neghs_ids
                #     jdx += sharp_num[idx]
                #     idx += 1

                syn_valid = np.where(np.array(id_a_syn_pad) != self.args.pad_token_id, 1, 0).tolist()  # keep syn positions
                valid_all.append(syn_valid)
        return syn_ids_all, syn_input_mask_all, syn_segment_ids_all, valid_all

    def transform_text(self, data, labels):
        input_ids_all, input_mask_all, segment_ids_all = self.convert_examples_to_features(data)
        all_input_ids = torch.tensor(input_ids_all, dtype=torch.long)
        all_input_mask = torch.tensor(input_mask_all, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids_all, dtype=torch.long)
        all_labels = torch.tensor(labels, dtype=torch.long)
        # print(len(input_ids_all))
        # print(torch.max(all_input_ids))
        all_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)

        if self.args.ascc:
            syn_ids_all, syn_input_mask_all, syn_segment_ids_all, valid_all = self.get_syn(data)
            syn_ids_all = torch.tensor(syn_ids_all, dtype=torch.long)
            syn_input_mask_all = torch.tensor(syn_input_mask_all, dtype=torch.long)
            syn_segment_ids_all = torch.tensor(syn_segment_ids_all, dtype=torch.long)
            valid_all = torch.tensor(valid_all, dtype=torch.long)
            all_data_syn = TensorDataset(syn_ids_all, syn_input_mask_all, syn_segment_ids_all, valid_all)

            assert len(syn_ids_all) == self.args.syn_num*len(input_ids_all)
            return all_data, all_data_syn
        else:
            return all_data, [1]

def perturb_texts(args, orig_texts=None, texts=None, tf_vocabulary=None, change_ratio = 1):
    """Replace word with syn with highest freq"""

    select_sents = []
    if orig_texts is None:
        orig_texts=[None for i in range(len(texts))]
    for orig_text,text_str in zip(orig_texts,texts):
        # 获得候选集
        text_str = [word for word in text_str if word != '\x85']
        text_str = [word.replace('\x85', '') for word in text_str]
        if ' '.join(text_str) in args.candidate_bags.keys(): # 若该文本见过（train and test）
            candidate_bag = args.candidate_bags[' '.join(text_str)]
        else:
            # 针对攻击代码
            if orig_text:  # 若orig_text不为空（仅在攻击中会赋值，此时所有的texts对应一个orig text）
                orig_text = [word for word in orig_text if word != '\x85']
                orig_text = [word.replace('\x85', '') for word in orig_text]
                pos_tag = args.pos_tags[' '.join(orig_text)]  # 注意这里是对测试集做！
            else:
                pos_tag = pos_tagger.tag(text_str)  # 词性标注，耗时0.45s

            # str转id（同义词集是用id存的），有不存在于词汇表的，保留原str，之后分类需要
            text_ids = []
            for word_str in text_str:
                if word_str in args.full_dict.keys():
                    text_ids.append(args.full_dict[word_str])  # id
                else:
                    text_ids.append(word_str)  # str

            # 获得候选集
            candidate_bag = {}
            for j in range(len(text_ids)):  # 对于每个位置
                word = text_ids[j]
                pos = pos_tag[j][1]  # 当前词语词性
                neigbhours = [word]
                if isinstance(word, int) and pos in args.pos_list and word < len(args.word_candidate):
                    if pos.startswith('JJ'):
                        pos = 'adj'
                    elif pos.startswith('NN'):
                        pos = 'noun'
                    elif pos.startswith('RB'):
                        pos = 'adv'
                    elif pos.startswith('VB'):
                        pos = 'verb'
                    neigbhours.extend(args.word_candidate[word][pos])  # 候选集
                candidate_bag[text_str[j]] = [args.inv_full_dict[i] if isinstance(i, int) else i for i in neigbhours]  # id转为str

        replace_text = text_str.copy()
        for i in range(len(text_str) - 1):  # 对于每个位置
            candi = candidate_bag[text_str[i]]
            # 若候选集只有自己
            if len(candi) == 1:
                continue
            else:
                eps=np.finfo(np.float64).eps
                sum_freq1=0.0
                sum_freq2 = 0.0
                max_freq = 0.0
                best_replace = replace_text[i] # 默认最好的是自己
                Ori_freq2 = 0.0
                Ori_freq1 = 0.0
                freq_list2 = []
                freq_list1=[]
                for c in candi:
                    freq1=0.0
                    freq2=0.0
                    two_gram = c + ' ' + text_str[i + 1]
                    if two_gram in tf_vocabulary.keys():
                        freq2=tf_vocabulary[two_gram]
                        if c==text_str[i]:
                            Ori_freq2 = tf_vocabulary[two_gram]+eps
                    if c in tf_vocabulary.keys():
                        freq1=tf_vocabulary[c]
                        if c==text_str[i]:
                            Ori_freq1 = tf_vocabulary[c]+eps
                    freq_list2.append(freq2+eps)
                    freq_list1.append(freq1+eps)

                sum_freq1=sum(freq_list1)
                sum_freq2=sum(freq_list2)
                lamda2=0.5     #0.5
                lamda1=0.5     #0.5

                Ori_freq2=Ori_freq2/sum_freq2
                Ori_freq1=Ori_freq1/sum_freq1
                Ori_freq=lamda2*Ori_freq2+lamda1*Ori_freq1

                for freq2, freq1,c in zip(freq_list2,freq_list1,candi):
                    freq2=freq2/sum_freq2
                    freq1=freq1/sum_freq1

                    freq=lamda2*freq2+lamda1*freq1
                    #sum_freq+=freq
                    #print(freq,end=' ')
                    if freq > max_freq:
                        max_freq = freq
                        best_replace = c

                r_seed = random.uniform(0, 1)

                #Ori_freq=0
                if (max_freq-Ori_freq) > r_seed:
                    replace_text[i] = best_replace

        select_sents.append(replace_text)

    return select_sents


def perturb_FGWS(args, orig_texts=None, texts=None, tf_vocabulary=None):
    """For FGWS"""

    select_sents = []
    if orig_texts is None:
        orig_texts=[None for i in range(len(texts))]
    for orig_text,text_str in zip(orig_texts,texts):
        # 获得候选集
        if ' '.join(text_str) in args.candidate_bags.keys(): # 若该文本见过（train and test）
            candidate_bag = args.candidate_bags[' '.join(text_str)]
        else:
            # 针对攻击代码
            if orig_text:  # 若orig_text不为空（仅在攻击中会赋值，此时所有的texts对应一个orig text）
                pos_tag = args.pos_tags[' '.join(orig_text)]  # 注意这里是对测试集做！
            else:
                pos_tag = pos_tagger.tag(text_str)  # 词性标注，耗时0.45s

            # str转id（同义词集是用id存的），有不存在于词汇表的，保留原str，之后分类需要
            text_ids = []
            for word_str in text_str:
                if word_str in args.full_dict.keys():
                    text_ids.append(args.full_dict[word_str])  # id
                else:
                    text_ids.append(word_str)  # str

            # 获得候选集
            candidate_bag = {}
            for j in range(len(text_ids)):  # 对于每个位置
                word = text_ids[j]
                pos = pos_tag[j][1]  # 当前词语词性
                neigbhours = [word]
                if isinstance(word, int) and pos in args.pos_list and word < len(args.word_candidate):
                    if pos.startswith('JJ'):
                        pos = 'adj'
                    elif pos.startswith('NN'):
                        pos = 'noun'
                    elif pos.startswith('RB'):
                        pos = 'adv'
                    elif pos.startswith('VB'):
                        pos = 'verb'
                    neigbhours.extend(args.word_candidate[word][pos])  # 候选集
                candidate_bag[text_str[j]] = [args.inv_full_dict[i] if isinstance(i, int) else i for i in neigbhours]  # id转为str

        replace_text = text_str.copy()
        for i in range(len(text_str) - 1):  # 对于每个位置
            candi = candidate_bag[text_str[i]]
            # 若候选集只有自己
            if len(candi) == 1:
                continue
            else:
                freq1=0
                bestc=replace_text[i]
                for c in candi:
                    if c in tf_vocabulary.keys():
                        if tf_vocabulary[c]>freq1:
                            freq1=tf_vocabulary[c]
                            bestc=c
                replace_text[i]=bestc
        select_sents.append(replace_text)
    return select_sents


def gen_sample_multiTexts(args, orig_texts=None, texts=None, sample_num=64, change_ratio = 1):
    """Random sample for text in train/test"""

    Finded_num=0
    all_sample_texts = []  # 返回值。list of list of str，包含输入每个数据的所有周围样本
    if orig_texts is None:
        orig_texts=[None for i in range(len(texts))]
    for orig_text,text_str in zip(orig_texts,texts):
        text_str = [word for word in text_str if word != '\x85']
        text_str = [word.replace('\x85', '') for word in text_str]
        if ' '.join(text_str) in args.candidate_bags.keys():  # seen data (from train or test dataset)
            # 获得候选集
            Finded_num+=1
            candidate_bag = args.candidate_bags[' '.join(text_str)]

            sample_texts=[]
            for ii in range(len(text_str)):
                word_str = text_str[ii]
                r_seed = np.random.rand(sample_num)
                n = choice(candidate_bag[word_str], size=sample_num, replace=True)
                n[np.where(r_seed>change_ratio)]=word_str
                sample_texts.append(n)
            sample_texts = np.array(sample_texts).T.tolist()

        else:  # unseen data
            # 针对攻击代码
            if orig_text: # 若orig_text不为空（仅在攻击中会赋值，此时所有的texts对应一个orig text）
                orig_text = [word for word in orig_text if word != '\x85']
                orig_text = [word.replace('\x85', '') for word in orig_text]
                pos_tag = args.pos_tags[' '.join(orig_text)]
            else:
                pos_tag = pos_tagger.tag(text_str)  # 词性标注，耗时0.45s

            # start_time = time.clock()
            # str转id（同义词集是用id存的），有不存在于词汇表的，保留原str，之后分类需要
            text_ids = []
            for word_str in text_str:
                if word_str in args.full_dict.keys():
                    text_ids.append(args.full_dict[word_str])  # id
                else:
                    text_ids.append(word_str)  # str

            # 获得候选集
            candidate_bag = {}
            for j in range(len(text_ids)):  # 对于每个位置
                word = text_ids[j]
                pos = pos_tag[j][1]  # 当前词语词性
                neigbhours = [word]
                if isinstance(word, int) and pos in args.pos_list and word < len(args.word_candidate):
                    if pos.startswith('JJ'):
                        pos = 'adj'
                    elif pos.startswith('NN'):
                        pos = 'noun'
                    elif pos.startswith('RB'):
                        pos = 'adv'
                    elif pos.startswith('VB'):
                        pos = 'verb'
                    neigbhours.extend(args.word_candidate[word][pos])  # 候选集
                candidate_bag[text_str[j]] = [args.inv_full_dict[i] if isinstance(i, int) else i for i in neigbhours] # id转为str
                # 可能一句话中一个词语出现多次

            sample_texts = []
            for ii in range(len(text_str)):
                word_str = text_str[ii]
                r_seed = np.random.rand(sample_num)
                n = choice(candidate_bag[word_str], size=sample_num, replace=True)
                n[np.where(r_seed>change_ratio)]=word_str
                sample_texts.append(n)
            sample_texts = np.array(sample_texts).T.tolist()

            # use_time = (time.clock() - start_time)
            # print("Time for the left: ", use_time)

        all_sample_texts.extend(sample_texts)
    #print("{:d}/{:d} texts are finded".format(Finded_num,len(texts)))
    return all_sample_texts


