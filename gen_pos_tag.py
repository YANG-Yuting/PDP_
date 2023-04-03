# /usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import pickle
from tqdm import tqdm
import dataloader
from sklearn.model_selection import train_test_split
import os
import json
# from config import args

from nltk.tag import StanfordPOSTagger
jar = args.model_path + 'stanford-postagger-full-2020-11-17/stanford-postagger.jar'
model = args.model_path + 'stanford-postagger-full-2020-11-17/models/english-left3words-distsim.tagger'
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

def pos_tag(data, outpath):
    """data: 二维list，每行是一条文本，其中每个元素是词语str"""
    # 删除数据中的奇怪符号
    # clean_data = []
    # for text in data:
    #     new_text = []
    #     for word in text:
    #         word = word.replace('\n','').replace('_','')
    #         if word == '\x85':
    #             continue
    #         else:
    #             new_text.append(word.replace('\x85', ''))
    #     clean_data.append(new_text)

    # POS
    all_pos_tags = {}
    a = pos_tagger.tag_sents(data)
    for i in range(len(data)):
        text = ' '.join(data[i])
        all_pos_tags[text] = a[i]
        pos_text = [b[0] for b in a[i]]
        if text != ' '.join([b[0] for b in a[i]]):
            print(set(data[i]).difference(set(pos_text)))
            for j in range(len(data)):
                if data[i][j] != pos_text[j]:
                    print(j)
                    print(data[i][j])
                    print(pos_text[j])
            print(i)
            print(text)
            print(' '.join([b[0] for b in a[i]]))
            print('Error!!!')
            exit(0)

    # write to file
    f = open(outpath, 'wb')
    pickle.dump(all_pos_tags, f)


if __name__ == '__main__':
    task = 'imdb'
    train = True

    """1. prepare data"""
    if task == 'mr':
        # train_x, train_y = dataloader.read_corpus('data/adversary_training_corpora/mr/train.txt', clean=False,FAKE=False,shuffle=False)
        # test_x, test_y = dataloader.read_corpus('data/adversary_training_corpora/mr/test.txt', clean=False, FAKE=False,shuffle=False)  # 为了观察，暂时不shuffle

        with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/mr/dataset_20000.pkl', 'rb') as f:
            datasets = pickle.load(f)
        train_x = datasets.train_seqs2
        train_x = [[datasets.inv_full_dict[word] for word in text] for text in train_x]
        train_y = datasets.train_y
        test_x = datasets.test_seqs2
        test_x = [[datasets.inv_full_dict[word] for word in text] for text in test_x]
        test_y = datasets.test_y
    elif task == 'imdb':
        # train_x, train_y = dataloader.read_corpus(os.path.join(data_path + 'imdb', 'train_tok.csv'), clean=False, FAKE=False, shuffle=True)
        # test_x, test_y = dataloader.read_corpusus(os.path.join(data_path + 'imdb', 'test_tok.csv'), clean=False, FAKE=False, shuffle=True)
        with open(args.data_path + 'imdb/dataset_50000_has_punctuation.pkl', 'rb') as f:
            datasets = pickle.load(f)
        train_x = datasets.train_seqs2
        train_x = [[datasets.inv_full_dict[word] for word in text] for text in train_x]
        train_y = datasets.train_y
        test_x = datasets.test_seqs2
        test_x = [[datasets.inv_full_dict[word] for word in text] for text in test_x]
        test_y = datasets.test_y
    elif task == 'fake':
        train_x, train_y = dataloader.read_corpus(data_path + '{}/train_tok.csv'.format(task), clean=False, FAKE=True, shuffle=True)
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1, random_state=1)  # tiz: 从训练集中获得测试集试试 --> 正常了

    if train:
        out_file = args.data_path + '%s/pos_tags_train_has_punctuation.pkl' % task
        data = train_x
    else:
        out_file = args.data_path + '%s/pos_tags_test_has_punctuation.pkl' % task
        data = test_x # [:200]

    """为了续写的新数据"""
    # data_path = '/pub/data/huangpei/TextFooler/prompt_results/'
    # data_path1 = data_path + 'gpt12_beamSearch_%s_comma.json' % task
    # with open(data_path1, 'r') as fr:
    #     gpt_results = json.load(fr)
    # data = []
    # for idx in gpt_results.keys():  # 对于每条测试样例
    #     candi_texts = gpt_results[idx]['candi_texts']  # 5条
    #     data.extend([can.split(' ') for can in candi_texts])
    # out_file = data_path + 'pos_gpt12_beamSearch_%s_comma.pkl' % task

    """2. POS"""
    pos_tag(data, out_file)

