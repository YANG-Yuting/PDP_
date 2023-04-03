
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

import os
#import nltk
import re
from collections import Counter


import data_utils
import glove_utils



if __name__ == '__main__':
    task = 'imdb'
    PATH = '/home/huangpei/TextFooler/data/' + task
    sizes = {'imdb': 50000, 'mr': 20000, 'fake': 50000}
    MAX_VOCAB_SIZE = sizes[task]
    GLOVE_PATH = '/home/huangpei/TextFooler/data/glove.6B/glove.6B.200d.txt'

    dataset = data_utils.myDataset(task=task, path=PATH, max_vocab_size=MAX_VOCAB_SIZE)
    # print(dataset.test_y[:100])

    # save the dataset
    # tiz: 2022.08.05 去除奇怪符号、有标点

    # tiz: 2021.12.22
    # 有符号
    with open(('//home/huangpei/TextFooler/data/%s/dataset_%d_has_punctuation_.pkl' % (task, MAX_VOCAB_SIZE)), 'wb') as f:
        pickle.dump(dataset, f)
    # 无符号，且与新产生的有符号数据顺序一致
    # with open(('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/dataset_%d_new.pkl' % (task, MAX_VOCAB_SIZE)), 'wb') as f:
    #     pickle.dump(dataset, f)

    # # create the glove embeddings matrix (used by the classification model)
    # glove_model = glove_utils.loadGloveModel(GLOVE_PATH)
    # glove_embeddings, _ = glove_utils.create_embeddings_matrix(glove_model, dataset.dict, dataset.full_dict)
    # # save the glove_embeddings matrix
    # np.save('data/adversary_training_corpora/%s/embeddings_glove_%d.pkl' % (task, MAX_VOCAB_SIZE), glove_embeddings)
    # print('All done')