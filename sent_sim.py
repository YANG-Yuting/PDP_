import json

import numpy as np
import criteria
import random
import tensorflow as tf
import tensorflow_hub as hub
import time
import torch
import pickle
from tqdm import tqdm

import os
from config import args
from torch.utils.data import DataLoader,SequentialSampler
from dataset import Dataset_BERT
from models import BERT
import torch.nn.functional as F
from attack_classification_hownet_top5 import USE

if __name__ == '__main__':
    args.rank = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'

    """Load data"""
    # textfooler
    # fr = open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/test_set/tf_%s_%s_adv_sym%s_%d.json' % (args.task, args.target_model, args.sym, args.split),'r')
    # data = fr.readlines()
    # adv_x,test_x,test_y = [],[],[]
    # for line in data:
    #     d = json.loads(line)
    #     adv_x.append(d['adv_texts'][0])
    #     test_x.append(d['text'])
    #     test_y.append(d['label'])

    # sempso
    fr = open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/sempso/test_set/org/sem_%s_%s_success.pkl' % (args.task, args.target_model), 'rb')
    input_list, true_label_list, output_list, success, change_list, num_change_list, success_time = pickle.load(fr)
    adv_x = output_list
    test_x = [d[0] for d in input_list]
    test_y = [d[1] for d in input_list]

    # pat
    # lpt = True if args.load_prompt_trained else False
    # we = True if args.word_emb else False
    # fr = open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/pat_%s_%s_%s_%s_%.2f_%d_%d_we%s_lpt%s_phead_p0.json' %
    #           (args.task, args.target_model, args.attack_level, args.mask_mode, args.mask_ratio, args.sample_size,
    #            args.topk, we, lpt), 'r')
    # data = json.load(fr)
    # adv_x, test_x, test_y = [], [], []
    # for idx, d in data.items():
    #     adv_x.append(d['adv_texts'][0].split())
    #     test_x.append(d['text'].split())
    #     test_y.append(d['label'])

    assert len(adv_x) == len(test_x)
    print('#Data:', len(adv_x))

    # sim_predictor = USE(args.USE_cache_path)
    # sent_sim = \
    # sim_predictor.semantic_sim(list(map(lambda x: ' '.join(x), adv_x)), list(map(lambda x: ' '.join(x), test_x)))[0]
    # print(np.mean(np.array(sent_sim)))
    # exit(0)


    dataset_test = Dataset_BERT(args)
    all_data_test, _ = dataset_test.transform_text(test_x, test_y)
    test_sampler = SequentialSampler(all_data_test)
    dataloader_test = DataLoader(all_data_test, sampler=test_sampler, batch_size=args.batch_size)

    dataset_adv = Dataset_BERT(args)
    all_data_adv, _ = dataset_adv.transform_text(adv_x, test_y)
    adv_sampler = SequentialSampler(all_data_adv)
    dataloader_adv = DataLoader(all_data_adv, sampler=adv_sampler, batch_size=args.batch_size)


    """Load BERT"""
    args.target_model_path = 'bert-base-uncased'
    model = BERT(args).cuda()
    model.eval()
    model.requires_grad_(False)

    """Get sentence emb"""
    # sent_emb_test = []
    # sent_emb_adv = []
    for idx, (*x, y) in enumerate(dataloader_test):
        sent_emb_test = model.sent_emb(x)
        # sent_emb_test.append(model.sent_emb(x).cpu().numpy().tolist())
    for idx, (*x, y) in enumerate(dataloader_adv):
        sent_emb_adv = model.sent_emb(x)
        # sent_emb_adv.append(model.sent_emb(x).cpu().numpy().tolist())
    # sent_emb_test = torch.tensor(sent_emb_test).cuda()
    # sent_emb_adv = torch.tensor(sent_emb_adv).cuda()

    # print(sent_emb_adv.shape)
    # print(sent_emb_test.shape)
    """Calculate sim"""

    sent_sim = F.cosine_similarity(sent_emb_test, sent_emb_adv, dim=1)


    # print(sent_sim)
    # print(sent_sim.shape)
    print(torch.mean(sent_sim))
