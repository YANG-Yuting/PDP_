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
from train_model import load_test_data, load_train_data, eval_model
from models import *
from dataset import Dataset_LSTM_ascc, Dataset_LSTM_snli_ascc, Dataset_BERT, Dataset_ROBERTA
from BERT.tokenization import BertTokenizer
from gen_pos_tag import pos_tagger

if __name__ == "__main__":
    args.rank = 0
    torch.cuda.set_device(args.rank)
    device = torch.device('cuda', args.rank)

    """Load data"""
    if args.train_set:
        train_x, train_y = load_train_data()
        c = list(zip(train_x, train_y))  # Randomly sample 25% train data
        train_x, train_y = zip(*c)
        train_x, train_y = train_x[:int(0.25*len(train_x))], train_y[:int(0.25*len(train_y))]

        num_split = 7
        aa = int(len(train_y) / num_split)
        print(aa*(args.split-1), aa*args.split)
        data_x, data_y = train_x[aa*(args.split-1):aa*args.split], train_y[aa*(args.split-1):aa*args.split]
    else:
        test_x, test_y = load_test_data()
        test_x, test_y = test_x[:1], test_y[:1]  # Select 200 test data for attacking
        data_x, data_y = test_x, test_y

    """Load model"""
    if args.target_model == 'bert':
        if args.task == 'snli':
            model = BERT_snli(args).cuda(args.rank)
            checkpoint = torch.load(args.target_model_path +'/pytorch_model.bin', map_location=device)
            model.load_state_dict(checkpoint)
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            if args.kind == 'Ensemble':
                if 'comp' in args.target_model_path or 'aw' in args.target_model_path :
                    print('Load comp model')
                    model = EnsembleBERT_comp(args).cuda(args.rank)
                    checkpoint = torch.load(args.target_model_path + '/pytorch_model.bin')
                    model.load_state_dict(checkpoint)
                else:
                    model = EnsembleBERT(args).cuda(args.rank)
            else:
                model = BERT(args).cuda(args.rank)
    elif args.target_model == 'roberta':
        model = ROBERTA(args).cuda(args.rank)
        checkpoint = torch.load(args.target_model_path+'/pytorch_model.bin', map_location=device)
        model.load_state_dict(checkpoint)
        args.pad_token_id = model.encoder.config.pad_token_id

    model.eval()

    all_data, _ = model.dataset.transform_text(data_x, labels=data_y)
    sampler = SequentialSampler(all_data)
    texts_dataloader = DataLoader(all_data, sampler=sampler, batch_size=args.batch_size)

    with torch.no_grad():
        for idx, (*x, y) in enumerate(texts_dataloader):
            input_ids, input_mask, segment_ids = x[:3]
            # print(input_ids)
            # print(input_ids.numpy().tolist()[0])
            input_text = model.dataset.tokenizer.convert_ids_to_tokens(input_ids.numpy().tolist()[0])
            print(len(input_text), input_text)
            input_ids = input_ids.cuda(args.rank)
            input_mask = input_mask.cuda(args.rank)
            segment_ids = segment_ids.cuda(args.rank)
            # 对于每个子模型
            for i in range(args.num_models):
                if 'comp' in args.target_model_path or 'aw' in args.target_model_path:
                    # 压缩集成
                    model_no = torch.tensor([i], dtype=torch.float).cuda(args.rank)
                    params_aux = model.aux(model_no)
                    s_index = 0  # 在辅助网络输出中的索引
                    param_org_bert = {}
                    for name, param in model.bert.named_parameters():
                        # print('name', name)
                        param_org = param.clone()  # param_org与param不共享内存
                        param_org_bert[name] = param_org
                        if args.modify_attentions:  # 只修改第0层self-attention部分
                            if 'encoder.layer.0.attention' not in name:
                                continue
                        param_num = param.numel()  # 参数个数
                        param.data.add_(args.aux_weight * params_aux[s_index: s_index + param_num].reshape(
                            param.shape))  # bert的参数随之改变
                        s_index += param_num
                    all_attentions, logits = model.bert(input_ids, token_type_ids=segment_ids,
                                                        attention_mask=input_mask)
                    for name, param in model.bert.named_parameters():
                        param.data = param_org_bert[name]
                        param.data.requires_grad = False
                else:
                    # 基本集成
                    all_attentions, logits = model.models[i](input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                # 12 torch.Size([1, 12, 256, 256]) batch_size, num_head, input_len, input_len
                print(logits)
                print(len(all_attentions), all_attentions[0].size())
                # 观察第0层每个位置对于第1层中所有词语的权重的均值
                all_attentions = all_attentions[0]  # 观察第0层 [1, 12, 256, 256]
                all_attentions = all_attentions[:, 0, :, :]  # 观察第0个head [1, 256, 256]
                # temp = all_attentions[:, 0, :]  # sum=1 表示第0个单词的embedding计算时，所有单词的权重
                out = torch.sum(all_attentions, dim=1)
                # out_ = torch.sum(all_attentions, dim=2)  # 全为1
                print(i, out)
                values, indices = torch.topk(out, k=20, dim=-1)
                print(values, indices)
                topk_ids = input_ids[0].index_select(dim=0, index=indices[0])
                print(model.dataset.tokenizer.convert_ids_to_tokens(topk_ids.cpu().numpy().tolist()))

            exit(0)





