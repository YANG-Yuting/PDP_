import json
import os
import sys
import argparse

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import args
from BERT.tokenization import BertTokenizer
from train_model import load_test_data, load_train_data, eval_model
from models import LSTM, ESIM, BERT, BERT_snli, ROBERTA
from dataset import Dataset_LSTM_ascc, Dataset_LSTM_snli_ascc, Dataset_BERT, Dataset_ROBERTA
from torch.utils.data import Dataset, DataLoader,SequentialSampler
from train_model import eval_model
from multi_train_utils.distributed_utils import init_distributed_mode, dist, is_main_process

if args.task == 'snli':
    triggers = [[',', 'implying', 'that', ':'], [',', 'implying', 'that', ':'], [',', 'is', 'contradictory', 'with', ':']]
elif args.task in ['mr', 'imdb']:
    triggers = [['it', 'is', 'a', 'good', 'movie','.'], ['it', 'is', 'a', 'bad', 'movie','.']]
# triggers = [[],[]]
# prompt_templates = [[',', 'implying', 'that', ':'], [',', 'implying', 'that', ':'], [',', 'is', 'contradictory', 'with', ':']]


class infoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(infoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeds_org, embeds_pos, embeds_neg):
        """
        一个batch内，输入大小均为 [batch_size, hidden_size]
        """
        # sim_pos = torch.cosine_similarity(embeds_org, embeds_pos, dim=1).unsqueeze(0) # (1,b_size) 与正例的余弦相似度
        # sim_neg = torch.cosine_similarity(embeds_org, embeds_neg, dim=1).unsqueeze(0) # (1,b_size) 与负例的余弦相似度
        sim_pos = torch.norm((embeds_org - embeds_pos),p=2,dim=1).unsqueeze(0)
        sim_neg = torch.norm((embeds_org - embeds_neg),p=2,dim=1).unsqueeze(0)

        sim_all = torch.cat((sim_pos, sim_neg),dim=0).T # (b_size,2) 第一列是与正例的相似度，第二列是与负例
        sim_all /= self.temperature
        labels =  torch.zeros(embeds_org.shape[0], dtype=torch.long).cuda() # 全0，第0列为正例所在列 (1, b_size)

        CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        loss = CrossEntropyLoss(sim_all, labels)

        return loss

class TripleLoss(nn.Module):
    def __init__(self, epsilon=30):
        super(TripleLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, embeds_org, embeds_pos, embeds_neg):
        """
        一个batch内，输入大小均为 [batch_size, hidden_size]
        """

        #dist_pos = torch.sqrt(torch.sum((embeds_org - embeds_pos)**2, dim=1))
        #dist_neg = torch.sqrt(torch.sum((embeds_org - embeds_neg)**2, dim=1))
        dist_pos=torch.norm((embeds_org - embeds_pos),p=2,dim=1)
        dist_neg=torch.norm((embeds_org - embeds_neg),p=2,dim=1)
        #dist_pos = torch.dist(embeds_org, embeds_pos, p=2,dim=1)
        #dist_neg = torch.dist(embeds_org, embeds_neg, p=2,dim=1)
        #print(dist_pos)
        #print(dist_neg)
        loss = torch.max(torch.tensor([0.]*embeds_org.shape[0]).cuda(), dist_pos - dist_neg + self.epsilon)
        #loss = torch.max(torch.tensor([0.] * embeds_org.shape[0]).cuda(), dist_pos - 10.0)
        #print(loss)
        #exit(0)
        #loss = torch.sum(dist_pos)
        loss = torch.mean(loss) # 一个batch内求均值

        return loss

def build_data(train_x, train_y):
    """
    获得mr和imdb训练集上的prompt前后句子对
    返回值：train_data: [[text, prompt_text, trigger], [...]]
    对于mr/imdb：期望返回值的prompt_text是trigger masked_text
    对于snli：train_x的每个元素是s1 sep_token s2，期望返回值的prompt_text是s1 trigger masked_s2
    """
    train_data = []
    train_labels = []

    # 随机打乱原数据
    data = list(zip(train_x, train_y))
    random.shuffle(data)
    train_x, train_y = zip(*data)

    for text, label in zip(train_x, train_y):
        # 避开重要位置
        """确定可mask位置"""
        # candi_ps = []
        # for ii, (word, ps) in enumerate(pos_tags[' '.join(text)]):
        #     if ps.startswith('JJ'):  # 避开形容词
        #         continue
        #     else:
        #         candi_ps.append(ii)
        """进行mask"""
        if args.task in ['mr', 'imdb']:
            masked_texts = []
            for jj in range(len(text)):
                word = text[jj]
                # if jj in candi_ps:
                #     r_seed = np.random.rand(args.sample_size)
                #     n = ['[MASK]' if rs < args.mask_ratio else word for rs in r_seed]
                # else:
                #     n = [word] * args.sample_size
                # masked_texts.append(n)
                r_seed = np.random.rand(args.sample_size)
                n = [args.mask_token if rs < args.mask_ratio else word for rs in r_seed]
                masked_texts.append(n)
            masked_texts = np.array(masked_texts).T.tolist()
            for mt in masked_texts:
                trigger = triggers[label]
                prompt = trigger + mt  # 在mask文本开头加入trigger
                train_data.append(text)
                train_data.append(prompt)
                train_data.append(trigger)
                train_labels.extend([label, label, abs(1-label)])
        elif args.task == 'snli':
            s1, s2 = ' '.join(text).split(' '+args.sep_token+' ')
            s1 = s1.split()
            s2 = s2.split()
            masked_texts = []
            for jj in range(len(s2)):
                word = s2[jj]
                r_seed = np.random.rand(args.sample_size)
                n = [args.mask_token if rs < args.mask_ratio else word for rs in r_seed]
                masked_texts.append(n)
            masked_texts = np.array(masked_texts).T.tolist()
            for mt in masked_texts:
                trigger = triggers[label]
                prompt = s1 + trigger + mt
                train_data.append(text)
                train_data.append(prompt)
                train_data.append(trigger)
                train_labels.extend([label, label, abs(1 - label)])

    assert len(train_data) % 3 == 0

    return train_data, train_labels

# def train_model(epoch, model, optimizer,train_data, train_labels, test_x,test_y, save_path,tokenizer):
#     model.train()
#
#     if args.group_mode == 'sent':
#         dataloader, _ = model.dataset.transform_text(train_data, labels=train_labels, batch_size=args.batch_size)
#     else:
#         dataloader, _ = model.dataset.transform_text_en(train_data, labels=train_labels, batch_size=args.batch_size)
#
#     niter=0
#     criterion_cls = nn.CrossEntropyLoss()
#     #criterion_grp = torch.nn.MSELoss(reduction='mean')  # 均方误差损失
#     # criterion_grp = infoNCELoss() # 对比学习损失
#     criterion_grp = TripleLoss() # 对比学习损失
#
#     for e in range(epoch):
#         best_acc=0
#         save_path_=None
#         start_time = time.clock()
#         sum_loss = 0
#         sum_loss_cls = 0
#         sum_loss_grp = 0
#         # model.zero_grad()
#         print('Batch num:', len(dataloader))
#         for input_ids, input_mask, segment_ids, labels in dataloader:
#
#             niter += 1
#             input_ids = input_ids.cuda()
#             input_mask = input_mask.cuda()
#             segment_ids = segment_ids.cuda()
#
#             layer_output, text_emb = model.model.bert(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)  # [batch_size, hidden_size]，每个元素是该数据的文本embedding表示
#
#             """loss"""
#             # 1. 对于分类任务
#             pooled_output = model.model.dropout(text_emb)
#             logits = model.model.classifier(pooled_output)
#             loss_cls = criterion_cls(logits[::3, :], labels[::3].cuda()) # 只计算原样例的loss
#             #loss_cls = criterion_cls(logits, labels.cuda())  # 都计算
#             # 2. 对于聚合任务
#             if args.group_mode == 'sent':
#                 text_emb_pos = text_emb[::3, :]   # 正例：原样例
#                 text_emb_org = text_emb[1::3, :]  # 锚：prompt样例
#                 text_emb_neg = text_emb[2::3, :]  # 负例：trigger
#                 #loss_grp = criterion_grp(text_emb_org, text_emb_pos) # MSE loss
#                 loss_grp = criterion_grp(text_emb_org, text_emb_pos, text_emb_neg) #
#             elif args.group_mode == 'word':
#                 text_emb = layer_output[-1] # [batch_size, seq_len, hidden_size]，最后一层对于各个位置的embedding输出
#                 loss_grp = 0.
#                 for i in range(text_emb.shape[0]):
#                     if i%3 == 0:
#                         # 只保留input_mask为1，即非padding位置(input_mask中1的综合即为有效位置个数)；去除首部的CLS
#                         text_emb_a = text_emb[i][1:int(torch.sum(input_mask[i]).cpu()),:]
#                     elif i%3 == 1:
#                         # 相比上面，多去除6个prompt位置
#                         text_emb_b = text_emb[i][7:int(torch.sum(input_mask[i]).cpu()),:]
#                         loss_grp = loss_grp + criterion_grp(text_emb_a, text_emb_b)
#             # 3. 调和两个loss
#             loss = args.alpha * loss_cls + (1-args.alpha) * loss_grp
#
#             optimizer.zero_grad()
#             loss = loss.mean() # 多GPU并行时，需要先求平均
#             loss.backward()
#             optimizer.step()
#             sum_loss += float(loss.item())
#             sum_loss_cls += float(loss_cls.item())
#             sum_loss_grp += float(loss_grp.item())
#
#         if torch.cuda.device_count() > 1:
#             test_acc = model.module.eval_model(test_x, test_y)
#         else:
#             test_acc = model.eval_model(test_x, test_y)
#         # train_acc =  model.eval_model(train_data[::2], train_labels[::2])
#         use_time = (time.clock() - start_time) # time of a epoch
#         sys.stdout.write("Time={} Epoch={} iter={} lr={:.6f} train_loss={:.4f} cls_loss={:.4f} group_loss={:.4f} test_acc={:.4f}\n"
#                          .format(use_time, e, niter,optimizer.param_groups[0]['lr'], sum_loss, sum_loss_cls, sum_loss_grp, test_acc))
#
#         if save_path:
#             if e % 5 == 0:
#                 save_path_ = save_path + '_ep%d' % e
#                 if not os.path.exists(save_path_):
#                     os.makedirs(save_path_)
#                 if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#                     torch.save(model.module.model.state_dict(), save_path_ + '/pytorch_model.bin')
#                     model.module.model.config.to_json_file(save_path_ + '/bert_config.json')
#                 else:
#                     torch.save(model.model.state_dict(), save_path_ + '/pytorch_model.bin')
#                     model.model.config.to_json_file(save_path_ + '/bert_config.json')
#                 tokenizer.save_vocabulary(save_path_)
#             else:
#                 if best_acc<test_acc:
#                     best_acc=test_acc
#                     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#                         torch.save(model.module.model.state_dict(), save_path_ + '/pytorch_model.bin')
#                         model.module.model.config.to_json_file(save_path_ + '/bert_config.json')
#                     else:
#                         torch.save(model.model.state_dict(), save_path_ + '/pytorch_model.bin')
#                         model.model.config.to_json_file(save_path_ + '/bert_config.json')
#                     tokenizer.save_vocabulary(save_path_)
#
#         # if e %3 == 0:
#         #     if save_path:
#         #         # if e in [56,57,58,71,72,73,80,90,100,110,120,130,150,180,200]: # word
#         #         # if e %10 == 0:  # word
#         #         save_path_ = save_path + '_ep%d' % e
#         #         if not os.path.exists(save_path_):
#         #             os.makedirs(save_path_)
#         # if test_acc > best_acc or e %3 == 0:
#         #     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         #         torch.save(model.module.model.state_dict(), save_path_+'/pytorch_model.bin')
#         #         model.module.model.config.to_json_file(save_path_ + '/bert_config.json')
#         #     else:
#         #         torch.save(model.model.state_dict(), save_path_+'/pytorch_model.bin')
#         #         model.model.config.to_json_file(save_path_ + '/bert_config.json')
#         #     tokenizer.save_vocabulary(save_path_)
#         lr_decay = 1
#         if lr_decay > 0:
#             optimizer.param_groups[0]['lr'] *= lr_decay
#     return None

def train_model(e, model, optimizer,train_x, train_y, test_x,test_y, dataset, tokenizer):

    train_data, train_labels = build_data(train_x, train_y)  # 包含随机mask、打乱部分

    model.train()
    start_time = time.clock()

    all_data, _ = dataset.transform_text(train_data, labels=train_labels)
    if torch.cuda.device_count() > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(all_data, num_replicas=torch.cuda.device_count(), rank=args.rank)
        nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
        dataloader = DataLoader(all_data, sampler=sampler, pin_memory=True, num_workers=nw, batch_size=args.batch_size)  # batch_
    else:
        sampler = SequentialSampler(all_data)
        dataloader = DataLoader(all_data, sampler=sampler, batch_size=args.batch_size)

    niter=0
    criterion_cls = nn.CrossEntropyLoss()
    criterion_grp = torch.nn.MSELoss(reduction='mean')  # 均方误差损失
    #criterion_grp = infoNCELoss() # 对比学习损失
    # criterion_grp = TripleLoss() # 对比学习损失

    sum_loss = 0
    sum_loss_cls = 0
    sum_loss_grp = 0
    save_path_=None
    # model.zero_grad()
    # print('Batch num:', len(dataloader))
    for input_ids, input_mask, segment_ids, labels in dataloader:
        optimizer.zero_grad()
        niter += 1
        input_ids = input_ids.cuda(args.rank)
        input_mask = input_mask.cuda(args.rank)
        segment_ids = segment_ids.cuda(args.rank)

        if torch.cuda.device_count() > 1:
            _, pooled_output = model.module.encoder(input_ids, input_mask, return_dict=False)  # [batch_size, seq_len, hidden], [batch_size, hidden]
            output = model.module.drop(pooled_output)
            logits = model.module.classifier(output)
        else:
            if args.target_model == 'roberta':
                _, pooled_output = model.encoder(input_ids, input_mask, return_dict=False) # [batch_size, seq_len, hidden], [batch_size, hidden]
                output = model.drop(pooled_output)
                logits = model.classifier(output)
                probs = nn.functional.softmax(logits, dim=-1)
            elif args.target_model == 'bert':
                if args.task == 'snli':
                    probs = model([input_ids, input_mask, segment_ids])
                else:
                    _, pooled_output = model.model.bert(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)  # [batch_size, hidden_size]，每个元素是该数据的文本embedding表示
                    pooled_output = model.model.dropout(pooled_output)
                    logits = model.model.classifier(pooled_output)
                    probs = nn.functional.softmax(logits, dim=-1)

        loss_cls = criterion_cls(probs[::3, :], labels[::3].cuda(args.rank))  # 只计算原样例的loss
        loss_grp = criterion_cls(probs[1::3, :], labels[1::3].cuda(args.rank))
        # loss_cls = loss_cls + loss_cls_prompt
        # text_emb_pos = pooled_output[::3, :]  # 正例：原样例
        # text_emb_org = pooled_output[1::3, :]  # 锚：prompt样例
        # text_emb_neg = pooled_output[2::3, :]  # 负例：trigger
        # loss_grp = criterion_grp(text_emb_org, text_emb_pos)  # MSE loss
        # loss_grp = 0.
        # for i in range(hidden_state.shape[0]):
            # if i % 3 == 0:
            #     # 只保留input_mask为1，即非padding位置(input_mask中1的综合即为有效位置个数)；去除首部的CLS
            #     text_emb_a = hidden_state[i][1:int(torch.sum(input_mask[i]).cpu()), :]
            # elif i % 3 == 1:
            #     # 相比上面，多去除prompt位置
            #     prompt_len = len(triggers[labels[i]])
            #     text_emb_b = hidden_state[i][1+prompt_len:int(torch.sum(input_mask[i]).cpu()), :]
            #     print(text_emb_a.shape)
            #     print(text_emb_b.shape)
            #     loss_grp = loss_grp + criterion_grp(text_emb_a, text_emb_b)
            # sequence_output = model.encoder(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0] # RobertaModel


        # if torch.cuda.device_count() > 1:
        #     layer_output, text_emb = model.module.model.bert(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)  # [batch_size, hidden_size]，每个元素是该数据的文本embedding表示
        # else:
        #     layer_output, text_emb = model.model.bert(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)  # [batch_size, hidden_size]，每个元素是该数据的文本embedding表示

        """loss"""
        """1. 对于分类任务"""
        # if torch.cuda.device_count() > 1:
        #     pooled_output = model.module.model.dropout(text_emb)
        #     logits = model.module.model.classifier(pooled_output)
        # else:
        #     pooled_output = model.model.dropout(text_emb)
        #     logits = model.model.classifier(pooled_output)


        # loss_cls = criterion_cls(logits[::3, :], labels[::3].cuda()) # 只计算原样例的loss
        # # loss_cls = criterion_cls(logits[::3, :], labels[::3].cuda())+criterion_cls(logits[2, :], labels[2,:].cuda())  # 只计算原样例的loss
        # #loss_cls = criterion_cls(logits, labels.cuda())  # 都计算
        #
        # #loss_cls = criterion_cls(logits[::3, :], labels[::3].cuda())+(criterion_cls(logits[2::3, :], labels[2::3].cuda())/text_emb.shape[0])
        #
        #
        # # 2. 对于聚合任务
        # if args.group_mode == 'sent':
        #     text_emb_pos = text_emb[::3, :]   # 正例：原样例
        #     text_emb_org = text_emb[1::3, :]  # 锚：prompt样例
        #     text_emb_neg = text_emb[2::3, :]  # 负例：trigger
        #     loss_grp = criterion_grp(text_emb_org, text_emb_pos) # MSE loss
        #     #loss_grp = criterion_grp(text_emb_org, text_emb_pos, text_emb_neg) #
        # elif args.group_mode == 'word':
        #     text_emb = layer_output[-1] # [batch_size, seq_len, hidden_size]，最后一层对于各个位置的embedding输出
        #     loss_grp = 0.
        #     for i in range(text_emb.shape[0]):
        #         if i%3 == 0:
        #             # 只保留input_mask为1，即非padding位置(input_mask中1的综合即为有效位置个数)；去除首部的CLS
        #             text_emb_a = text_emb[i][1:int(torch.sum(input_mask[i]).cpu()),:]
        #         elif i%3 == 1:
        #             # 相比上面，多去除6个prompt位置
        #             text_emb_b = text_emb[i][7:int(torch.sum(input_mask[i]).cpu()),:]
        #             loss_grp = loss_grp + criterion_grp(text_emb_a, text_emb_b)
        # 3. 调和两个loss
        loss = args.alpha * loss_cls + (1-args.alpha) * loss_grp

        loss = loss.mean()  # 多GPU并行时，需要先求平均
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        sum_loss += float(loss.item())
        sum_loss_cls += float(loss_cls.item())
        sum_loss_grp += float(loss_grp.item())

    test_acc = eval_model(model, test_x, test_y)
    use_time = (time.clock() - start_time) # time of a epoch
    if is_main_process():
        print("Time={} Epoch={} iter={} lr={:.6f} train_loss={:.4f} cls_loss={:.4f} group_loss={:.4f} test_acc={:.4f}"
                     .format(use_time, e, niter,optimizer.param_groups[0]['lr'], sum_loss, sum_loss_cls, sum_loss_grp, test_acc))


    if args.save_path:
        if not os.path.exists(args.save_path + '_ep%d' % e):
            os.makedirs(args.save_path+'_ep%d' % e)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), args.save_path+'_ep%d' % e + '/pytorch_model.bin')
        else:
            torch.save(model.state_dict(), args.save_path+'_ep%d' % e + '/pytorch_model.bin')
            if args.task != 'snli' and args.target_model == 'bert':
                tokenizer.save_vocabulary(args.save_path+'_ep%d' % e)
                model.model.config.to_json_file(args.save_path+'_ep%d' % e + '/bert_config.json')  # bert_

    # if e %3 == 0:
    #     if save_path:
    #         # if e in [56,57,58,71,72,73,80,90,100,110,120,130,150,180,200]: # word
    #         # if e %10 == 0:  # word
    #         save_path_ = save_path + '_ep%d' % e
    #         if not os.path.exists(save_path_):
    #             os.makedirs(save_path_)
    # if test_acc > best_acc or e %3 == 0:
    #     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #         torch.save(model.module.model.state_dict(), save_path_+'/pytorch_model.bin')
    #         model.module.model.config.to_json_file(save_path_ + '/bert_config.json')
    #     else:
    #         torch.save(model.model.state_dict(), save_path_+'/pytorch_model.bin')
    #         model.model.config.to_json_file(save_path_ + '/bert_config.json')
    #     tokenizer.save_vocabulary(save_path_)
    lr_decay = 0.99
    if lr_decay > 0 and optimizer.param_groups[0]['lr']>0.000005:
        optimizer.param_groups[0]['lr'] *= lr_decay
    return None

def main(args):
    """Setting for parallel training"""
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    # args.rank = 0
    init_distributed_mode(args=args)
    print(args.rank)
    torch.cuda.set_device(args.rank)
    device = torch.device('cuda', args.rank)

    print("Building Model...")
    if args.target_model == 'lstm':
        if args.task == 'snli':
            model = ESIM(args).cuda(args.rank)
            """Load from trained model with different para name"""
            checkpoint = torch.load(args.target_model_path, map_location=args.device)
            checkpoint_new = {}
            for name, value in checkpoint.items():
                name = name.replace('enc_lstm.', '')
                checkpoint_new[name] = value
            model.load_state_dict(checkpoint_new)
        else:
            model = LSTM(args).cuda()
            checkpoint = torch.load(args.target_model_path, map_location=args.device)
            model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        if args.task == 'snli':
            model = BERT_snli(args).cuda(args.rank)
            checkpoint = torch.load(args.target_model_path, map_location=args.device)
            model.load_state_dict(checkpoint)
            dataset = Dataset_BERT(args)
            tokenizer = None
        else:
            tokenizer = BertTokenizer.from_pretrained(args.target_model_path, do_lower_case=True)  # 用来保存模型
            model = BERT(args).cuda(args.rank)  # args.rank
            checkpoint = torch.load(args.target_model_path + '/pytorch_model.bin', map_location=args.device)
            model.model.load_state_dict(checkpoint)
            dataset = Dataset_BERT(args)
    elif args.target_model == 'roberta':
        model = ROBERTA(args).cuda(args.rank)
        checkpoint = torch.load(args.target_model_path+'/pytorch_model.bin')
        model.load_state_dict(checkpoint)
        args.pad_token_id = model.encoder.config.pad_token_id
        dataset = Dataset_ROBERTA(args)
        tokenizer = None

    # model = NLI_infer_BERT(args, args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    """Build data..."""
    """Load data"""
    test_x, test_y = load_test_data()
    train_x, train_y = load_train_data()
    if is_main_process():
        test_acc = eval_model(model, test_x, test_y)
        print('Acc for test set is: {:.2%}'.format(test_acc))

    print('Train...')
    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=args.lr)
    epoch = 30
    best_test = 0
    for e in range(epoch):
        if args.task == 'snli':
            data = list(zip(train_x, train_y))
            random.shuffle(data)
            train_x, train_y = zip(*data)
            train_x, train_y = train_x[:10000], train_y[:10000]
        train_model(e, best_test, model, optimizer, train_x, train_y, test_x, test_y, dataset, tokenizer)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '5'  # args.gpu_id
    main(args)
