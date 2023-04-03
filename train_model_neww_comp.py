from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
import os
import torch.optim as optim
import sys
import random
import json

from models import LSTM, ESIM, BERT, BERT_snli, ROBERTA, EnsembleBERT, EnsembleBERT_comp
from dataset import *
from config import args
from BERT.tokenization import BertTokenizer
import time
import torch.nn.functional as F


# parallel training
from multi_train_utils.distributed_utils import init_distributed_mode, dist, is_main_process
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import autograd

def load_test_data():
    """Load data for test"""
    if args.task == 'snli':
        """ concate s1, s2 with '[SEP]/</s>' """
        test_x = []  # [[s1 '[SEP]' s2], [s1 '[SEP]' s2],...]
        for s1, s2 in zip(args.test_s1, args.test_s2):
            s = s1 + [args.sep_token] + s2
            test_x.append(s)
        test_y = args.test_labels
    else:
        test_x = args.datasets.test_seqs2
        test_x = [[args.inv_full_dict[w] for w in x] for x in test_x]
        test_y = args.datasets.test_y

    return test_x, test_y


def load_train_data():
    """Load data for training"""

    """Original train data"""
    if args.task == 'snli':
        """ concate s1, s2 with '[SEP]' """
        train_x = []  # [[s1 '[SEP]' s2], [s1 '[SEP]' s2],...]
        for s1, s2 in zip(args.train_s1, args.train_s2):
            s = s1 + [args.sep_token] + s2
            train_x.append(s)
        train_y = args.train_labels
    else:
        train_x = args.datasets.train_seqs2
        train_x = [[args.inv_full_dict[w] for w in x] for x in train_x]
        train_y = list(args.datasets.train_y)

    """Adversarial augmentation"""
    """Load adversarial examples written in the file (generated by semPSO for top 10% train data)"""
    # # with open(args.data_path + '/%s/AD_dpso_sem_%s.pkl' % (args.task, args.target_model), 'rb') as f:
    # #     input_list, test_list, adv_y, adv_x, success, change_list, target_list = pickle.load(f)
    # # adv_x = [[args.inv_full_dict[w] for w in x] for x in adv_x]  # 网络输入是词语
    # """Load adversarial examples written in the file (generated by textfooler for top 25% train data)"""
    # train_x, train_y = [], []
    # adv_x, adv_y = [], []
    # # num_changed = []
    # adv_files = os.listdir(args.adv_path)
    # for file in adv_files:
    #     with open(args.adv_path + '/' + file, 'rb') as f:
    #         for line in f.readlines():
    #             js = json.loads(line.strip(), encoding='utf-8')
    #             adv_x.append(js['adv_texts'][0].split())
    #             adv_y.append(js['label'])
    #             # num_changed.append(js['num_changed'])
    # # num_changed = np.array(num_changed).mean()
    # # print('Average num changed:', num_changed)
    # # exit(0)
    # if is_main_process():
    #     print('#Adv examples: ', len(adv_y))
    # train_x.extend(adv_x)
    # train_y.extend(adv_y)
    # if is_main_process():
    #     print('#Final train data: ', len(train_y))

    # imdb打乱训练更好
    if args.task == 'imdb':
        c = list(zip(train_x, train_y))
        random.seed(15)
        random.shuffle(c)
        train_x, train_y = zip(*c)
    return train_x, train_y


def eval_model(model, inputs_x, inputs_y):  # inputs_x is list of list with word
    model.eval()
    correct = 0.0
    if torch.cuda.device_count() > 1:
        predictor = model.module.text_pred()
    else:
        predictor = model.text_pred()
    # data_size = len(inputs_y)
    with torch.no_grad():
        outputs = predictor(inputs_x, inputs_x)
        pred = torch.argmax(outputs, dim=1)
        data_size = pred.shape[0]
        correct += torch.sum(torch.eq(pred, torch.LongTensor(inputs_y[:data_size]).cuda(args.rank)))
        acc = (correct.cpu().numpy())/float(data_size)
    return acc

# lamda = 1
# log_det_lamda = 0.1
log_offset = 1e-8 # 1e-20
norm_offset = 1e-8  # 我加入的
det_offset = 1e-6
CEloss = nn.CrossEntropyLoss()


## Function ##
def Entropy(input):
    #input shape is batch_size X num_class
    ent = torch.sum(-torch.mul(input, torch.log(input + log_offset)), dim=-1)
    return ent
    # return tf.reduce_sum(-tf.multiply(input, tf.log(input + log_offset)), axis=-1)


def Ensemble_Entropy(y_true, y_pred, num_model=args.num_models):
    y_p = torch.split(y_pred, split_size_or_sections=int(y_pred.size()[0]/num_model), dim=0)
    y_p_all = 0
    for i in range(num_model):
        y_p_all += y_p[i]
    Ensemble = Entropy(y_p_all / num_model)
    return Ensemble


def log_det(y_true, y_pred, num_model=args.num_models):  # y_true为标签值

    # bool_R_y_true = tf.not_equal(tf.ones_like(y_true) - y_true, 0)  # batch_size X (num_class X num_models), 2-D
    # mask_non_y_pred = tf.boolean_mask(y_pred, bool_R_y_true) # batch_size X (num_class-1) X num_models, 1-D
    # mask_non_y_pred = tf.reshape(mask_non_y_pred, [-1, num_model, args.num_classes-1]) # batch_size X num_model X (num_class-1), 3-D
    # mask_non_y_pred = mask_non_y_pred / tf.norm(mask_non_y_pred, axis=2, keepdims=True) # batch_size X num_model X (num_class-1), 3-D
    # matrix = tf.matmul(mask_non_y_pred, tf.transpose(mask_non_y_pred, perm=[0, 2, 1])) # batch_size X num_model X num_model, 3-D
    # all_log_det = tf.linalg.logdet(matrix + det_offset * tf.expand_dims(tf.eye(num_model), 0)) # batch_size X 1, 1-D

    # if args.num_labels > 2:
    #     y_true = F.one_hot(y_true, args.num_labels)
    #     bool_R_y_true = torch.ne(torch.ones_like(y_true) - y_true, 0)  # 标记y_true为0的地方为True（过滤实际label对应位置）
    #     mask_non_y_pred = torch.masked_select(y_pred, bool_R_y_true)  # 保留除了实际label以外其他label的预测概率
    #     mask_non_y_pred = torch.reshape(mask_non_y_pred, [-1, num_model, args.num_labels - 1])
    #     mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, dim=2, keepdim=True)  # 对于二分类，此时mask_non_pred里只剩一个label，因此二范数仍然是对于该label的预测值本身，因此除完之后是1
    #     matrix = torch.matmul(mask_non_y_pred, torch.transpose(mask_non_y_pred, 2, 1))
    #     all_log_det = torch.logdet(matrix + det_offset * torch.unsqueeze(torch.eye(num_model).cuda(args.rank), 0))
    # else:
    #     y_pred = torch.reshape(y_pred, [-1, num_model, args.num_labels])
    #     y_pred_norm = y_pred / torch.norm(y_pred, dim=2, keepdim=True)
    #     matrix = torch.matmul(y_pred_norm, torch.transpose(y_pred_norm, 2, 1))
    #     all_log_det = torch.logdet(matrix + det_offset * torch.unsqueeze(torch.eye(num_model).cuda(args.rank), 0))
    print('------')
    print(y_pred.size())
    y_pred = torch.reshape(y_pred, [-1, num_model, args.num_labels])
    # y_pred_norm = y_pred / torch.norm(y_pred, dim=2, keepdim=True)
    matrix = torch.matmul(y_pred, torch.transpose(y_pred, 2, 1))
    print(matrix).size()
    exit(0)
    all_log_det = torch.logdet(matrix + det_offset * torch.unsqueeze(torch.eye(num_model).cuda(args.rank), 0))
    return all_log_det


def Loss_withEE_DPP(y_true, y_pred, num_model=args.num_models): # y_pred [batch_size*num_models, num_classes]
    chunk_size = int(y_true.size()[0]/num_model)
    y_p = torch.split(y_pred, split_size_or_sections=chunk_size, dim=0)
    y_t = torch.split(y_true, split_size_or_sections=chunk_size, dim=0)
    CE_all = 0
    for i in range(num_model):
        CE_all += CEloss(y_p[i], y_t[i])
    if args.lamda == 0 and args.log_det_lamda == 0:
        return CE_all
    y_pred = nn.functional.softmax(y_pred, dim=-1)  # 需要softmax？为什么需要？
    y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)  # 截断，防止过小导致后续log操作出现-inf
    if args.lamda == 0:
        EE = torch.tensor(0)
        log_dets = log_det(y_true, y_pred, num_model)
    elif args.log_det_lamda == 0:
        EE = Ensemble_Entropy(y_true, y_pred, num_model)
        log_dets = torch.tensor(0)
    else:
        EE = Ensemble_Entropy(y_true, y_pred, num_model)
        log_dets = log_det(y_true, y_pred, num_model)
    return CE_all - args.lamda * EE - args.log_det_lamda * log_dets, CE_all, args.lamda * EE, args.log_det_lamda * log_dets


def train_epoch(epoch, best_test, model, optimizer, dataloader_train, test_x, test_y, tokenizer):
    """Train for a epoch"""

    # Optimize
    time_start = time.time()
    model.train()

    """Eval for bert"""
    # if args.target_model == 'bert':
    #     if torch.cuda.device_count() > 1:
    #         if args.task == 'snli':
    #             model.module.model.eval()
    #         else:
    #             model.module.model.bert.eval()
    #     else:
    #         if args.task == 'snli':
    #             model.model.eval()
    #         else:
    #             model.model.bert.eval()

    # criterion = nn.CrossEntropyLoss()
    cnt = 0
    with autograd.detect_anomaly():
        for idx, (*train_x, train_y) in enumerate(dataloader_train):
            optimizer.zero_grad()
            input_ids, input_mask, segment_ids, model_nos, train_y = \
                train_x[0].cuda(args.rank), train_x[1].cuda(args.rank), train_x[2].cuda(args.rank), train_x[3].cuda(args.rank), train_y.cuda(args.rank)
            cnt += 1
            if torch.cuda.device_count() > 1:
                logits = []
                true_labels = []
                for i in range(args.num_models):
                    # model_no = torch.tensor([i], dtype=torch.float, requires_grad=True).cuda(args.rank)
                    _, lg = model.module.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, model_ids=i)
                    # params_aux = model.aux(model_no)
                    # s_index = 0  # 在辅助网络输出中的索引
                    # param_org_bert = {}
                    # for name, param in model.bert.named_parameters():
                    #     param_org = param.clone()  # param_org与param不共享内存
                    #     # param_org.requires_grad = False  # 不计算梯度
                    #     param_org_bert[name] = param_org
                    #     # print('param_org', param_org, param_org.requires_grad)
                    #     # print('delta param', args.aux_weight * params_aux[s_index: s_index + param_num].reshape(param.shape), params_aux.requires_grad)
                    #     if args.modify_attentions:  # 只修改第0层self-attention部分
                    #         if 'encoder.layer.0.attention' not in name:
                    #             continue
                    #     param_num = param.numel()  # 参数个数
                    #     # 修改param
                    #     # temp = args.aux_weight * params_aux[s_index: s_index + param_num].reshape(param.shape)
                    #     # print(param_org.requires_grad, temp.requires_grad)
                    #     # param.data.add_(temp)  # bert的参数随之改变
                    #     # print(param.weight)
                    #     # print(param)
                    #     # param = param +
                    #     # param = param_org + args.aux_weight * params_aux[s_index: s_index + param_num].reshape(param.shape)
                    #     param.add_(args.aux_weight * params_aux[s_index: s_index + param_num].reshape(param.shape))
                    #     # print('param', param, param.requires_grad)
                    #     # for name_, param_ in model.bert.named_parameters():
                    #     #     if name_ == name:
                    #     #         print('param in bert', param_, param_.requires_grad)
                    #     #         exit(0)
                    #     s_index += param_num
                    # 用修改后的bert进行预测
                    # _, lg = model.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, model_ids=model_no)
                    # lg = nn.functional.softmax(lg, dim=-1)  # 需要softmax？为什么需要？
                    # lg = torch.clamp(lg, min=1e-7, max=1 - 1e-7)  # 截断，防止过小导致后续log操作出现-inf
                    logits.append(lg)
                    true_labels.append(train_y)
                    # 将bert参数重置为初始值
                    # print('-----------')
                    # for name, param in model.bert.named_parameters():
                    #     param.data = param_org_bert[name]
                    #     param.data.requires_grad = True
                    #     # print('name', name)
                    #     # print('param_org', param_org_bert[name])
                    #     # param.data = param_org_bert[name].cpu()  #赋值并释放param_org_bert所占显存
                    #     # print('param_org', param_org_bert[name])
                    #     # param.data = param.data.cuda(args.rank)
                    #     # for name_, param_ in model.bert.named_parameters():
                    #     #     if name_ == name:
                    #     #         print('param in bert', param_, param_.requires_grad)
                    #     #         exit(0)
                logits = torch.cat(logits, dim=0)
                true_labels = torch.cat(true_labels, dim=0)
            else:
                logits = []
                true_labels = []
                for i in range(args.num_models):
                    # model_no = torch.tensor([i], dtype=torch.float, requires_grad=True).cuda(args.rank)
                    _, lg = model.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, model_ids=i)
                    # params_aux = model.aux(model_no)
                    # s_index = 0  # 在辅助网络输出中的索引
                    # param_org_bert = {}
                    # for name, param in model.bert.named_parameters():
                    #     param_org = param.clone()  # param_org与param不共享内存
                    #     # param_org.requires_grad = False  # 不计算梯度
                    #     param_org_bert[name] = param_org
                    #     # print('param_org', param_org, param_org.requires_grad)
                    #     # print('delta param', args.aux_weight * params_aux[s_index: s_index + param_num].reshape(param.shape), params_aux.requires_grad)
                    #     if args.modify_attentions:  # 只修改第0层self-attention部分
                    #         if 'encoder.layer.0.attention' not in name:
                    #             continue
                    #     param_num = param.numel()  # 参数个数
                    #     # 修改param
                    #     # temp = args.aux_weight * params_aux[s_index: s_index + param_num].reshape(param.shape)
                    #     # print(param_org.requires_grad, temp.requires_grad)
                    #     # param.data.add_(temp)  # bert的参数随之改变
                    #     # print(param.weight)
                    #     # print(param)
                    #     # param = param +
                    #     # param = param_org + args.aux_weight * params_aux[s_index: s_index + param_num].reshape(param.shape)
                    #     param.add_(args.aux_weight * params_aux[s_index: s_index + param_num].reshape(param.shape))
                    #     # print('param', param, param.requires_grad)
                    #     # for name_, param_ in model.bert.named_parameters():
                    #     #     if name_ == name:
                    #     #         print('param in bert', param_, param_.requires_grad)
                    #     #         exit(0)
                    #     s_index += param_num
                    # 用修改后的bert进行预测
                    # _, lg = model.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, model_ids=model_no)
                    # lg = nn.functional.softmax(lg, dim=-1)  # 需要softmax？为什么需要？
                    # lg = torch.clamp(lg, min=1e-7, max=1 - 1e-7)  # 截断，防止过小导致后续log操作出现-inf
                    logits.append(lg)
                    true_labels.append(train_y)
                    # 将bert参数重置为初始值
                    # print('-----------')
                    # for name, param in model.bert.named_parameters():
                    #     param.data = param_org_bert[name]
                    #     param.data.requires_grad = True
                    #     # print('name', name)
                    #     # print('param_org', param_org_bert[name])
                    #     # param.data = param_org_bert[name].cpu()  #赋值并释放param_org_bert所占显存
                    #     # print('param_org', param_org_bert[name])
                    #     # param.data = param.data.cuda(args.rank)
                    #     # for name_, param_ in model.bert.named_parameters():
                    #     #     if name_ == name:
                    #     #         print('param in bert', param_, param_.requires_grad)
                    #     #         exit(0)
                logits = torch.cat(logits, dim=0)
                true_labels = torch.cat(true_labels, dim=0)

            # loss = criterion(output, train_y)
            loss, loss_ce, loss_e, loss_d = Loss_withEE_DPP(true_labels, logits)
            # loss = torch.mean(param)  # 这样可训练
            loss = loss.mean()
            # print(loss.requires_grad, loss.grad)
            # loss.requires_grad = True
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            # print(temp.grad)
            # for name, p in model.bert.bert.encoder.aux.named_parameters():
            #     if p.requires_grad:
            #         print(name, p.grad, p.grad_fn)
            optimizer.step()
            # for name, p in model.bert.bert.encoder.aux.named_parameters():
            #     if p.requires_grad:
            #         print(name, p.grad, p.grad_fn)
            # exit(0)
    lr_decay = 1.0
    if lr_decay > 0:
        optimizer.param_groups[0]['lr'] *= lr_decay

    # For process 0
    if is_main_process() and epoch % 1 == 0:
        test_acc = eval_model(model, test_x, test_y)
        time_end = time.time()
        time_used = time_end - time_start
        sys.stdout.write("Epoch={} time={:.2f}s train_loss={:.6f} loss_ce={:.6f} loss_e={:.6f} loss_d={:.6f} test_acc={:.6f}(#{})\n".format(
        epoch, time_used, loss.item(), loss_ce.mean().item(), loss_e.mean().item(), loss_d.mean().item(), test_acc, len(test_y)))
        if test_acc > best_test:
            best_test = test_acc
            if torch.cuda.device_count() > 1:
                if args.save_path:
                    save_path = args.save_path
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(model.module.state_dict(), save_path + '/pytorch_model.bin')
                    model.module.bert.config.to_json_file(save_path + '/bert_config.json')
                    if args.task != 'snli':
                        tokenizer.save_vocabulary(save_path)
            else:
                if args.save_path:
                    save_path = args.save_path
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(model.state_dict(), save_path + '/pytorch_model.bin')
                    model.bert.config.to_json_file(save_path + '/bert_config.json')
                    if args.task != 'snli':
                        tokenizer.save_vocabulary(save_path)
            print('save model when test acc=', test_acc)

    # time_end = time.time()
    # time_used = time_end - time_start
    # 保存当前epoch
    if is_main_process():
        if torch.cuda.device_count() > 1:
            save_path = args.save_path + '_final'
            if not os.path.exists(save_path): os.makedirs(save_path)
            torch.save(model.module.state_dict(), save_path + '/pytorch_model.bin')
            model.module.bert.config.to_json_file(save_path + '/bert_config.json')
            if args.task != 'snli':
                tokenizer.save_vocabulary(save_path)
            # torch.save(model.model.state_dict(), save_path + '/pytorch_model.bin')
            # model.model.config.to_json_file(save_path + '/bert_config.json')
            # tokenizer.save_vocabulary(save_path)
            # print('Save final model when epoch=', epoch)
        else:
            save_path = args.save_path + '_final'
            if not os.path.exists(save_path): os.makedirs(save_path)
            torch.save(model.state_dict(), save_path + '/pytorch_model.bin')
            model.bert.config.to_json_file(save_path + '/bert_config.json')
            if args.task != 'snli':
                tokenizer.save_vocabulary(save_path)
            # torch.save(model.model.state_dict(), save_path + '/pytorch_model.bin')
            # model.model.config.to_json_file(save_path + '/bert_config.json')
            # tokenizer.save_vocabulary(save_path)
            # print('Save final model when epoch=', epoch)

    dist.barrier()  # 这一句作用是：所有进程(gpu)上的代码都执行到这，才会执行该句下面的代码
    return best_test


def main(args):
    """Setting for parallel training"""
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    # args.rank = 0

    init_distributed_mode(args=args)
    torch.cuda.set_device(args.rank)

    """Load data"""
    test_x, test_y = load_test_data()
    train_x, train_y = load_train_data()

    # 随机选取1000个测试集作为验证集（从200个往后取，这样与攻击数据无重合）
    if args.task == 'imdb':
        test_x, test_y = test_x[200:], test_y[200:]
        c = list(zip(test_x, test_y))
        random.seed(15)
        random.shuffle(c)
        test_x, test_y = zip(*c)
        test_x, test_y = test_x[:1000], test_y[:1000]

    # train_x, train_y = train_x[:200], train_y[:200]
    # test_x, test_y = test_x[:10], test_y[:10]

    tokenizer = BertTokenizer.from_pretrained(args.target_model_path, do_lower_case=True)  # 用来保存模型
    model = EnsembleBERT_comp(args).cuda(args.rank)

    dataset_train = Dataset_BERT(args)

    if torch.cuda.device_count() > 1:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], find_unused_parameters=True)
        # model = model.module
    all_data_train, _ = dataset_train.transform_text(train_x, train_y)

    if torch.cuda.device_count() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(all_data_train)
        nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
        dataloader_train = DataLoader(all_data_train, sampler=train_sampler, pin_memory=True, num_workers=nw,
                                      batch_size=args.batch_size)  # batch_
    else:
        train_sampler = SequentialSampler(all_data_train)
        dataloader_train = DataLoader(all_data_train, sampler=train_sampler, batch_size=args.batch_size)

    """关闭bert部分的参数更新"""
    if torch.cuda.device_count() > 1:
        model.module.bert.requires_grad_(False)
        model.module.bert.bert.encoder.aux.requires_grad_(True)
    else:
        model.bert.requires_grad_(False)
        model.bert.bert.encoder.aux.requires_grad_(True)
    # if args.target_model == 'bert':
    #     if torch.cuda.device_count() > 1:
    #         if args.task == 'snli':
    #             model.module.model.requires_grad_(False)  # snli
    #         else:
    #             model.module.model.bert.requires_grad_(False)  # mr/imdb
    #     else:
    #         if args.task == 'snli':
    #             model.model.requires_grad_(False)  # snli
    #         else:
    #             model.model.bert.requires_grad_(False)  # mr/imdb

    # if is_main_process():
    #     test_acc = eval_model(model, test_x, test_y)
    #     print('Acc for test set is: {:.2%}(#{})'.format(test_acc, len(test_y)))
    # exit(0)

    if is_main_process():
        print('#Train data:', len(train_y))
        para = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Model {} : {:4f}M params'.format(model._get_name(), para * 4 / 1000 / 1000))

    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=args.lr)
    epoch = 200  # 10
    best_test = 0
    # params_aux_s = {}
    for e in range(epoch):
        if torch.cuda.device_count() > 1:
            train_sampler.set_epoch(epoch)
        # for name, p in model.bert.bert.encoder.aux.named_parameters():
        #     if name == 'dense.weight':
        #         print(name, e, p)
        #         params_aux_s[e] = p
        best_test = train_epoch(e, best_test, model, optimizer, dataloader_train, test_x, test_y, tokenizer)
    # if params_aux_s[0].equal(params_aux_s[1]):
    #     print('eeeeeee')
    #     exit(0)

    # if is_main_process():
    #     test_acc = eval_model(model, test_x, test_y)
    #     print('Finally, acc for test set is: {:.2%}(#{})'.format(test_acc, len(test_y)))


if __name__ == '__main__':
    main(args)

    # with open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/train_set/tf_imdb_bert_success.pkl', 'rb') as fp:
    #     input_list, true_label_list, output_list, success, change_list, num_change_list, success_time = pickle.load(fp)
    #     output_list = [adv.split(' ') for adv in output_list]
    #     print(output_list[1])
