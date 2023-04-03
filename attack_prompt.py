import random
import torch
import pickle

import os
from config import args
import itertools
import numpy as np
import re
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, RobertaForMaskedLM, RobertaTokenizer
from transformers import pipeline
from BERT.modeling import BertForSequenceClassification, BertConfig,BertForMaskedLM
from BERT.tokenization import BertTokenizer
from models import LSTM, ESIM, BERT, BERT_snli, ROBERTA
from torch.utils.data import DataLoader,SequentialSampler
from dataset import Dataset_ROBERTA, Dataset_BERT
import math
from train_model import load_test_data

stop_mark = ['.', ',', '?', ';', '!']
# prompt_templates = [[], []]
prompt_templates = [['it', 'is', 'a', 'good', 'movie', '.'], ['it', 'is', 'a', 'bad', 'movie', '.']]
# prompt_templates = [['i', 'like', 'the', 'movie', 'so', 'much', '.'], ['i', 'hate', 'the', 'movie', 'so', 'much', '.']]
# prompt_templates = [['it', 'is', 'a', 'funny', 'movie', '.'], ['it', 'is', 'a', 'boring', 'movie', '.']]
# prompt_templates = [['i', 'think', 'it', 'is', 'funny' , '.'], ['i', 'think', 'it', 'is', 'boring','.']]

"""利用GPT2计算文本困惑度"""
def cal_ppl_bygpt2():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()

    """计算GPT2对于一条文本的困惑度"""
    def score(text):
        text = text.replace(' ##','')
        tokenize_input = tokenizer.tokenize(text)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).cuda()
        loss = model(tensor_input, labels=tensor_input)['loss']
        ppl = math.exp(loss) # 这是真正的困惑度，返回的是log ppl
        return ppl

    """1. 读取生成的对抗样本"""
    # 确定读取的文件名

    """1.1 prompt产生的"""
    # lpt = True if args.load_prompt_trained else False
    # we = True if args.word_emb else False
    # fr = open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/pat_%s_%s_%s_%s_%.2f_%d_%d_we%s_lpt%s_phead_p0.json' %
    #           (args.task, args.target_model, args.attack_level, args.mask_mode, args.mask_ratio, args.sample_size,
    #            args.topk, we, lpt), 'r')
    # # """textfooler产生的"""
    # # fr = open('/pub/data/huangpei/TextFooler/adv_results/test_set/org/tf_%s_%s_adv_symTrue.json' % (args.task, args.target_model), 'r')
    # data = json.load(fr)
    # print('Data size for evaluating:', len(data))
    # """对每条文本进行困惑度评分"""
    # adv_num = 0  # 保存所有的对抗样本个数（每条测试样例可能有多条对抗样本）
    # adv_ppl_sum = 0.0  # 求和所有对抗样本的ppl
    # # 每条样例有多条对抗样本，怎么处理？
    # all_best_ppls = []  # 保存每条测试样例对抗样本中ppl最小的那个 的ppl
    # for idx, dd in data.items():  # 对于每条样例
    #     # start_time = time.clock()
    #     adv_texts = dd['adv_texts']
    #     adv_ppls = []
    #     for adv_text in adv_texts:
    #         adv_text = adv_text[:512]  # gpt的输入要小于512
    #         score_a = score(adv_text)
    #         adv_ppls.append(score_a)
    #         adv_num += 1
    #         adv_ppl_sum += score_a
    #     best_ppl = np.min(np.array(adv_ppls))
    #     all_best_ppls.append(best_ppl)
    #     # data[idx]['adv_ppls'] = adv_ppls
    #     # print('Time for a data:', time.clock()-start_time)

    """1.2 sempso产生的(adv只有一条)"""
    fr = open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/sempso/test_set/org/sem_%s_%s_success.pkl' % (args.task, args.target_model), 'rb')
    # fr = open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/sempso/test_set/org/sem_%s_%s_success.pkl' % (args.task, args.target_model), 'rb')
    input_list, true_label_list, output_list, success, change_list, num_change_list, success_time = pickle.load(fr)
    data = output_list
    print('Data size for evaluating:', len(data))
    """对每条文本进行困惑度评分"""
    adv_num = 0  # 保存所有的对抗样本个数（每条测试样例可能有多条对抗样本）
    all_best_ppls = []
    for idx, dd in enumerate(data):
        adv_text = ' '.join(dd)
        adv_text = adv_text[:512]  # gpt的输入要小于512

        score_a = score(adv_text)
        adv_num += 1
        all_best_ppls.append(score_a)


    """3. 保存评分结果"""
    # if args.word_emb:
    #     fw = open('/pub/data/huangpei/TextFooler/prompt_results/adv_ppl_%s_%s_%s_%.2f_%d_%d_we_lpt%s.json' % (
    #     args.task, args.target_model, args.mask_mode, args.mask_ratio, args.sample_size, args.topk, lpt), 'w')
    # else:
    #     fw = open('/pub/data/huangpei/TextFooler/prompt_results/adv_ppl_%s_%s_%s_%.2f_%d_%d_lpt%s.json' % (
    #     args.task, args.target_model, args.mask_mode, args.mask_ratio, args.sample_size, args.topk, lpt), 'w')
    # json.dump(data, fw, indent=4)

    # adv_ppl_mean = adv_ppl_sum/adv_num
    adv_ppl_mean = np.mean(np.array(all_best_ppls))
    print('%d instances, %d adversarial examples, %f mean ppl for GPT2' % (len(data), adv_num, adv_ppl_mean))


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values

def gpt_generate(data, args):
    """利用gpt进行续写，再做词语mask-fill"""
    # text = "I like the movie."
    # text = "A Three Stooges short , this one featuring Shemp . Of all those involving Shemp I 've seen , this is my favorite performance by him in a Stooges short . The basic plot is that Shemp must get married by 6 o'clock that very evening if he 's to inherit the half a million dollars a rich uncle left him in his will . So Shemp sets out to get himself a bride but finds it a tougher road than expected , that is until they learn of his inheritance money . Best bits here involve Shemp shaving , Shemp and Moe in a telephone booth and Larry on piano as accompaniment to Shemp 's voice - training session . Also the sequence where Shemp is mistaken as Cousin Basil and its outcome proves hilarious ."
    # generator = pipeline('text-generation', model='gpt2')
    # output_length = len(text.split(' ')) + 50
    # time_start = time.time()
    # """greedy"""
    # outs = generator(text, early_stopping=True)
    # print('----------greedy---------------')
    # """beam search"""
    # outs = generator(text, max_length=output_length, num_beams=3, early_stopping=True, num_return_sequences=5, ) # no_repeat_ngram_size=2
    # print('----------beam search---------------')
    # """sample"""
    # # tf.random.set_seed(10)
    # # outs =  generator(text, max_length=output_length, do_sample=True, top_k=0, num_return_sequences=5, temperature=0.7) # temperature=0.7
    # # print('----------sample---------------')
    # """Top k"""
    # # tf.random.set_seed(10)
    # # outs =  generator(text, max_length=output_length, do_sample=True, top_k=50, num_return_sequences=5,) # temperature=0.7
    # # print('----------Top k---------------')
    # """Top p"""
    # # outs = generator(text, max_length=output_length, do_sample=True, top_k=0, num_return_sequences=5, top_p=0.92)  # temperature=0.7
    # # print('----------Top p---------------')
    # """"""
    # # outs = generator(text, max_length=output_length, do_sample=True, top_k=50, top_p=0.92, num_return_sequences=5)  # temperature=0.7
    # # print('----------Top p2---------------')
    # print('Outs: ', outs)
    # time_end = time.time()
    # gen_time = time_end - time_start
    # print('Time: ', gen_time)


    generator = pipeline('text-generation', model='gpt2')
    # output_length = args.max_seq_length + 10 # mr
    output_length = 350 # imdb
    num_candi_sents = 5
    texts = []
    outs = {}
    for idx, (text, label) in enumerate(data):
        # 文本截断到max_seq_length之内
        text = text[:args.max_seq_length-9] # 至少留10个位置用来续写
        # 若文本末尾本身有符号，去掉
        if text[-1] in stop_mark:
            text = text[:-1]
        # 文本末尾加入逗号
        text += [',']
        texts.append(' '.join(text))
        outs[idx] = {'label': label}  # 处理第100-200条数据时，100+idx


    """greedy"""
    # gpt_output = generator(texts, max_length=output_length, num_return_sequences=num_candi_sents)
    """beam search"""
    gpt_output = generator(texts, max_length=output_length, num_beams=5, early_stopping=True, num_return_sequences=num_candi_sents, )

    for idx in range(len(gpt_output)):
        candi_sents = [go['generated_text'] for go in gpt_output[idx]]
        outs[idx]['candi_texts'] = candi_sents  # 处理第100-200条数据时，100+idx

    with open('/pub/data/huangpei/TextFooler/prompt_results/gpt12_beamSearch_%s_comma.json' % (args.task), 'w') as fw:
        json.dump(outs, fw, indent=4)

    # 写到第一个句号就结束（减少生成时间）

"""处理gpt生成的文本"""
def clean_gpt_generate(data): # data：续写前数据
    data_path = '/pub/data/huangpei/TextFooler/prompt_results/'
    data_path1 = data_path + 'gpt12_beamSearch_%s_comma.json' % args.task
    # data_path2 = data_path + 'gpt12_beamSearch_%s_comma_2.json' % args.task
    with open(data_path1, 'r') as fr:
        gpt_results = json.load(fr)
    # with open(data_path2, 'r') as fr:
    #     gpt_results2 = json.load(fr)
    # gpt_results = dict(gpt_results1, **gpt_results2)

    for idx in gpt_results.keys():
        org_text, label = data[int(idx)]
        candi_texts = gpt_results[idx]['candi_texts']  # 5条
        candi_texts_clean = []
        for candi_text in candi_texts:
            wrote_sent = candi_text.split(' ')[len(org_text)+1:]   # 取出续写部分
            wrote_sent = re.split(r"([.。!！?？；;])", ' '.join(wrote_sent))[0]  # 取出续写的第一条句子
            wrote_sent = wrote_sent.replace('\n\n','').replace('\\','').replace('\u00a0','') # 过滤奇怪符号
            wrote_sent = (' '.join(org_text) + ' , ' + wrote_sent + ' .').replace('  ',' ') # 在续写句子前添上逗号，后添上句号；替换连续两个空格为1个空格
            candi_texts_clean.append(wrote_sent)
        gpt_results[idx]['candi_texts'] = candi_texts_clean

    with open('/pub/data/huangpei/TextFooler/prompt_results/gpt12_beamSearch_%s_comma.json' % (args.task), 'w') as fw:
        json.dump(gpt_results, fw, indent=4)


def mask_and_fill(data, predictor, generator, tokenizer, dataset):
    """1. 获得mask位置，并mask"""
    outs = {}

    if args.mask_mode == 'random':
        """1.1 随机mask"""
        for idx, (text, label) in enumerate(data):
            if text[-1] not in stop_mark:  # 可能存在原始文本的最后一个词语不是分隔符
                text.append('.')
            masked_flags = []
            masked_ori_words = []
            masked_texts = []

            for jj in range(min(args.max_seq_length, len(text))):
                word = text[jj]
                r_seed = np.random.rand(args.sample_size)
                n = [args.mask_token if rs < args.mask_ratio else word for rs in r_seed]
                masked_texts.append(n)
                m = [True if rs < args.mask_ratio else False for rs in r_seed]
                masked_flags.append(m)
            masked_texts = np.array(masked_texts).T.tolist()
            masked_flags = np.array(masked_flags).T.tolist()
            for i in range(args.sample_size):
                masked_ori_words.append(np.array(text[:min(args.max_seq_length, len(text))])[masked_flags[i]].tolist())
            # 注：可能会把符号mask掉
            # 在文本开头加入prompt
            masked_texts = [prompt_templates[label] + mt for mt in masked_texts]
            # 在每句话后面加入prompt
            # for i in range(len(masked_texts)):
            #     text_lst = re.split(r"([.。!！?？；;，,])", ' '.join(masked_texts[i]))
            #     text_lst.append("")
            #     text_lst = ["".join(i) for i in zip(text_lst[0::2], text_lst[1::2])]
            #     text_lst = [tl + ' ' + ' '.join(prompt_templates[label]) for tl in text_lst]
            #     text_lst = ' '.join(prompt_templates[label]) + ' ' + ' '.join(text_lst)
            #     masked_texts[i] = text_lst.split(' ')

            outs[idx] = {'label': label, 'text': text, 'masked_texts': masked_texts, 'masked_ori_words':masked_ori_words}
    # 输入的每个位置进行打分：1. 分类重要性 2. 语法重要性

    elif args.mask_mode == 'clsImp':
        """1.2 分类重要性"""
        for idx, (text, label) in enumerate(data): # 对于每条测试样例
            masked_texts = []
            orig_probs = predictor([text], [text]).squeeze()
            orig_label = torch.argmax(orig_probs)
            orig_prob = orig_probs.max()
            len_text = len(text)
            leave_1_texts = [text[:ii] + ['<oov>'] + text[min(ii + 1, len_text):] for ii in range(len_text)]
            leave_1_probs = predictor([text for i in range(len(leave_1_texts))], leave_1_texts)
            leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
            a = (leave_1_probs_argmax != orig_label).float()
            b = leave_1_probs.max(dim=-1)[0]
            c = torch.index_select(orig_probs, 0, leave_1_probs_argmax)
            d = b - c
            if len(leave_1_probs.shape) == 1: # 说明该文本只有一个单词，增加一维
                leave_1_probs = leave_1_probs.unsqueeze(0)
            import_scores = (orig_prob - leave_1_probs[:, orig_label] + a * d).data.cpu().numpy()
            masked_text = text.copy()
            # mask掉重要的：取前mask_ratio个重要的位置进行mask（所以只有1条？）
            masked_pos = np.argsort(-import_scores).tolist()[:int(len_text*args.mask_ratio)]
            for mp in masked_pos:
                masked_text[mp] = args.mask_token
            # mask掉不重要的
            # masked_pos = np.argsort(import_scores).tolist()[:int(len_text * args.mask_ratio)]
            # for mp in masked_pos:
            #     masked_text[mp] = args.mask_token

            masked_texts.append(masked_text)
            # 在每句话后面加入prompt
            for i in range(len(masked_texts)):
                text_lst = re.split(r"([.。!！?？；;，,])", ' '.join(masked_texts[i]))
                text_lst.append("")
                text_lst = ["".join(i) for i in zip(text_lst[0::2], text_lst[1::2])]
                text_lst = [tl + ' ' + ' '.join(prompt_templates[label]) for tl in text_lst]
                text_lst = ' '.join(prompt_templates[label]) + ' ' + ' '.join(text_lst)
                masked_texts[i] = text_lst.split(' ')

            outs[idx] = {'label': label, 'text': tokenizer.tokenize(' '.join(text)), 'masked_texts': masked_texts}

    elif args.mask_mode == 'selectedPS':
        """1.3 语义保持：避开重要位置"""
        for idx, (text, label) in enumerate(data):  # 对于每条测试样例
            # 确定可mask位置
            # print(text)
            candi_ps = []
            for ii, (word, ps) in enumerate(args.pos_tags[' '.join(text)]):
                # print(word, ps)
                if ps.startswith('JJ'):  # 避开形容词JJ、副词RB、实义动词VB or ps.startswith('RB') or ps.startswith('VB')
                    continue
                else:
                    candi_ps.append(ii)
            if text[-1] not in stop_mark:  # 可能存在原始文本的最后一个词语不是分隔符
                text.append('.')
                candi_ps.append(len(text)-1) # 候选位置加上最后一个分隔符
            # print(args.pos_tags[' '.join(text)])
            # print(candi_ps)
            # exit(0)
            # 进行mask
            masked_texts = []
            masked_flags = []
            masked_ori_words = []
            for jj in range(len(text)):  #######截断不合适，后面补全的时候可能可以用到后面的信息
                word = text[jj]
                if jj in candi_ps:
                    r_seed = np.random.rand(args.sample_size)
                    n = [args.mask_token if rs < args.mask_ratio else word for rs in r_seed]
                    m = [True if rs < args.mask_ratio else False for rs in r_seed]
                else:
                    n = [word] * args.sample_size
                    m = [False] * args.sample_size
                masked_texts.append(n)
                masked_flags.append(m)

            masked_texts = np.array(masked_texts).T.tolist()
            masked_flags = np.array(masked_flags).T.tolist()
            for i in range(args.sample_size):
                # masked_ori_words.append(np.array(text[:min(args.max_seq_length, len(text))])[masked_flags[i]].tolist())  # 截断
                masked_ori_words.append(np.array(text)[masked_flags[i]].tolist())  # 不截断
            # 注：可能会把符号mask掉
            # 在文本开头加入prompt
            masked_texts = [prompt_templates[label] + mt for mt in masked_texts]
            # 在每句话后面加入prompt
            # for i in range(len(masked_texts)):
            #     text_lst = re.split(r"([.。!！?？；;，,])", ' '.join(masked_texts[i])) # 符号可能被mask了，怎么办
            #     text_lst.append("")
            #     text_lst = ["".join(i) for i in zip(text_lst[0::2], text_lst[1::2])]
            #     text_lst = [tl + ' ' + ' '.join(prompt_templates[label]) for tl in text_lst]
            #     text_lst = ' '.join(prompt_templates[label]) + ' ' + ' '.join(text_lst)
            #     masked_texts[i] = text_lst.split(' ')
            outs[idx] = {'label': label, 'text': text, 'masked_texts': masked_texts, 'masked_ori_words':masked_ori_words}

            # outs[idx] = {'label':label, 'text':tokenizer.tokenize(' '.join(text)),'masked_texts':masked_texts,'masked_ori_words': masked_ori_words}

    elif args.mask_mode == 'weightedPS':
        """每个位置有权重，综合考虑"""
        for idx, (text, label) in enumerate(data):  # 对于每条测试样例
            masked_texts = np.array([text] * args.sample_size)
            """权重1：分类重要性"""
            orig_probs = predictor([text], [text]).squeeze()
            orig_label = torch.argmax(orig_probs)
            orig_prob = orig_probs.max()
            len_text = len(text)
            leave_1_texts = [text[:ii] + ['<oov>'] + text[min(ii + 1, len_text):] for ii in range(len_text)]
            leave_1_probs = predictor([text for i in range(len(leave_1_texts))], leave_1_texts)
            leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
            a = (leave_1_probs_argmax != orig_label).float()
            b = leave_1_probs.max(dim=-1)[0]
            c = torch.index_select(orig_probs, 0, leave_1_probs_argmax)
            d = b - c
            if len(leave_1_probs.shape) == 1:  # 说明该文本只有一个单词，增加一维
                leave_1_probs = leave_1_probs.unsqueeze(0)
            import_scores = (orig_prob - leave_1_probs[:, orig_label] + a * d)
            weight_cls = torch.nn.functional.softmax(import_scores)
            # weight_cls = (import_scores-torch.min(import_scores))/(torch.max(import_scores)-torch.min(import_scores))# normalize
            # softmax至0-1之间
            """权重2：词性重要性：对于adj、adv和vb给予较小权重。调节权重为0.5"""
            weight_pos = []
            alpha = 0.0
            for ii, (word, ps) in enumerate(args.pos_tags[' '.join(text)]):
                if ps.startswith('JJ') or ps.startswith('RB') or ps.startswith('VB'):
                    weight_pos.append(-1 * alpha)
                else:
                    weight_pos.append(1 * alpha)
            """合并多个权重"""
            weight = weight_cls.cpu() * torch.tensor(weight_pos)
            weight = torch.nn.functional.softmax(weight).data.numpy() # softmax

            """按照权重mask"""
            for i in range(args.sample_size):
                # m_pos = np.random.choice(len(text), p=weight.ravel(), size=int(args.mask_ratio*len(text)), replace=False)
                m_pos = np.random.choice(len(text), p=weight.ravel(), size=int(args.mask_ratio*len(text)), replace=False)
                masked_texts[i][m_pos] = args.mask_token
            masked_texts = masked_texts.tolist()
            # 在每句话后面加入prompt
            for i in range(len(masked_texts)):
                text_lst = re.split(r"([.。!！?？；;，,])", ' '.join(masked_texts[i]))
                text_lst.append("")
                text_lst = ["".join(i) for i in zip(text_lst[0::2], text_lst[1::2])]
                text_lst = [tl + ' ' + ' '.join(prompt_templates[label]) for tl in text_lst]
                text_lst = ' '.join(prompt_templates[label]) + ' ' + ' '.join(text_lst)
                masked_texts[i] = text_lst.split(' ')

            outs[idx] = {'label': label, 'text': tokenizer.tokenize(' '.join(text)), 'masked_texts': masked_texts}

    elif args.mask_mode == 'sentInsert':
        """获得插入的句子"""
        insert_sent_len = 5
        # 插入的句子：[args.mask_token, args.mask_token, args.mask_token, args.mask_token, args.mask_token, '.']
        # insert_sent = [args.mask_token] * insert_sent_len + ['.']  # 句末加入句号。所以插入的长度是insert_sent_len+1
        # 插入一条无情感句子（随机mask掉5个位置）
        insert_sent = np.array('my friend and me watch this movie in a sunny morning .'.split(' '))
        masked_ps = random.sample(range(len(insert_sent)), 5)
        insert_sent[masked_ps] = args.mask_token
        insert_sent = insert_sent.tolist()

        for idx, (text, label) in enumerate(data):  # 对于每条测试样例
            masked_texts = []
            """ 1.1 获得输入中各个句子分隔符位置"""
            if text[-1] not in stop_mark:  # 可能存在原始文本的最后一个词语不是分隔符
                text.append('.')
            stop_ps = []  # 保存分隔符位置
            for i in range(len(text)):
                if text[i] in stop_mark:
                    stop_ps.append(i)
            """ 1.2 确定插入句子位置"""
            # 随机插入一条句子
            # insert_sent_ps = random.sample(stop_ps, 1)
            # 在每个句子后面插入一条句子
            insert_sent_ps = stop_ps
            # 在prompt后面紧接着插入一条句子
            # insert_sent_ps = [-1]
            """ 1.3 进行mask插入"""
            for ip in insert_sent_ps:
                masked_text = prompt_templates[label] + text[:ip+1] + insert_sent + text[ip+1:]  # 插入prompt和mask句子
                masked_texts.append(masked_text)
            outs[idx] = {'label': label, 'text': tokenizer.tokenize(' '.join(text)), 'masked_texts': masked_texts}

    elif args.mask_mode == 'combine':
        """混合词语和句子级改写"""
        with open('/pub/data/huangpei/TextFooler/prompt_results/gpt12_beamSearch_%s_comma.json' % (args.task), 'r') as fr:
            gpt_results = json.load(fr)

        for idx, (text, label) in enumerate(data):  # 对于每条测试样例
            """词语级"""
            # 确定可mask位置
            candi_ps = []
            for ii, (word, ps) in enumerate(args.pos_tags[' '.join(text)]):
                if ps.startswith('JJ'):  # 避开形容词JJ、副词RB、实义动词VB or ps.startswith('RB') or ps.startswith('VB')
                    continue
                else:
                    candi_ps.append(ii)
            if text[-1] not in stop_mark:  # 可能存在原始文本的最后一个词语不是分隔符
                text.append('.')
                candi_ps.append(len(text)-1) # 候选位置加上最后一个分隔符
            # 进行mask
            masked_texts = []
            masked_flags = []
            masked_ori_words = []
            for jj in range(len(text)):  #######截断不合适，后面补全的时候可能可以用到后面的信息
                word = text[jj]
                if jj in candi_ps:
                    r_seed = np.random.rand(args.sample_size)
                    n = [args.mask_token if rs < args.mask_ratio else word for rs in r_seed]
                    m = [True if rs < args.mask_ratio else False for rs in r_seed]
                else:
                    n = [word] * args.sample_size
                    m = [False] * args.sample_size
                masked_texts.append(n)
                masked_flags.append(m)

            masked_texts = np.array(masked_texts).T.tolist()
            masked_flags = np.array(masked_flags).T.tolist()
            for i in range(args.sample_size):
                # masked_ori_words.append(np.array(text[:min(args.max_seq_length, len(text))])[masked_flags[i]].tolist())  # 截断
                masked_ori_words.append(np.array(text)[masked_flags[i]].tolist())  # 不截断
            # 注：可能会把符号mask掉
            # 在文本开头加入prompt
            masked_texts = [prompt_templates[label] + mt for mt in masked_texts]

            """句子级"""
            """确定续写文本"""
            wrote_text = gpt_results[str(idx)]['candi_texts'][0].split(' ')  # 5条，暂时取第一条
            with open('/pub/data/huangpei/TextFooler/prompt_results/pos_gpt12_beamSearch_%s_comma.pkl' % args.task, 'rb') as fp:
                wrote_pos_tags = pickle.load(fp)
            """对文本进行词语级"""
            # 确定可mask位置
            wrote_candi_ps = []
            for ii, (word, ps) in enumerate(wrote_pos_tags[' '.join(wrote_text)]):
                if ps.startswith('JJ'):  # 避开形容词JJ、副词RB、实义动词VB or ps.startswith('RB') or ps.startswith('VB')
                    continue
                else:
                    wrote_candi_ps.append(ii)
            if wrote_text[-1] not in stop_mark:  # 可能存在原始文本的最后一个词语不是分隔符
                wrote_text.append('.')
                wrote_candi_ps.append(len(wrote_text) - 1)  # 候选位置加上最后一个分隔符
            # 进行mask
            wrote_masked_texts = []
            wrote_masked_flags = []
            wrote_masked_ori_words = []
            for jj in range(len(wrote_text)):  #######截断不合适，后面补全的时候可能可以用到后面的信息
                word = wrote_text[jj]
                if jj in wrote_candi_ps:
                    r_seed = np.random.rand(args.sample_size)
                    n = [args.mask_token if rs < args.mask_ratio else word for rs in r_seed]
                    m = [True if rs < args.mask_ratio else False for rs in r_seed]
                else:
                    n = [word] * args.sample_size
                    m = [False] * args.sample_size
                wrote_masked_texts.append(n)
                wrote_masked_flags.append(m)

            wrote_masked_texts = np.array(wrote_masked_texts).T.tolist()
            wrote_masked_flags = np.array(wrote_masked_flags).T.tolist()
            for i in range(args.sample_size):
                # masked_ori_words.append(np.array(text[:min(args.max_seq_length, len(text))])[masked_flags[i]].tolist())  # 截断
                wrote_masked_ori_words.append(np.array(wrote_text)[wrote_masked_flags[i]].tolist())  # 不截断
            # 注：可能会把符号mask掉
            # 在文本开头加入prompt
            wrote_masked_texts = [prompt_templates[label] + mt for mt in wrote_masked_texts]

            masked_texts.extend(wrote_masked_texts)
            masked_ori_words.extend(wrote_masked_ori_words)
            outs[idx] = {'label':label, 'text':tokenizer.tokenize(' '.join(text)),'masked_texts':masked_texts,'masked_ori_words': masked_ori_words}

    elif args.mask_mode == 'sentInsert_gpt':
        with open('/pub/data/huangpei/TextFooler/prompt_results/gpt12_beamSearch_%s_comma.json' % (args.task), 'r') as fr:
            gpt_results = json.load(fr)
        for idx in gpt_results.keys():  # 对于每条测试样例
            text, label = data[int(idx)]
            if text[-1] not in stop_mark:  # 可能存在原始文本的最后一个词语不是分隔符
                text.append('.')
            masked_texts = []
            candi_texts = gpt_results[idx]['candi_texts']  # 5条
            # 先取第一条
            for candi_text in candi_texts:
                # 随机mask掉续写的第一个句子中的
                wrote_sent = candi_text.split(' ')[len(text)-1:]   # 取出续写部分
                wrote_sent = re.split(r"([.。!！?？；;])", ' '.join(wrote_sent))[0]  # 取出续写的第一条句子
                wrote_sent = np.array(wrote_sent.split(' ') + ['.'])
                # wrote_sent = np.array(wrote_sent.split(' ')[: min(10, len(wrote_sent))] + ['.'])  # 第一条句子可能会太长，截取前10个词
                masked_ps = random.sample(range(len(wrote_sent)), int(len(wrote_sent) * 0.25))  # mask 25%的
                wrote_sent[masked_ps] = args.mask_token
                wrote_sent = wrote_sent.tolist()
                # prompt插在续写的第一个句子前面
                masked_text = text + prompt_templates[label] + wrote_sent
                masked_texts.append(masked_text)
            outs[int(idx)] = {'label': label, 'text': tokenizer.tokenize(' '.join(text)), 'masked_texts': masked_texts}


    """2. 对于所有mask位置，进行补全"""
    for idx, (text, label) in enumerate(data):  # 对于每条测试样例
        masked_texts = outs[idx]['masked_texts']
        temp = []

        # dataloader, _ = dataset.transform_text0(masked_texts, labels=[label]*len(masked_texts), batch_size=args.batch_size)
        all_data, _ = dataset.transform_text(masked_texts, labels=[label]*len(masked_texts))
        sampler = SequentialSampler(all_data)
        dataloader = DataLoader(all_data, sampler=sampler, batch_size=args.batch_size)

        # 对每个batch
        for input_ids, input_mask, segment_ids, _ in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            with torch.no_grad():
                if args.target_model == 'roberta':
                    prediction_scores = generator(input_ids, input_mask).logits  # [batch_size,seq_len,vocab_size]
                else:
                    prediction_scores = generator(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)  # [b_size,128,30522]
                mask_ps = []  # list，每个list是该句子mask位置的所有索引 注：是对于处理完的input_ids来说的位置（经过token之后）
                for i in range(len(input_ids)):
                    if args.target_model == 'roberta':
                        m = np.where(input_ids[i].cpu().numpy() == tokenizer(args.mask_token)['input_ids'][1])[0].tolist()
                    else:
                        m = np.where(input_ids[i].cpu().numpy() == tokenizer.convert_tokens_to_ids([args.mask_token]))[0].tolist()
                    mask_ps.append(m)
                # 每个位置保留topk
                _, pred_ids = prediction_scores.topk(k=args.topk, dim=2, largest=True, sorted=True)  # [batch_size,seq_len,k)
                # 每个位置选择第Top k的
                # pred_ids = pred_ids[:,:,2].unsqueeze(2) # top 3的
                # pred_ids = pred_ids[:,:,0].unsqueeze(2) # top 1的  #######删

                for i in range(len(input_ids)):  # 对于每条mask数据
                    mp_a = mask_ps[i]  # 对于该数据，mask的位置
                    # 存在无mask位置的情况
                    if len(mp_a) == 0:
                        continue
                    input_ids_a = input_ids[i]  # 当前数据
                    num_padding = len(input_mask[i]) - torch.sum(input_mask[i])  # 当前数据中padding位置的个数
                    candi_ids = pred_ids[i, mp_a[:], :] #.cpu().numpy().tolist()  # 对于该数据，mask位置的候选集

                    if args.word_emb:
                        """每个位置从topk里选择word embedding最相近的"""
                        candi_ids = candi_ids.cpu().numpy().tolist()
                        ori_words = outs[idx]['masked_ori_words'][i]
                        ori_word_embeddings = []
                        oov_ps = []
                        for o in range(min(len(mp_a), len(ori_words))):
                            ow = ori_words[o]
                            if ow not in args.glove_model.keys():  # 可能原始词语不在glove词汇表中
                                ori_word_embeddings.append([0.1]*200)
                                oov_ps.append(o)
                            else:
                                ori_word_embeddings.append(args.glove_model[ow])
                        ori_word_embeddings = torch.tensor(np.array(ori_word_embeddings)).repeat(1,args.topk).reshape(-1,args.topk,200) # 200是word embedding维度
                        candi_word_embeddings = []
                        for ci in candi_ids: # 对于每个位置的候选词
                            candi_words_a = tokenizer.convert_ids_to_tokens(ci)
                            # candi_word_embeddings_a = list(itemgetter(*candi_words_a)(args.glove_model))
                            candi_word_embeddings_a = []
                            for c in range(len(candi_words_a)):
                                cw = candi_words_a[c]
                                if cw not in args.glove_model.keys():  # 可能候选词语不在glove词汇表中
                                    candi_word_embeddings_a.append([0.1] * 200)  # 随便给了个emb，可能有问题
                                else:
                                    candi_word_embeddings_a.append(args.glove_model[cw])
                            candi_word_embeddings.append(candi_word_embeddings_a)
                        candi_word_embeddings = torch.tensor(np.array(candi_word_embeddings))
                        sim_cos = torch.cosine_similarity(ori_word_embeddings, candi_word_embeddings, dim=2)
                        # sim_cos = torch.sqrt(torch.sum((ori_word_embeddings - candi_word_embeddings) ** 2, dim=2))

                        sim_cos_max_id = torch.argmax(sim_cos, dim=1)
                        sim_cos_max_id[oov_ps] = 0  # 对于原词是oov的，直接取top1
                        candi_ids = np.array(candi_ids)[range(len(candi_ids)), sim_cos_max_id].reshape(len(candi_ids), 1)
                    else:
                        candi_ids = candi_ids[:, 0].cpu().numpy().reshape(len(candi_ids), 1)  # 保存每个位置的第一个候选
                        # candi_ids = candi_ids.cpu().numpy().tolist()  # 所有位置的所有候选都保存

                    candi_comb = list(itertools.product(*candi_ids))  # 对于该数据，所有mask位置候选集的可能组合
                    input_ids_aa = input_ids_a.unsqueeze(0).repeat(len(candi_comb), 1)  # 当前数据的所有生成候选数据
                    # 为所有候选数据，完成替换
                    for j in range(len(input_ids_aa)): # 对于每条候选数据
                        input_ids_aa[j][mp_a] = torch.tensor(list(candi_comb[j])).cuda()
                        # 注意：只保存和输入相对应的输出
                        # temp = input_ids_aa[j][1+6: (args.max_seq_length - num_padding) - 1].cpu().numpy().tolist()
                        # 删掉首尾的[CLS]，padding，[SEP]，以及prompt
                        tt = input_ids_aa[j][1: (args.max_seq_length - num_padding) - 1].cpu().numpy().tolist()
                        para_sents = tokenizer.convert_ids_to_tokens(tt)
                        if args.target_model == 'roberta':
                            # 将带G的分词复原
                            para_sents = tokenizer.convert_tokens_to_string(para_sents)
                            pretok_sent = para_sents.replace(' '.join(prompt_templates[label]), '').strip().replace('  ', ' ').split()
                        else:
                            para_sents = ' '.join(para_sents).replace(' '.join(prompt_templates[label]),'').strip().replace('  ', ' ').split()
                            pretok_sent = para_sents
                            # 带##的分词复原
                            # pretok_sent = ""
                            # for tok in para_sents:
                            #     if tok.startswith("##"):
                            #         pretok_sent += tok[2:]
                            #     else:
                            #         pretok_sent += " " + tok
                            # pretok_sent = pretok_sent.split()
                        temp.append({'masked_pos': mp_a, 'para_texts': pretok_sent})
        outs[idx]['para'] = temp

    # 选择写不写入文件
    # lpt = True if args.load_prompt_trained else False
    # we = True if args.word_emb else False
    # with open('/pub/data/huangpei/TextFooler/prompt_results/outs_%s_%s_%s_%s_%.2f_%d_%d_we%s_lpt%s_phead.json' %
    #           (args.task, args.target_model, args.attack_level, args.mask_mode, args.mask_ratio, args.sample_size,
    #            args.topk, we, lpt), 'w') as fw:
    #     json.dump(outs, fw, indent=4)

    return outs


def main():
    np.random.seed(10)

    print('Whether use sym candidates: ', args.sym)

    if args.cal_ppl:
        cal_ppl_bygpt2()
        exit(0)

    if os.path.exists(args.adv_path) and os.listdir(args.adv_path):
        print("Output directory ({}) already exists and is not empty.".format(args.adv_path))
    else:
        os.makedirs(args.adv_path, exist_ok=True)

    """Get data to attack"""
    if args.train_set:
    # train set
        texts = args.datasets.train_seqs2
        texts = [[args.inv_full_dict[word] for word in text]for text in texts]  # 网络输入是词语
        labels = args.datasets.train_y
        data = list(zip(texts, labels))
        data = data[:int(0.25*(len(labels)))]  # 训练集取前25%攻击
    else:
    # test set
        if args.attack_level == 'word':
            # texts = args.datasets.test_seqs2
            # texts = [[args.inv_full_dict[word] for word in text] for text in texts]  # 网络输入是词语
            # labels = args.datasets.test_y
            # data = list(zip(texts, labels))
            # data = data[:200]  # 测试集取前1000条攻击

            test_x, test_y = load_test_data()
            data = list(zip(test_x[:200], test_y[:200]))

        elif args.attack_level == 'sent':
            # 待攻击的数据是gpt续写产生的
            with open('/pub/data/huangpei/TextFooler/prompt_results/gpt12_beamSearch_%s_comma.json' % (args.task), 'r') as fr:
                gpt_results = json.load(fr)
            texts = []
            labels = []
            for idx in gpt_results.keys():  # 对于每条测试样例
                candi_text = gpt_results[idx]['candi_texts'][0]  # 5条，暂时取第一条
                texts.append(candi_text.split(' '))
                labels.append(gpt_results[idx]['label'])
            # 加载对应词性标注结果
            with open('/pub/data/huangpei/TextFooler/prompt_results/pos_gpt12_beamSearch_%s_comma.pkl' % args.task, 'rb') as fp:
                args.pos_tags = pickle.load(fp)
            data = list(zip(texts, labels))


    print("Data import finished!")
    print('Attaked data size', len(data))

    # clean_gpt_generate(data)
    # exit(0)

    if args.gpt_generate:
        gpt_generate(data, args)
        exit(0)

    """construct the model"""
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
            dataset = Dataset_LSTM_snli(args)
        else:
            model = LSTM(args).cuda()
            checkpoint = torch.load(args.target_model_path, map_location=args.device)
            model.load_state_dict(checkpoint)
            dataset = Dataset_LSTM(args)
    elif args.target_model == 'bert':
        if args.task == 'snli':
            model = BERT_snli(args).cuda(args.rank)
            checkpoint = torch.load(args.target_model_path, map_location=args.device)
            model.load_state_dict(checkpoint)
            tokenizer = None
        else:
            tokenizer = BertTokenizer.from_pretrained(args.target_model_path, do_lower_case=True)  # 用来保存模型
            model = BERT(args).cuda(args.rank)
            # checkpoint = torch.load(args.target_model_path + '/pytorch_model.bin', map_location=device)
            # model.model.load_state_dict(checkpoint)
        dataset = Dataset_BERT(args)
    elif args.target_model == 'roberta':
        model = ROBERTA(args).cuda(args.rank)
        checkpoint = torch.load(args.target_model_path+'/pytorch_model.bin')
        model.load_state_dict(checkpoint)
        args.pad_token_id = model.encoder.config.pad_token_id
        tokenizer = None
        dataset = Dataset_ROBERTA(args)

    # if args.target_model == 'wordLSTM':
    #     model = Model(args, args.max_seq_length, args.embedding, nclasses=args.nclasses).cuda()
    #     checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
    #     print('Load model from: %s' % args.target_model_path)
    #     model.load_state_dict(checkpoint)
    # elif args.target_model == 'bert':
    #     model = NLI_infer_BERT(args, args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length).cuda()
    #     print('Load model from: %s' % args.target_model_path)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    # 待攻击网络
    predictor = model.text_pred()
    # 补写网络
    if args.load_prompt_trained:
        generator = BertForMaskedLM.from_pretrained('/pub/data/huangpei/TextFooler/models/bert/%s' % (args.task) + '_prompt_phead').cuda()  ##############
    else:
        if args.target_model == 'bert':
            generator = BertForMaskedLM.from_pretrained('bert-base-uncased').cuda()
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            generator = RobertaForMaskedLM.from_pretrained(args.model_path + 'roberta-base').cuda()
            tokenizer = RobertaTokenizer.from_pretrained(args.model_path + 'roberta-base')

    # dataset = NLIDataset_BERT('bert-base-uncased', max_seq_length=args.max_seq_length, batch_size=args.batch_size)


    """生成候选集"""
    print("Generate candidate adversarial examples using prompt...")
    outs = mask_and_fill(data, predictor, generator, tokenizer, dataset)

    """攻击"""
    print("Attack...")
    pred_true = 0
    attack_success = 0
    no_para = 0

    adv_outs = {}
    for idx in outs.keys(): # 对于每条测试样例
        # 1.对原始数据进行预测
        orig_probs = predictor([outs[idx]['text']], [outs[idx]['text']]).squeeze()
        pred_label = torch.argmax(orig_probs)
        if pred_label != outs[idx]['label']:
            continue
        else:
            # 2.对预测正确的，判断是否攻击成功
            pred_true += 1
            # 获得所有的改写
            para_texts = [o['para_texts'] for o in outs[idx]['para']]
            if len(para_texts) > 0:
                para_probs = predictor(para_texts, para_texts)
                para_labels = torch.argmax(para_probs, dim=1)
                # if torch.sum(torch.eq(para_labels, target_label))>0:
                if torch.sum(torch.ne(para_labels, outs[idx]['label']))>0:
                    attack_success+=1
                    # 保存对抗样本
                    # print(para_texts)
                    if para_labels.size()[0] == 1:  # 只有一条改写
                        adv_texts = para_texts
                    else:
                        adv_texts = np.array(para_texts)[torch.ne(para_labels, outs[idx]['label']).cpu()].tolist()
                    adv_texts = [' '.join(at) for at in adv_texts]
                    adv_outs[idx] = {'label': outs[idx]['label'], 'text': ' '.join(outs[idx]['text']), 'adv_texts': adv_texts}
            else:
                no_para += 1

    # 选择写不写入文件
    lpt = True if args.load_prompt_trained else False
    we = True if args.word_emb else False
    fw = open(args.adv_path + '/pat_%s_%s_%s_%s_%.2f_%d_%d_we%s_lpt%s_phead_p0.json' %
              (args.task, args.target_model, args.attack_level, args.mask_mode, args.mask_ratio, args.sample_size,
               args.topk, we, lpt), 'w')
    json.dump(adv_outs, fw, indent=4)
    fw.close()
    # print(adv_outs)
    print('Model:', args.target_model_path)
    print('Acc: %.4f, Attack succ: %.4f, robust acc: %.4f' % (float(pred_true/len(data)), attack_success/pred_true, 1.0-(attack_success+len(data)-pred_true)/float(len(data))))
    # print(no_para)

    return

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    args.rank = 0
    main()
    exit(0)

    """统计不同attack方法的攻击成功数据差别"""
    # mask_mode = 'selectedPS' if args.task in ['mr','imdb'] else 'random'
    lpt = True if args.load_prompt_trained else False
    we = True if args.word_emb else False
    # attack_num = {'mr-wordLSTM':165, 'mr-bert':182, 'imdb-wordLSTM': 177, 'imdb-bert':182, 'snli-wordLSTM':161,'snli-bert':173,}
    file_tf = open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/test_set/tf_%s_%s_adv_sym%s_%d.json' %
                   (args.task, args.target_model, args.sym, args.split), 'r')
    file_sem = open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/sempso/test_set/org/sem_%s_%s_success.pkl'
                    % (args.task, args.target_model), 'rb')
    file_pat = open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/pat_%s_%s_%s_%s_%.2f_%d_%d_we%s_lpt%s_phead_p0.json' %
              (args.task, args.target_model, args.attack_level, args.mask_mode, args.mask_ratio, args.sample_size,
               args.topk, we, lpt), 'r')

    data = file_tf.readlines()
    suc_id_1 = []
    for line in data:
        d = json.loads(line)
        suc_id_1.append(str(d['idx']))
    input_list, true_label_list, output_list, success, change_list, num_change_list, success_time = pickle.load(file_sem)

    suc_id_2 = [str(s) for s in success]

    data3 = json.load(file_pat)
    suc_id_3 = list(data3.keys())

    com = set(suc_id_1).union(set(suc_id_2))
    print(suc_id_3)
    print(com)
    diffs1 = list(set(suc_id_3).difference(com))  # 在前不在后
    print(args.task, args.target_model, args.attack_level, 'PAT*:%s' % lpt)
    print(float(len(diffs1))/float(len(suc_id_3)))

    # print('Non-overlap in PAT\'s suc attack / attack num:', float(len(diffs1))/attack_num['%s-%s' % (args.task,args.target_model)])
    # print('Non-overlap in PAT\'s suc attack / suc attack of PAT:', float(len(diffs1))/len(suc_id_1))

