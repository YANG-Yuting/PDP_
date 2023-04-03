import json
import xmlrpc.client
import requests
import heapq
import pandas as pd
import numpy as np
import time

import OpenHowNet
hownet_dict_advanced = OpenHowNet.HowNetDict(init_babel=True)
# hownet_dict_advanced = OpenHowNet.HowNetDict(init_sim=True)

import jieba
import jieba.posseg as pseg

# 名词、形容词、动词、副词
pos_list_noun = ['n', 'an']
pos_list_adj = ['a']
pos_list_verb = ['v', 'vd', 'vn']
pos_list_adv = ['ad']
pos_list = ['n', 'an', 'a', 'v', 'vd', 'vn', 'ad']

cate_convert = {"社会生活": "社会", "医药健康": "健康",  "文体娱乐": "体育", "科技": "科技",  "财经商业": "财经",
                    "军事": "军事", "政治": "政治", "教育考试": "教育"}
label_convert = {"rumor": 0, "truth": 1, "uncertain": 2, "piyao": 3, "useless": 4}

syn_size = 5

"""For one-text's attack"""


# Candidates donot include word self
def get_syn_words(perturb_idx, pos_tag):
    # if isinstance(text, list):
    #     text = ''.join(text)

    neigbhours_list = []
    # text_list = jieba.lcut(text)
    # pos_result = pseg.cut(text)
    # pos_tag = {}
    # j = -1
    # for word, pos in pos_result:
    #     j += 1
    #     pos_tag[j] = {'word': word, 'pos': pos}
    # print(pos_tag)
    # print(len(pos_tag))
    # print(text_list)
    # print(len(text_list))
    # assert len(pos_tag) == len(text_list)
    # exit(0)

    for idx in perturb_idx:
        pos = pos_tag[idx]['pos']
        word = pos_tag[idx]['word']
        saved_syns = []
        if pos not in pos_list:
            neigbhours_list.append(saved_syns)
            continue

        # sem
        # if pos in pos_list_noun:
        #     pos = 'noun'
        # elif pos in pos_list_adj:
        #     pos = 'adj'
        # elif pos in pos_list_verb:
        #     pos = 'verb'
        # elif pos in pos_list_adv:
        #     pos = 'adv'
        # print(pos)
        # syns = hownet_dict_advanced.get_nearest_words(word, language='zh', K=5, pos=pos)

        # bable net
        syns = hownet_dict_advanced.get_synset(word)
        if len(syns) == 0:
            neigbhours_list.append([])
        else:
            for i in range(len(syns)):
                if syns[i].pos == pos:  # 保持同义词和原始词语词性相同
                    saved_syns = syns[i].zh_synonyms[: min(len(syns), syn_size)]  # 最多保留5个同义词
                    break
            neigbhours_list.append(saved_syns)

    return neigbhours_list


def TopK_texts(adv_bach, adv_probs, ture_lable, K):

    adv_bach_with_score = list()  # 每个adv text，对于目标标签的预测概率，升序排列
    for i in range(0, len(adv_bach)):
        SS = {'adv_text_id': i, 'score': adv_probs[i][ture_lable]}
        adv_bach_with_score.append(SS)

    adv_bach_with_score = heapq.nsmallest(min(K, len(adv_bach_with_score)), adv_bach_with_score, key=lambda e: e['score'])

    adv_bach1 = list() # 保留top k个
    for i in range(0, min(K, len(adv_bach_with_score))):
        adv_bach1.append(adv_bach[adv_bach_with_score[i]['adv_text_id']].copy())

    return adv_bach1


def Look_Ahead(ori_text, category, true_lable, adv_bach, text_syns, t, position_conf_score, predictor): #更新t位置之后的打分.
    position_conf_score1 = position_conf_score.copy()

    pos_len = len(position_conf_score)
    # sample_len = min(10, len(adv_bach))  # ？这个参数是为了什么？ 我暂时设置为1 tiz
    sample_len = 1
    adv_batch1 = list()
    record_list = list()
    last_Num = 0
    for i in range(t, pos_len):
        SS={
            'start': last_Num,
            'end': -1
        }
        for j in range(0, sample_len):        #考虑10个文本给后面打分
            for k in range(0, len(text_syns[position_conf_score[i]['pos']])):
                a_adv = adv_bach[j].copy()
                a_adv[position_conf_score[i]['pos']] = text_syns[position_conf_score[i]['pos']][k]
                adv_batch1.append(a_adv)
                last_Num += 1
        SS['end'] = last_Num
        record_list.append(SS)

    adv_probs = []
    # print('Predicting for %d data (look ahead)' % len(adv_batch1))
    for adv_text in adv_batch1:  # 耗时
        # s_time = time.time()
        prob_true = float(predictor.predict(''.join(adv_text), category))
        # e_time = time.time()
        # print('Time of a request for predicting: %.2f s' % (e_time - s_time))
        prob_rumor = 1.0 - prob_true
        a_probs = [prob_rumor, prob_true]
        adv_probs.append(a_probs)
    # adv_probs = predictor([ori_text for i in range(len(adv_batch1))], adv_batch1)
    adv_probs = np.array(adv_probs)
    pre_confidence = adv_probs[:, true_lable]

    for i in range(t, pos_len):
        position_conf_score1[i]['score'] = np.min(pre_confidence[record_list[i-t]['start']:record_list[i-t]['end']])

    position_conf_score1 = position_conf_score1[t:]
    position_conf_score1 = sorted(position_conf_score1, key=lambda e: e['score'], reverse=False)
    return position_conf_score[0:t]+position_conf_score1


def filt_best_adv(ori_text, category, true_lable, adv_bach, adv_labels, predictor):
    best_changeNum = 9999
    best_adv = None
    changeList = None
    for i in range(len(adv_bach)):
        changeNum = 0
        if adv_labels[i] != true_lable:
            tempList = list()
            for j in range(len(ori_text)):
                if ori_text[j] != adv_bach[i][j]:
                    tempList.append(j)
                    changeNum += 1
            if changeNum < best_changeNum:
                changeList = tempList
                best_changeNum = changeNum
                best_adv = adv_bach[i]
    # （没有可以么  tiz注释了
    #finetune一下
    # No_changed=True
    # while No_changed:
    #     adv_batch = list()
    #     for pos in changeList:
    #         adv_text = best_adv.copy()
    #         adv_text[pos] = ori_text[pos]
    #         adv_batch.append(adv_text)
    #     #=====判断
    #     adv_batch1 = adv_batch.copy()
    #     adv_probs = []
    #     for adv_text in adv_batch1:
    #         prob_true = float(predictor.predict(''.join(adv_text), category))
    #         prob_rumor = 1.0 - prob_true
    #         a_probs = [prob_rumor, prob_true]
    #         adv_probs.append(a_probs)
    #     # adv_probs = predictor([ori_text for i in range(len(adv_batch1))], adv_batch1)
    #     adv_probs = np.array(adv_probs)
    #
    #     pre_confidence = adv_probs[:, true_lable]
    #     adv_label = np.argmax(adv_probs, axis=1)
    #     Re = np.sum(adv_label != true_lable)
    #
    #     if Re == 0:
    #         No_changed = False
    #     else:
    #         i = np.argmin(pre_confidence)
    #         best_adv = adv_batch[i]
    #         del changeList[i]
    #         best_changeNum = best_changeNum - 1

    return best_adv.copy(), best_changeNum


def Pseudo_DP(ori_text, category, true_lable, text_syns, pertub_psts, predictor):
    """Apply PDP for a text"""

    if len(pertub_psts) == 0:
        print("No positions to perturb. Certificated Robustness!")
        return None, 0

    """获得位置重要性"""
    print('---Get position importance---')
    position_conf_score = list()  # 保存每个位置，单独替换，对于目标标签预测概率的最小值
    first_adv_bach = list()  # 保存每个位置，单独替换，获得的所有候选集
    first_adv_probs = None  # 保存每个位置，单独替换，模型的预测概率
    for i in range(len(pertub_psts)):
        # print('For position %d' % pertub_psts[i])
        adv_batch = list()
        for j in range(1, len(text_syns[pertub_psts[i]])):
            adv_tex = ori_text.copy()
            adv_tex[pertub_psts[i]] = text_syns[pertub_psts[i]][j]
            adv_batch.append(adv_tex)

        first_adv_bach = first_adv_bach + adv_batch
        adv_probs = []
        # print('Predicting for %d data (position importance)' % len(adv_batch))
        for adv_text in adv_batch:
            # s_time = time.time()
            prob_true = float(predictor.predict(''.join(adv_text), category))
            # e_time = time.time()
            # print('Time of a request for predicting: %.2f s' % (e_time - s_time))
            prob_rumor = 1.0 - prob_true
            a_probs = [prob_rumor, prob_true]
            adv_probs.append(a_probs)
        # adv_probs = [0.1, 0.9] * len(adv_batch)

        if first_adv_probs is None:
            first_adv_probs = adv_probs
        else:
            first_adv_probs.extend(adv_probs.copy())

        adv_probs = np.array(adv_probs)
        adv_label = np.argmax(adv_probs, axis=1)

        pre_confidence = adv_probs[:, true_lable]

        SS = {'pos': pertub_psts[i], 'score': np.min(pre_confidence)}
        position_conf_score.append(SS)

        Re = np.sum(adv_label != true_lable)
        if Re > 0:
            # print(np.where(adv_label != true_lable)[0])
            best_adv = adv_batch[np.where(adv_label != true_lable)[0][0]]
            # print(best_adv)
            # best_adv = adv_batch[np.where(adv_label != true_lable)[0][1]]
            # print(best_adv)
            return best_adv, 1
    if len(position_conf_score) < 2:
        print("certified Robustness at r=1")
        return None, 0
    position_conf_score = sorted(position_conf_score, key=lambda e: e['score'], reverse=False)

    """按照位置重要性，开始搜索替换"""
    print('---Begin search---')
    last_adv_bach = first_adv_bach
    last_adv_probs = first_adv_probs
    for t in range(1, len(position_conf_score)):
        # tiz
        if float(t)/len(ori_text) > 0.25:
            return None, 0

        # print('For time %d:' % t)
        last_adv_bach = TopK_texts(last_adv_bach, last_adv_probs, true_lable, K=10)  # 过滤保留打分好的 !!!!!!!!!K的设置
        position_conf_score = Look_Ahead(ori_text, category, true_lable, last_adv_bach, text_syns, t, position_conf_score, predictor)

        temp_adv_bach = list()    # 每条数据扩大r位置可替换词个数倍后的待测试样本
        for tex_id in range(0, len(last_adv_bach)):
            for i in range(1, len(text_syns[position_conf_score[t]['pos']])):
                adv_tex = last_adv_bach[tex_id].copy()
                adv_tex[position_conf_score[t]['pos']] = text_syns[position_conf_score[t]['pos']][i]
                temp_adv_bach.append(adv_tex)

        #=====预测=====
        last_adv_bach = temp_adv_bach
        temp_adv_probs = []
        # print('Predicting for %d data (for all generated candidates)' % len(temp_adv_bach))
        for a_text in temp_adv_bach:
            # s_time = time.time()
            prob_true = float(predictor.predict(''.join(a_text), category))
            # e_time = time.time()
            # print('Time of a request for predicting: %.2f s' % (e_time - s_time))
            prob_rumor = 1.0 - prob_true
            a_probs = [prob_rumor, prob_true]
            temp_adv_probs.append(a_probs)
        temp_adv_probs = np.array(temp_adv_probs)
        # temp_adv_probs = predictor([ori_text for i in range(len(temp_adv_bach))], temp_adv_bach)

        last_adv_probs = temp_adv_probs
        temp_adv_label = np.argmax(temp_adv_probs, axis=1)
        Re = np.sum(temp_adv_label != true_lable)
        if Re > 0:
            best_adv, changeNum = filt_best_adv(ori_text, category, true_lable, last_adv_bach, temp_adv_label, predictor)
            return best_adv, changeNum
        else:
            pass

    return None, 0

def launch_attack():
    """
    Attack
    """

    """1. Load data"""
    data_path = './data/new_test_edit.csv'
    data = pd.read_csv(data_path)

    """2. Load model"""
    predictor = xmlrpc.client.ServerProxy("http://www.newsverify.com:8018/")

    """3. Attack"""
    result_file = './data/adv_exps_rumor_syn5.json'
    f = open(result_file, 'a+', encoding='utf-8')
    outs = {}
    attack_num = 0
    succ_num = 0
    for index, row in data.iterrows():  # for each text
        # if index < 22:
        #     continue
        Start_time = time.time()
        ori_text = row['content']
        category = cate_convert[row['category']]
        true_label = label_convert[row['label']]
        if true_label != 0:  # 暂时只攻击谣言or事实
            continue
        # ori_text = jieba.lcut("河北省儿童医院心理行为科（残联救助定点单位）康复救助开始报名。")
        # category = "社会"
        # true_label = 1

        prob_true = float(predictor.predict(ori_text, category))
        End_time_0 = time.time()
        print('Time of a request for predicting: %.2f s' % (End_time_0-Start_time))
        prob_rumor = 1.0 - prob_true
        orig_probs = np.array([prob_rumor, prob_true])
        orig_label = np.argmax(orig_probs)  # 0 rumor 1 truth

        if orig_label == true_label:  # 只攻击原本模型可以正确判断为谣言的
            """Only attack data which is predicted with true label originally"""
            attack_num += 1
            print('Attack num: ', attack_num)
            """Get syn words"""
            pertub_psts = []  # positions which have syn words
            # pos tag
            pos_result = pseg.cut(ori_text)
            pos_tag = {}
            ori_text = []
            j = -1
            for word, pos in pos_result:
                j += 1
                pos_tag[j] = {'word': word, 'pos': pos}
                ori_text.append(word)
            # get syn
            length = len(ori_text)
            text_syns = get_syn_words([i for i in range(length)], pos_tag)
            for i in range(length):
                if ori_text[i] not in text_syns[i]:  # if original word not in syn set, add it
                    text_syns[i] = [ori_text[i]] + text_syns[i]
            for ii in range(length):
                if len(text_syns[ii]) > 1:  # syn set is not null
                    pertub_psts.append(ii)

            print("Apply Pseudo DP")
            print(ori_text)
            best_adv, best_r = Pseudo_DP(ori_text, category, true_label, text_syns, pertub_psts, predictor)
            End_time = time.time()
            if best_r > 0:
                print("Found an adversarial example. Time: %.2f" % (End_time - Start_time))
                print("Text length: %d  Changed: %d  %.4f" % (length, best_r, float(best_r) / length))
                print('Original text: %s' % ''.join(ori_text))
                print('Adv exp: %s' % ''.join(best_adv))
                if float(best_r) / length > 0.25:
                    continue
                succ_num += 1
                print('Succ num: %d / %d' % (succ_num, attack_num))
                outs[row['clue_id']] = {'content': ori_text, 'category': category, 'label': true_label,
                                        'adv_exp': best_adv, 'changed_ratio': float(best_r) / length}
                f.write(json.dumps(outs[row['clue_id']], ensure_ascii=False) + '\n')
                f.flush()

            else:
                print("Failed. Time: %.2f" % (End_time - Start_time))
    f.close()

if __name__ == '__main__':
    launch_attack()
