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

class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        # module_url = "https://hub.tensorflow.google.cn/google/universal-sentence-encoder-large"
        # self.embed = hub.Module(module_url)
        # Attention: Hub only includes newest version5 with tf2.0 which is not applicable, we store the version3 and load it.
        module_url = args.model_path + "universal-sentence-encoder-large-3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores


# Candidates donot include word self
def get_syn_words(idx, perturb_idx, text_str):
    neigbhours_list = []
    text_str = [word.replace('\x85', '') for word in text_str]

    if ' '.join(text_str) in args.candidate_bags.keys():  # seen data (from train or test dataset)
        # 获得候选集
        candidate_bag = args.candidate_bags[' '.join(text_str)]
        for j in perturb_idx:
            neghs = candidate_bag[text_str[j]].copy()
            neghs.remove(text_str[j])  # 同义词中删掉自己
            neigbhours_list.append(neghs)
    else:
        print('Time warning! Re-generate syn words for a text!')
        pos_tag = pos_tagger.tag(text_str)
        # str转id（同义词集是用id存的），有不存在于词汇表的，保留原str，之后分类需要
        text_ids = []
        for word_str in text_str:
            if word_str in args.full_dict.keys():
                text_ids.append(args.full_dict[word_str])  # id
            else:
                text_ids.append(word_str)  # str
        # 对于所有扰动位置
        for j in perturb_idx:
            word = text_ids[j]
            pos = pos_tag[j][1]  # 当前词语词性
            if isinstance(word, int) and pos in args.pos_list:
                if pos.startswith('JJ'):
                    pos = 'adj'
                elif pos.startswith('NN'):
                    pos = 'noun'
                elif pos.startswith('RB'):
                    pos = 'adv'
                elif pos.startswith('VB'):
                    pos = 'verb'
                neigbhours_list.append(args.word_candidate[word][pos].copy())  # 候选集
            else:
                neigbhours_list.append([])
        neigbhours_list = [[args.inv_full_dict[i] if isinstance(i, int) else i for i in position] for position in neigbhours_list]  # id转为str

    return neigbhours_list

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


def attack(idx, text_ls, true_label, predictor, stop_words_set, word2idx, sim_predictor=None, import_score_threshold=-1.,
           sim_score_threshold=0.5, sim_score_window=15):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls], [text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, 0, orig_label, orig_label, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        # sim_score_threshold = -np.inf
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # get importance score
        leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        # start_time = time.clock()
        leave_1_probs = predictor([text_ls for i in range(len(leave_1_texts))], leave_1_texts)   # [103,2]

        # use_time = (time.clock() - start_time)
        # print("Predict for replaced texts for a text: " , use_time)
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)  # [103]
        a = (leave_1_probs_argmax != orig_label).float()
        b = leave_1_probs.max(dim=-1)[0]
        c = torch.index_select(orig_probs, 0, leave_1_probs_argmax)
        d = b - c
        if len(leave_1_probs.shape) == 1:  # 说明该文本只有一个单词，增加一维
            leave_1_probs = leave_1_probs.unsqueeze(0)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + a * d).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))

        # ---替换同义词获得方式--- #
        # origion
        # words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        # synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)  # list of len(words_perturb_idx), each is a list with syn word
        # tiz
        perturb_idx = [idx for idx, word in words_perturb]
        try:
            synonym_words = get_syn_words(idx, perturb_idx, text_ls)
        except:
            return '', 0, 0, orig_label, orig_label, 0
        # synonym_words = [[args.inv_full_dict[j] for j in k] for k in synonym_word_ids]

        synonyms_all = []
        for idx, word in words_perturb:
            synonyms = synonym_words.pop(0)  # tiz
            if word in word2idx:
                # synonyms = synonym_words.pop(0)  # origion
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            # start_time = time.clock()
            new_probs = predictor([text_ls for i in range(len(new_texts))], new_texts)
            # use_time = (time.clock() - start_time)
            # print("Predict for candidate texts for a perturbed position in a text: " , use_time)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy((semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]

        # new_text_prime = [inv_full_dict[word] for word in text_prime]  # 网络输入是词语
        # new_probs = predictor([new_text_prime]).squeeze()
        # new_label = torch.argmax(new_probs)
        # if new_label != orig_label:

        # start_time = time.clock()
        attack_label = torch.argmax(predictor([text_ls for i in range(len(text_prime))], [text_prime]))
        # use_time = (time.clock() - start_time)
        # print("Predict for the final selected candidate text: " , use_time)

        modify_ratio = float(num_changed) / len(text_ls)
        return ' '.join(text_prime), modify_ratio, num_changed, orig_label, attack_label, num_queries


def attack_snli(idx, text_ls, true_label, predictor, stop_words_set, sim_predictor=None, import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls], [text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, 0, orig_label, orig_label, 0
    else:
        s1, s2 = ' '.join(text_ls).split(' %s ' % args.sep_token)
        s1, s2 = s1.split(' '), s2.split(' ')
        len_text = len(s2)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        # s2_str = [args.inv_vocab[i] for i in s2_id]
        pos_ls = criteria.get_pos(s2)

        # get importance score
        leave_1_texts = [s2[:ii] + ['<oov>'] + s2[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_texts_pred = [s1 + [args.sep_token] + s2_ for s2_ in leave_1_texts]  # whole input for predictor
        # leave_1_ids = [[args.vocab[i] for i in tex] for tex in leave_1_texts]
        leave_1_probs = predictor([text_ls]*len_text, leave_1_texts_pred)
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)  #[103]

        a = (leave_1_probs_argmax != orig_label).float()
        b = leave_1_probs.max(dim=-1)[0]
        c = torch.index_select(orig_probs, 0, leave_1_probs_argmax)
        d = b-c
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + a * d).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and s2[idx] not in stop_words_set:
                    words_perturb.append((idx, s2[idx]))
            except:
                print(idx, len(s2), import_scores.shape, s2, len(leave_1_texts))

        # ---替换同义词获得方式--- #
        # origion
        # words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        # synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)  # list of len(words_perturb_idx), each is a list with syn word
        # tiz
        perturb_idx = [idx for idx, word in words_perturb]
        synonym_words = get_syn_words(idx, perturb_idx, s2)
        # synonym_words = [[args.inv_vocab[j] for j in k] for k in synonym_word_ids]

        synonyms_all = []
        for idx, word in words_perturb:
            synonyms = synonym_words.pop(0)  # tiz
            if word in args.full_dict.keys():
                # synonyms = synonym_words.pop(0)  # origion
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = s2[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_texts_pred = [s1 + [args.sep_token] + s2_ for s2_ in new_texts]  # whole input for predictor
            # new_probs = predictor(new_texts, batch_size=batch_size)
            # new_ids = [[args.vocab[i] for i in tex] for tex in new_texts]
            new_probs = predictor([text_ls]*len(new_texts), new_texts_pred)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()

            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy((semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        # text_prime_ids = [args.vocab[i] for i in text_prime]
        attack_label = torch.argmax(predictor([text_ls], [s1 + [args.sep_token] + text_prime]))

        modify_ratio = float(num_changed) / len(s2)
        return ' '.join(s1 + [args.sep_token] + text_prime), modify_ratio, num_changed, orig_label, attack_label, num_queries


def main(data_x, data_y, split):
    args.rank = 0
    torch.cuda.set_device(args.rank)
    device = torch.device('cuda', args.rank)

    # Illegal data
    wrong_clas_id = []  # 保存错误预测的数据id
    wrong_clas = 0  # 记录错误预测数据个数
    # too_long_id = []  # 保存太长被过滤的数据id

    # attack_list = []  # 记录待攻击样本id（整个数据集-错误分类的-长度不合法的）

    failed_list = []  # 记录攻击失败数据id
    failed_time = []  # 记录攻击失败时间
    failed_input_list = []  # 记录攻击失败的数据及其实际标签

    input_list = []  # 记录成功攻击的输入数据
    output_list = []  # 记录成功攻击的对抗样本
    success = []  # 记录成功攻击的数据id
    change_list = []  # 记录成功攻击的替换比例
    true_label_list = []  # 记录成功攻击的数据真实label
    success_count = 0  # # 记录成功攻击数据个数
    num_change_list = []  # 记录成功攻击的替换词个数
    success_time = []  # 记录成功攻击的时间

    print('Whether use sym candidates: ', args.sym)
    if os.path.exists(args.adv_path) and os.listdir(args.adv_path):
        print("Output directory ({}) already exists and is not empty.".format(args.adv_path))
    else:
        os.makedirs(args.adv_path, exist_ok=True)


    """Load model"""
    # print("Build model and dataloader")
    # if args.target_model == 'lstm':
    #     if args.task == 'snli':
    #         model = ESIM(args).cuda()
    #         """Load from trained model with different para name"""
    #         checkpoint = torch.load(args.target_model_path, map_location=args.device)
    #         checkpoint_new = {}
    #         for name, value in checkpoint.items():
    #             name = name.replace('enc_lstm.', '')
    #             checkpoint_new[name] = value
    #         model.load_state_dict(checkpoint_new)
    #     else:
    #         checkpoint = torch.load(args.target_model_path, map_location=args.device)
    #         """orig"""
    #         model = LSTM(args).cuda()
    #         args.word2id = model.emb_layer.word2id
    #         model.load_state_dict(checkpoint)
    #
    #         """ASCC"""
    #         # with open('/home/huangpei/ASCC-main/temp/imdb_word2id.pickle', 'rb') as f:
    #         #     args.word2id = pickle.load(f)
    #         # args.word2id['<oov>'] = len(args.word2id.keys())
    #         # args.word2id['<pad>'] = 0
    #         # dataset_test = Dataset_LSTM_ascc_imdb(args)
    #         # args.max_seq_length = 300
    #         # """load model"""
    #         # model = LSTM_ascc(args).cuda()
    #         # print(checkpoint.keys())
    #         # checkpoint_emb_layer, checkpoint_linear_transform_embd_1, checkpoint_encoder, checkpoint_out = {}, {}, {}, {}
    #         # for name, value in checkpoint.items():
    #         #     if 'embedding.' in name:
    #         #         # name = name.replace('embedding.', '')
    #         #         checkpoint_emb_layer[name] = value
    #         #     if 'linear_transform_embd_1.' in name:
    #         #         name = name.replace('linear_transform_embd_1.', '')
    #         #         checkpoint_linear_transform_embd_1[name] = value
    #         #     if 'bilstm.' in name:
    #         #         name = name.replace('bilstm.', '')
    #         #         checkpoint_encoder[name] = value
    #         #     if 'hidden2label.' in name:
    #         #         name = name.replace('hidden2label.', '')
    #         #         checkpoint_out[name] = value
    #         # model.emb_layer.load_state_dict(checkpoint_emb_layer)
    #         # model.linear_transform_embd_1.load_state_dict(checkpoint_linear_transform_embd_1)
    #         # model.encoder.load_state_dict(checkpoint_encoder)
    #         # model.out.load_state_dict(checkpoint_out)
    #
    # elif args.target_model == 'bert':
    #     if args.task == 'snli':
    #         model = BERT_snli(args).cuda()
    #         checkpoint = torch.load(args.target_model_path, map_location=args.device)
    #         model.load_state_dict(checkpoint)
    #     else:
    #         model = BERT(args).cuda()
    #     print('Load model from: %s' % args.target_model_path)
    print("Build model and dataloader")
    tokenizer = None
    if args.target_model == 'lstm':
        if args.task == 'snli':
            model = ESIM(args).cuda(args.rank)
            """Load from trained model with different para name"""
            # checkpoint = torch.load(args.target_model_path, map_location=args.device)
            # checkpoint_new = {}
            # for name, value in checkpoint.items():
            #     name = name.replace('enc_lstm.', '')
            #     checkpoint_new[name] = value
            # model.load_state_dict(checkpoint_new)
            dataset_train = Dataset_LSTM_snli_ascc(args)
        else:
            model = LSTM(args).cuda(args.rank)
            checkpoint = torch.load(args.target_model_path, map_location=device)
            model.load_state_dict(checkpoint)
            args.word2id = model.emb_layer.word2id
            dataset_train = Dataset_LSTM_ascc(args)
            all_data_train = dataset_train.transform_text(train_x, train_y)
            if torch.cuda.device_count() > 1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(all_data_train,num_replicas=torch.cuda.device_count(), rank=args.rank)
                nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
                dataloader_train = DataLoader(all_data_train, sampler=train_sampler, pin_memory=True, num_workers=nw,batch_size=args.batch_size)  # batch_
            else:
                train_sampler = SequentialSampler(all_data_train)
                dataloader_train = DataLoader(all_data_train, sampler=train_sampler, batch_size=args.batch_size)
    elif args.target_model == 'bert':
        # if args.task == 'snli':
        #     model = BERT_snli(args).cuda(args.rank)
        #     checkpoint = torch.load(args.target_model_path +'/pytorch_model.bin', map_location=device)
        #     model.load_state_dict(checkpoint)
        # else:
        #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        #     if args.kind == 'Ensemble':
        #         if 'comp' in args.target_model_path or 'aw' in args.target_model_path:
        #             print('Load comp model')
        #             model = EnsembleBERT_comp(args).cuda(args.rank)
        #             checkpoint = torch.load(args.target_model_path + '/pytorch_model.bin')
        #             model.load_state_dict(checkpoint)
        #         else:
        #             model = EnsembleBERT(args).cuda(args.rank)
        #     else:
        #         model = BERT(args).cuda(args.rank)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        if args.kind == 'Ensemble':
            if 'comp' in args.target_model_path or 'aw' in args.target_model_path:
                print('Load comp model')
                model = EnsembleBERT_comp(args).cuda(args.rank)
                # for n,p in model.named_parameters():
                #     print(n)
                checkpoint = torch.load(args.target_model_path + '/pytorch_model.bin')
                # for it in checkpoint:
                #     print(it)
                model.load_state_dict(checkpoint)
            else:
                model = EnsembleBERT(args).cuda(args.rank)
        else:
            model = BERT(args).cuda(args.rank)
            # checkpoint = torch.load(args.target_model_path+'/pytorch_model.bin', map_location=args.device)
            # model.model.load_state_dict(checkpoint)
    elif args.target_model == 'roberta':
        model = ROBERTA(args).cuda(args.rank)
        checkpoint = torch.load(args.target_model_path+'/pytorch_model.bin', map_location=device)
        model.load_state_dict(checkpoint)
        args.pad_token_id = model.encoder.config.pad_token_id

    model.eval()
    predictor = model.text_pred()

    # test_acc = eval_model(model, test_x, test_y)
    # print('Acc for test set is: {:.2%}(#%d)'.format(test_acc,len(data_y)))
    # exit(0)

    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1


    cos_sim = None
    use = USE(args.USE_cache_path)

    # start attacking
    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []



    stop_words_set = criteria.get_stopwords()
    print('Start attacking!')
    predict_true = 0

    outs = {}

    if not os.path.exists(args.adv_path): os.makedirs(args.adv_path)
    w_path = args.adv_path + '/tf_%s_%s_adv_sym%s_%d.json' % (args.task, args.target_model, args.sym, split)
    if os.path.exists(w_path):
        os.remove(w_path)
        print('Clear existed content!')
    with open(w_path, 'a+') as f:
        for idx, (text_word, true_label) in enumerate(zip(data_x, data_y)):  # for each data
            s_time = time.time()
            # print(idx)
            time_start = time.time()
            if args.task == 'snli':
                new_text, modify_ratio, num_changed, orig_label, new_label, num_queries = attack_snli(idx, text_word, true_label, predictor, stop_words_set,
                                            sim_predictor=use, sim_score_threshold=args.sim_score_threshold, import_score_threshold=args.import_score_threshold, sim_score_window=args.sim_score_window)
            else:
                new_text, modify_ratio, num_changed, orig_label, new_label, num_queries = attack(idx, text_word, true_label, predictor, stop_words_set, word2idx,
                                            sim_predictor=use, sim_score_threshold=args.sim_score_threshold, import_score_threshold=args.import_score_threshold, sim_score_window=args.sim_score_window)

            time_end = time.time()
            adv_time = time_end - time_start

            if true_label != orig_label:
                orig_failures += 1
                wrong_clas += 1
                wrong_clas_id.append(idx)
                # print('Wrong classified!')
                continue
            else:
                predict_true += 1
                nums_queries.append(num_queries)

            if true_label == new_label:
                # print('failed! time:', adv_time)
                failed_list.append(idx)
                failed_time.append(adv_time)
                failed_input_list.append([text_word, true_label])
                continue

            if true_label != new_label:
                # tiz：20210823 为方便观察，再攻击代码内部过滤到替换比例过高的（原论文是没有过滤的，所以我是在外面统计的时候过滤的）
                if modify_ratio > 0.25:
                    continue
                # 对抗样本再输入模型判断
                probs = predictor([text_word], [new_text.split(' ')]).squeeze()
                pred_label = torch.argmax(probs)
                if true_label == pred_label:
                    continue
                # print('Success! time:', adv_time)
                # print('Modify ratio:', modify_ratio)
                # print('Changed num:', num_changed)
                success_count += 1
                true_label_list.append(true_label)
                input_list.append([text_word, true_label])
                output_list.append(new_text)
                success.append(idx)
                change_list.append(modify_ratio)
                num_change_list.append(num_changed)
                success_time.append(adv_time)
                # 保存对抗样本
                outs[idx] = {'idx': idx, 'text': ' '.join(text_word), 'label': true_label, 'num_changed':num_changed, 'modify_ratio': modify_ratio, 'adv_texts':[new_text]}
                # f.write(json.dumps(outs[idx]) + '\n')
                # f.flush()
            e_time = time.time()
            print('idx:%d, time: %.2f, Acc: %.4f, Attack succ: %.4f, robust acc: %.4f' % (idx, e_time-s_time,
            float(predict_true / (idx+1)), float(success_count) / float(predict_true),
            1.0 - (success_count + wrong_clas) / float(idx+1)))


    try:
        print('Model: %s, Train set: %s' % (args.target_model_path, args.train_set))
    except:
        pass
    print('Acc: %.4f, Attack succ: %.4f, robust acc: %.4f' % (float(predict_true / len(data_y)), float(success_count) / float(predict_true),
                                                              1.0 - (success_count + wrong_clas) / float(len(data_y))))


    # def np_encoder(object):
    #     if isinstance(object, np.generic):
    #         return object.item()
    #
    # with open(w_dir + '/tf_%s_%s_adv_sym%s.json' % (args.task, args.target_model, args.sym), 'w') as f:
    #     json.dump(outs, f, indent=4, default=np_encoder)

    return len(data_y), predict_true, success_count, wrong_clas


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3' #args.gpu_id
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

        test_x, test_y = test_x[:200], test_y[:200]  # Select 200 test data for attacking
        # num_split = 1
        # aa = int(len(test_y) / num_split)
        # print(aa * (args.split - 1), aa * args.split)
        # data_x, data_y = test_x[aa*(args.split-1):aa*args.split], test_y[aa*(args.split-1):aa*args.split]

        # data_x, data_y = test_x[:2], test_y[:2]
        data_x, data_y = test_x, test_y

    len_data, predict_true, success_count, wrong_clas = main(data_x, data_y, args.split)
    print('len_data, predict_true, success_count, wrong_clas:', len_data, predict_true, success_count, wrong_clas)


