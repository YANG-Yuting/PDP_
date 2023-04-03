import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from config import args

task = 'imdb'
vocab_size = 50000
word_emb_path = args.data_path + '%s/embeddings_glove_%d.pkl.npy' % (task, vocab_size)
embedding_matrix = np.load(word_emb_path)
embedding_matrix = embedding_matrix.T  # (50001, 200)

with open(args.data_path + '%s/dataset_%d_has_punctuation.pkl' % (task, vocab_size), 'rb') as fh:
    dataset = pickle.load(fh)
with open(args.data_path + '%s/word_candidates_sense_all_has_punctuation.pkl'% task, 'rb') as fp:
    word_candidate = pickle.load(fp)
inv_full_dict = dataset.inv_full_dict
full_dict = dataset.full_dict

pos_list = ['noun', 'verb', 'adj', 'adv']
k = 5

for word_id in word_candidate.keys():
    print(word_id)
    all_syns = word_candidate[word_id]
    new_all_syns = all_syns.copy()
    for pos in pos_list:
        syn_pos = new_all_syns[pos]
        if len(syn_pos) == 0:
            continue
        else:
            ori_word_emb = embedding_matrix[word_id]  # 原始单词向量
            syn_pos_word_emb = embedding_matrix[syn_pos]
            sim = []
            for i in range(len(syn_pos)):
                a_sim = cosine_similarity([ori_word_emb, syn_pos_word_emb[i]])[0][1]
                sim.append([a_sim, syn_pos[i]])
            sim.sort(reverse=True)
            top_k = sim[:k]
            top_k_syn = [s[1] for s in top_k]
            word_candidate[word_id][pos] = top_k_syn
            # print(word_candidate[word_id][pos])

f = open(args.data_path + '%s/word_candidates_sense_top5_has_punctuation.pkl' % task, 'wb')
pickle.dump(word_candidate, f)
