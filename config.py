import pickle
import argparse
import numpy as np
import glove_utils

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/data/yangyuting/TextFooler/data/", help="path of data")
parser.add_argument("--model_path", type=str, default="/pub/data/huangpei/PAT-AAAI23/TextFooler/models/", help="path of data")
parser.add_argument("--code_path", type=str, default="/data/yangyuting/WordRobust/", help="path of code/project")
parser.add_argument("--save_path", type=str, default="/data/yangyuting/WordRobust/", help="path of saved code/project")
parser.add_argument("--target_model_path", type=str, default=None)
parser.add_argument("--adv_path", type=str, default=None, help="path of saved adv examples")

# for textfooler
parser.add_argument("--USE_cache_path",type=str,default='',help="Path to the USE encoder cache.")
parser.add_argument("--sim_score_window",default=15,type=int,help="Text length or token number to compute the semantic similarity score")
parser.add_argument("--import_score_threshold",default=-1.,type=float,help="Required mininum importance score.")
parser.add_argument("--sim_score_threshold",default=0, type=float,help="Required minimum semantic similarity score.") # 0.7
parser.add_argument("--synonym_num",default=50,type=int,help="Number of synonyms to extract")
parser.add_argument("--perturb_ratio",default=0.,type=float,help="Whether use random perturbation for ablation study")

parser.add_argument("--task", type=str, default='mr', help="task name: mr/imdb/snli")
parser.add_argument("--target_model",type=str,default='lstm',help="Target models for text classification: fasttext, charcnn, word level lstm ""For NLI: InferSent, ESIM, bert-base-uncased")
parser.add_argument("--batch_size",default=128,type=int,help="Batch size to get prediction")

# for train
parser.add_argument("--cnn", action='store_true', help="whether to use cnn")
parser.add_argument("--lstm", action='store_true', help="whether to use lstm")
parser.add_argument("--max_epoch", type=int, default=100)
parser.add_argument("--d", type=int, default=150)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--hidden_size", type=int, default=150)
parser.add_argument("--depth", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.00005)
parser.add_argument("--lr_decay", type=float, default=0.0)
parser.add_argument("--cv", type=int, default=0)
parser.add_argument("--save_data_split", action='store_true', help="whether to save train/test split")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--mode", type=str, default='test', help='train, test')
parser.add_argument("--select_nonrobust", type=bool, default=False)


# for robust.py
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training') # rank
parser.add_argument("--sample_num", type=int, default=256)
parser.add_argument("--change_ratio", type=float, default=1.0, help='the percentage of changed words in a text while sampling')

parser.add_argument("--sym", action='store_true', help="if use the symmetric candidate")
parser.add_argument("--train_set", action='store_true',help='if attack train set')
parser.add_argument("--attack_robot", action='store_true',help='if attack robot classifier')
parser.add_argument("--kind", type=str, default='org', help='the model to be attacked')
parser.add_argument("--prompt_generate", action='store_true', help='use prompt and bert to generate candidate adversarial examples')
parser.add_argument("--prompt_level", type=str, default='word-level', help='word-level or sentence-level')
parser.add_argument("--mask_ratio", type=float, default=0.15, help='the ratio of words to be masked')
parser.add_argument("--sample_size", type=int, default=50, help='the num of candidate adversarial examples for each instance')
parser.add_argument("--topk", type=int, default=5, help='the num of candidate for each masked word')
parser.add_argument("--mask_mode", type=str, default='random', help='the way of where to mask')
parser.add_argument("--word_emb", action='store_true',help='whether add word embedding for the candidate word selection while prompting')
parser.add_argument("--gpt_generate", action='store_true',help='use gpt to write a following sentence')
parser.add_argument("--load_prompt_trained", action='store_true',help='load the bert trained via mask-and-fill for adversarial examples')
parser.add_argument("--cal_ppl", action='store_true',help='calculate the ppl of texts')
parser.add_argument("--group_mode", type=str, default='sent', help='the way of group')
parser.add_argument("--attack_level", type=str, default='word', help='word or sentence')

parser.add_argument("--emb_dim", type=int, default=200)  # ours 200, ascc 300 (for lstm)
parser.add_argument('--emb_out_dim', type=int, default=100, help='embedding_dim')  # ascc
parser.add_argument('--syn_num', type=int, default=20)
parser.add_argument('--sparse_weight', type=int, default=15)
parser.add_argument('--kl_start_epoch', type=int, default=0, help='')
parser.add_argument('--weight_adv', type=float, default=0, help='')
parser.add_argument('--weight_clean', type=float, default=1, help='')
parser.add_argument('--weight_kl', type=float, default=4, help='')
parser.add_argument('--smooth_ce', type=str, default='false', help='')
parser.add_argument('--ascc', type=bool, default=False, help='')
parser.add_argument('--num_labels', type=int, default=2) # for ROBERTA
parser.add_argument('--pad_token_id', type=int, default=0) # for ROBERTA
parser.add_argument('--sep_token', type=str, default='[SEP]')  # For snli: [SEP] for BERT, </s> ROBERTA
parser.add_argument('--mask_token', type=str, default='[MASK]')  #  [MASK] for BERT, <mask> ROBERTA

parser.add_argument('--split', type=int, default=0)  #  [MASK] for BERT, <mask> ROBERTA

# ensemble
parser.add_argument('--num_models', type=int, default=3)
parser.add_argument('--lamda', type=float, default=0)
parser.add_argument('--log_det_lamda', type=float, default=0)
parser.add_argument('--num_param', type=int, default=2363905)  # 109483778 为bert参数个数，2363905 为第0层self-attention的参数
parser.add_argument('--aux_weight', type=float, default=1)
parser.add_argument("--modify_attentions", action='store_true')

args = parser.parse_args()

if args.modify_attentions:
    args.num_param = 2363905
else:
    args.num_param = 109483778

args.word_embeddings_path = args.data_path + 'glove.6B/glove.6B.200d.txt'
if args.target_model_path is None:
    args.target_model_path = args.model_path + '%s/%s' % (args.target_model, args.task)
args.counter_fitting_embeddings_path = args.data_path + 'counter-fitted-vectors.txt'
args.counter_fitting_cos_sim_path = args.data_path + 'data/cos_sim_counter_fitting.npy'
# args.output_dir = args.data_path + 'adv_results'  # directory where the attack results will be written

seq_len_list = {'imdb': 256, 'mr': 128, 'snli': 128}  # imdb:256-->300
args.max_seq_length = seq_len_list[args.task]
sizes = {'imdb': 50000, 'mr': 20000, 'snli': 60000}
args.max_vocab_size = sizes[args.task]
nclasses_dict = {'imdb': 2, 'mr': 2, 'snli': 3}
args.num_labels = nclasses_dict[args.task]
if args.task == 'snli' and args.target_model == 'lstm':
    args.emb_dim = 300
if args.target_model in ['bert', 'roberta']:
    args.emb_dim = 768

if args.target_model == 'roberta':
    args.sep_token = '</s></s>'
    args.mask_token = '<mask>'

if args.task == 'imdb':
    with open(args.data_path + '%s/dataset_%d_has_punctuation.pkl' % (args.task, args.max_vocab_size), 'rb') as f:  # prompt攻击时使用
    # with open(args.data_path + '%s/dataset_%d_new.pkl' % (args.task, args.max_vocab_size), 'rb') as f: # textfooler攻击时使用 旧的、无符号
        args.datasets = pickle.load(f)
    if args.train_set:
        with open(args.data_path + '%s/pos_tags_train_has_punctuation.pkl' % args.task,'rb') as fp:
            args.pos_tags = pickle.load(fp)
    else:
        with open(args.data_path + '%s/pos_tags_test_has_punctuation.pkl' % args.task,'rb') as fp:  # prompt攻击时使用
        # with open(args.data_path + '%s/pos_tags_test_new.pkl' % args.task,'rb') as fp:  # textfooler攻击时使用
            args.pos_tags = pickle.load(fp)
    args.inv_full_dict = args.datasets.inv_full_dict
    args.full_dict = args.datasets.full_dict
    args.full_dict['<oov>'] = len(args.full_dict.keys())
    args.inv_full_dict[len(args.full_dict.keys())] = '<oov>'
elif args.task == 'mr':
    with open(args.data_path + '%s/dataset_%d.pkl' % (args.task, args.max_vocab_size), 'rb') as f:
        args.datasets = pickle.load(f)

    if args.train_set:
        with open(args.data_path + '%s/pos_tags.pkl' % args.task, 'rb') as fp:
            args.pos_tags = pickle.load(fp)
    else:
        with open(args.data_path + '%s/pos_tags_test.pkl' % args.task,'rb') as fp:
            args.pos_tags = pickle.load(fp)
    args.inv_full_dict = args.datasets.inv_full_dict
    args.full_dict = args.datasets.full_dict
    args.full_dict['<oov>'] = len(args.full_dict.keys())
    args.inv_full_dict[len(args.full_dict.keys())] = '<oov>'
elif args.task == 'snli':
    with open(args.data_path + 'snli/nli_tokenizer.pkl', 'rb') as fh:
        args.tokenizer = pickle.load(fh)
    with open(args.data_path + 'snli/all_seqs.pkl', 'rb') as fh:
        args.train, _, args.test = pickle.load(fh)
    if args.train_set:
        with open(args.data_path + 'snli/pos_tags.pkl', 'rb') as fp:
            args.pos_tags = pickle.load(fp)
    else:
        with open(args.data_path + 'snli/pos_tags_test.pkl', 'rb') as fp:
            args.pos_tags = pickle.load(fp)
    np.random.seed(3333)
    args.full_dict = {w: i for (w, i) in args.tokenizer.word_index.items()}
    args.inv_full_dict = {i: w for (w, i) in args.full_dict.items()}
    # tiz add oov and pad
    args.full_dict['<oov>'] = len(args.full_dict) #42391
    args.inv_full_dict[len(args.full_dict)-1] = '<oov>'
    args.full_dict['<pad>'] = len(args.full_dict)  # 42391
    args.inv_full_dict[len(args.full_dict) - 1] = '<pad>'
    # tiz 20210827 过滤掉空的数据（7459: ['<s>','</s2>']，创建数据时把词汇表之外的删了，所以会出现空的数据。所幸只有一条），按batch预测的时候会报错，而且空数据也没有意义
    # test set
    null_idx = [i for i in range(len(args.test['s2'])) if len(args.test['s2'][i]) <= 2]
    args.test['s1'] = np.delete(args.test['s1'], null_idx)
    args.test['s2'] = np.delete(args.test['s2'], null_idx)
    args.test['label'] = np.delete(args.test['label'], null_idx)
    # 删除首尾<s>、</s>；id转str
    args.test_s1 = [[args.inv_full_dict[w] for w in t[1:-1]] for t in args.test['s1']]
    args.test_s2 = [[args.inv_full_dict[w] for w in t[1:-1]] for t in args.test['s2']]
    # args.test_s1 = [t[1:-1] for t in args.test['s1']]
    # args.test_s2 = [t[1:-1] for t in args.test['s2']]
    args.test_labels = args.test['label']
    # if args.train_set:
    # train set
    null_idx = [i for i in range(len(args.train['s2'])) if len(args.train['s2'][i]) <= 2]
    args.train['s1'] = np.delete(args.train['s1'], null_idx)
    args.train['s2'] = np.delete(args.train['s2'], null_idx)
    args.train['label'] = np.delete(args.train['label'], null_idx)

    args.train_s1 = [[args.inv_full_dict[w] for w in t[1:-1]] for t in args.train['s1']]
    args.train_s2 = [[args.inv_full_dict[w] for w in t[1:-1]] for t in args.train['s2']]
    args.train_labels = args.train['label']

    # print('the length of test cases is:', len(args.test_s1))
    # print('the length of train cases is:', len(args.train_s1))


if args.sym:
    if args.task == 'imdb':
        with open(args.data_path + '%s/word_candidates_sense_top5_sym_has_punctuation.pkl' % args.task, 'rb') as fp:  # 我们过滤的同义词表
            args.word_candidate = pickle.load(fp)
        with open(args.data_path + '%s/candidates_train_sym_has_punctuation.pkl' % args.task, 'rb') as fp:
            candidate_bags_train = pickle.load(fp)
        with open(args.data_path + '%s/candidates_test_sym_has_punctuation.pkl' % args.task, 'rb') as fp:  # 注意imdb的换成有符号的了
            candidate_bags_test = pickle.load(fp)
        # 以下为旧版本加载的数据
        # with open(args.data_path + '%s/word_candidates_sense_top5_sym.pkl' % args.task, 'rb') as fp:
        #     args.word_candidate = pickle.load(fp)
        # with open(args.data_path + '%s/candidates_train_sym.pkl' % args.task, 'rb') as fp:
        #     candidate_bags_train = pickle.load(fp)
        # with open(args.data_path + '%s/candidates_test_sym.pkl' % args.task, 'rb') as fp:
        #     candidate_bags_test = pickle.load(fp)

    else:
        with open(args.data_path + '%s/word_candidates_sense_top5_sym.pkl' % args.task, 'rb') as fp:  # 我们过滤的同义词表
            args.word_candidate = pickle.load(fp)
        with open(args.data_path + '%s/candidates_train_sym.pkl' % args.task, 'rb') as fp:
            candidate_bags_train = pickle.load(fp)
        with open(args.data_path + '%s/candidates_test_sym.pkl' % args.task, 'rb') as fp:
            candidate_bags_test = pickle.load(fp)
else:
    if args.task == 'imdb':
        with open(args.data_path + '%s/word_candidates_sense_top5_has_punctuation.pkl' % args.task, 'rb') as fp:  # 我们过滤的同义词表
            args.word_candidate = pickle.load(fp)
        with open(args.data_path + '%s/candidates_train_has_punctuation.pkl' % args.task, 'rb') as fp:
            candidate_bags_train = pickle.load(fp)
        with open(args.data_path + '%s/candidates_test_has_punctuation.pkl' % args.task, 'rb') as fp:
            candidate_bags_test = pickle.load(fp)
    else:
        with open(args.data_path + '%s/word_candidates_sense_top5.pkl' % args.task, 'rb') as fp:  # 我们过滤的同义词表
            args.word_candidate = pickle.load(fp)
        with open(args.data_path + '%s/candidates_train.pkl' % args.task, 'rb') as fp:
            candidate_bags_train = pickle.load(fp)
        with open(args.data_path + '%s/candidates_test.pkl' % args.task, 'rb') as fp:
            candidate_bags_test = pickle.load(fp)
args.candidate_bags = {**candidate_bags_train, **candidate_bags_test}
args.tf_vocabulary = pickle.load(open(args.data_path + '%s/tf_vocabulary.pkl' % args.task, "rb"))

args.pos_list = ['NN', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

if args.word_emb:
    args.glove_model = glove_utils.loadGloveModel(args.word_embeddings_path)

