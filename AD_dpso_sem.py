"""
attack for mr and imdb, models are wordLSTM and bert-MIT
"""

from __future__ import division
import pickle
import torch
import time
import numpy as np
from BERT.tokenization import BertTokenizer

from attack_dpso_sem import PSOAttack, PSOAttack_snli
from train_model import load_test_data
from dataset import Dataset_LSTM_ascc_imdb
from models import LSTM, ESIM, BERT, BERT_snli, ROBERTA, EnsembleBERT
import os
from config import args

def main():
    """Load data"""
    test_x, test_y = load_test_data()
    test_x, test_y = test_x[:200], test_y[:200]  # Select 200 test data for attacking

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
            model = LSTM(args).to(device)
            checkpoint = torch.load(args.target_model_path, map_location=args.device)
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
        if args.task == 'snli':
            model = BERT_snli(args).cuda(args.rank)
            checkpoint = torch.load(args.target_model_path +'/pytorch_model.bin', map_location=args.rank)
            model.load_state_dict(checkpoint)
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            if args.kind == 'Ensemble':
                model = EnsembleBERT(args).cuda(args.rank)
            else:
                model = BERT(args).cuda(args.rank)
            # checkpoint = torch.load(args.target_model_path+'/pytorch_model.bin', map_location=args.device)
            # model.model.load_state_dict(checkpoint)
    elif args.target_model == 'roberta':
        model = ROBERTA(args).cuda(args.rank)
        checkpoint = torch.load(args.target_model_path+'/pytorch_model.bin')#, map_location=args.rank)
        model.load_state_dict(checkpoint)
        args.pad_token_id = model.encoder.config.pad_token_id

    model.eval()
    predictor = model.text_pred()

    print('Whether use sym candidates: ', args.sym)


    pop_size = 60
    if args.task == 'snli':
        adversary = PSOAttack_snli(args, predictor, pop_size=pop_size, max_iters=20)
    else:
        adversary = PSOAttack(args, predictor, pop_size=pop_size, max_iters=20)


    # 不合法数据
    wrong_clas_id = []  # 保存错误预测的数据id
    wrong_clas = 0  # 记录错误预测数据个数

    attack_list = []  # 记录待攻击样本id（整个数据集-错误分类的-长度不合法的）

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

    predict_true = 0

    print('Start attacking!')

    for idx, (text_word, true_label) in enumerate(zip(test_x, test_y)):  # for each data, text word
        print('text id: ', idx)
        text_ids = [args.full_dict[w] if w!= '</s></s>' else -1 for w in text_word]  # word to id (for snli, input has '[SEP]', replace it with -1 for flag.)
        orig_probs = predictor([text_word], [text_word]).squeeze()
        orig_label = torch.argmax(orig_probs)
        if orig_label != true_label:
            wrong_clas += 1
            wrong_clas_id.append(idx)
            print('wrong classifed ..')
            print('--------------------------')
            continue

        predict_true += 1
        """Filter sents too long"""
        # if x_len >= args.max_seq_length:
        #     too_long_id.append(i)
        #     print('skipping too long input..')
        #     print('--------------------------')
        #     continue
        # if x_len < 100:
        #     i += 1
        #     print('skipping too short input..')
        #     print('--------------------------')
        #     continue

        attack_list.append(idx)
        target = 1 if orig_label == 0 else 0
        time_start = time.time()
        new_text, num_changed, modify_ratio = adversary.attack(np.array(text_ids), np.array(target))  # new_text: perturbed text (list of id)
        sep_idx = np.where(np.array(new_text) == -1)[0]
        if sep_idx != None:
            new_text[sep_idx[0]] = '[SEP]'  # replace -1 with [SEP] (for snli)
        time_end = time.time()
        adv_time = time_end - time_start

        if new_text is None:
            print('failed! time:', adv_time)
            failed_list.append(idx)
            failed_time.append(adv_time)
            failed_input_list.append([text_word, true_label])
            continue

        print('-------------')
        print('%d changed.' % int(num_changed))
        if modify_ratio > 0.25:
            continue
        # Present perturbed text to nn
        new_text = [args.inv_full_dict[ii] if ii != '[SEP]' else ii for ii in new_text]
        probs = predictor([text_word], [new_text]).squeeze()
        pred_label = torch.argmax(probs)
        if pred_label != target:
            continue
        print('Success! time:', adv_time)
        print('Changed num:', num_changed)

        success_count += 1
        true_label_list.append(true_label)
        input_list.append([text_word, true_label])
        output_list.append(new_text)
        success.append(idx)
        change_list.append(modify_ratio)
        num_change_list.append(num_changed)
        success_time.append(adv_time)
        print('Num of data: %d, num of predict true: %d, num of successful attack: %d' % (idx+1, predict_true, success_count))

    print("Attack num:", predict_true)
    print('Acc: %.4f, Attack succ: %.4f, robust acc: %.4f' % (float(predict_true / len(test_y)), float(success_count)/float(predict_true),
                                                              1.0-(success_count+wrong_clas)/float(len(test_y))))
    # exit(0)

    w_dir = args.adv_path + '/sempso'
    if args.train_set:
        w_dir += '/train_set/'
    else:
        w_dir += '/test_set/'
    w_dir += args.kind
    if not os.path.exists(w_dir):
        os.makedirs(w_dir)

    if args.train_set:
        with open(w_dir + '/sem_%s_%s_ifvalid.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((wrong_clas_id), f)
        with open(w_dir + '/sem_%s_%s_fail.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((failed_list, failed_input_list, failed_time), f)
        with open(w_dir + '/sem_%s_%s_success.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((input_list, true_label_list, output_list, success, change_list, num_change_list, success_time), f)
    else:
        with open(w_dir + '/sem_%s_%s_ifvalid.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((wrong_clas_id), f)
        with open(w_dir + '/sem_%s_%s_fail.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((failed_list, failed_input_list, failed_time), f)
        with open(w_dir + '/sem_%s_%s_success.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((input_list, true_label_list, output_list, success, change_list, num_change_list, success_time), f)

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    main()






