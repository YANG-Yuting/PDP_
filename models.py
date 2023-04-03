import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import pickle
import modules
import dataloader  # for LSTM
from dataset import Dataset_BERT, Dataset_LSTM, Dataset_LSTM_ascc_imdb, Dataset_LSTM_snli, Dataset_ROBERTA,perturb_texts, gen_sample_multiTexts, perturb_FGWS
from BERT.modeling import BertForSequenceClassification, BertConfig, BertForMaskedLM
from pytorch_transformers import BertModel  # for snli-bert
from torch.utils.data import DataLoader, SequentialSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, RobertaModel
from transformers import BertTokenizer, BertConfig
from datasets import Dataset
from transformers import Trainer, TrainingArguments


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args

    def text_pred(self):

        """
        Used for test

        org: original predictor
        Enhance: Enhancement version
        adv:adv trained
        SAFER：SAFER
        FGWS：FGWS
        """
        call_func = getattr(self, 'text_pred_%s' % self.args.kind, 'Didn\'t find your text_pred_*.')
        return call_func

    # texts: list of list, word
    def text_pred_org(self, orig_texts, texts):
        # texts_dataloader = self.dataset.transform_text(texts, labels=[0]*len(texts))  # labels are only given for call
        if self.args.target_model in ['bert', 'roberta']:
            all_data, all_data_syn = self.dataset.transform_text(texts, labels=[0] * len(texts))
            sampler = SequentialSampler(all_data)
            texts_dataloader = DataLoader(all_data, sampler=sampler, batch_size=self.args.batch_size)
            # sampler_syn = SequentialSampler(all_data_syn)
            # texts_dataloader_syn = DataLoader(all_data_syn, sampler=sampler_syn, batch_size=self.args.batch_size)
        else:
            all_data = self.dataset.transform_text(texts, labels=[0]*len(texts))
            sampler = SequentialSampler(all_data)
            texts_dataloader = DataLoader(all_data, sampler=sampler, batch_size=self.args.batch_size)

        outs = []
        with torch.no_grad():
            for idx, (*x, y) in enumerate(texts_dataloader):
                output = self.forward(x)
                outs.append(output)
                # outs.append(F.softmax(output, dim=-1)) forward里做了softmax
        return torch.cat(outs, dim=0)

    """为之前集成多次对抗训练的方法写的"""
    # texts: list of list, word
    # def text_pred_Ensemble(self, orig_texts, texts):
    #     model_num = 4  # 原模型+3*对抗训练模型
    #     if self.args.task == 'mr' and self.args.target_model == 'bert':
    #         model_acc = torch.tensor([89.60, 83.97, 84.54, 83.13]).cuda() * 0.01
    #     elif self.args.task == 'imdb' and self.args.target_model == 'bert':
    #         model_acc = torch.tensor([93.68, 90.14, 90.42, 90.75]).cuda() * 0.01
    #     # model_weight = model_acc
    #     """归一化"""
    #     # mean_a = torch.mean(model_acc)
    #     # std_a = torch.std(model_acc)
    #     # model_weight = torch.nn.functional.softmax((model_acc - mean_a) / std_a)
    #     # print('Model\'s acc:', model_acc)
    #     # print('Model\'s weight:', model_weight)
    #     # Model's weight: [0.6989, 0.1015, 0.1234, 0.0761]
    #     """adaboost"""
    #     # model_wrong = 1-model_acc
    #     # model_weight = 0.5 * torch.log((1-model_wrong)/model_wrong)
    #     # print('Model\'s weight:', model_weight)
    #     # Model's weight: [1.0768, 0.8280, 0.8495, 0.7974]
    #
    #     if self.args.target_model in ['bert', 'roberta']:
    #         all_data, all_data_syn = self.dataset.transform_text(texts, labels=[0] * len(texts))
    #         sampler = SequentialSampler(all_data)
    #         texts_dataloader = DataLoader(all_data, sampler=sampler, batch_size=self.args.batch_size)
    #     else:
    #         all_data = self.dataset.transform_text(texts, labels=[0]*len(texts))
    #         sampler = SequentialSampler(all_data)
    #         texts_dataloader = DataLoader(all_data, sampler=sampler, batch_size=self.args.batch_size)
    #
    #     with torch.no_grad():
    #         probs_boost_all = []
    #         for idx, (*x, y) in enumerate(texts_dataloader):
    #             outs = []
    #             # origin
    #             output = self.forward(x)
    #             b_size = output.size()[0]
    #             outs.append(F.softmax(output, dim=-1))  # [128, 2]
    #             # print('---------', idx)
    #             # print(outs)
    #             # adv models
    #             for model in self.adv_models:
    #                 output = model.forward(x)
    #                 outs.append(F.softmax(output, dim=-1))
    #             # print(outs)
    #             outs = torch.cat(outs, dim=0)
    #
    #             label_num = outs.size()[1]
    #             probs = []
    #             for m in range(b_size):
    #                 probs.append(outs[m::b_size, :])
    #             probs = torch.cat(probs, dim=0)
    #             probs = probs.view(b_size, model_num, label_num)
    #             # print(probs)
    #
    #             """某标签的概率：预测值为该标签的模型比例"""
    #             # probs_boost = []
    #             # for l in range(label_num):
    #             #     num = torch.sum(torch.eq(torch.argmax(probs, dim=2), l), dim=1)  # 获得预测值的比例作为对应标签的概率
    #             #     prob = num.float() / float(model_num)
    #             #     probs_boost.append(prob.view(x[0].size()[0], 1))
    #             # probs_boost = torch.cat(probs_boost, dim=1)
    #             # print(probs_boost)
    #
    #             """某标签的概率：所有模型对于该标签预测概率的加权平均"""
    #             probs_boost = torch.mean(probs, dim=1)
    #
    #             """某标签的概率：所有模型对于该标签预测概率的加权平均"""
    #             # weight = model_weight.unsqueeze(1).repeat(1, label_num).unsqueeze(0).repeat(b_size, 1, 1)
    #             # probs = probs * weight
    #             # # print(probs)
    #             # probs_boost = torch.mean(probs, dim=1)
    #             # # print(probs_boost)
    #
    #
    #             probs_boost_all.append(probs_boost)
    #     probs_boost_all = torch.cat(probs_boost_all, dim=0)
    #     # print('+++')
    #     # print(probs_boost_all)
    #     return probs_boost_all

    # 集成多个模型，

    # texts: list of list, word
    # 和text_pred_org一样

    # 和org一样
    def text_pred_Ensemble(self, orig_texts, texts):
        # texts_dataloader = self.dataset.transform_text(texts, labels=[0]*len(texts))  # labels are only given for call
        if self.args.target_model in ['bert', 'roberta']:
            all_data, all_data_syn = self.dataset.transform_text(texts, labels=[0] * len(texts))
            sampler = SequentialSampler(all_data)
            texts_dataloader = DataLoader(all_data, sampler=sampler, batch_size=self.args.batch_size)
            # sampler_syn = SequentialSampler(all_data_syn)
            # texts_dataloader_syn = DataLoader(all_data_syn, sampler=sampler_syn, batch_size=self.args.batch_size)
        else:
            all_data = self.dataset.transform_text(texts, labels=[0]*len(texts))
            sampler = SequentialSampler(all_data)
            texts_dataloader = DataLoader(all_data, sampler=sampler, batch_size=self.args.batch_size)

        outs = []
        with torch.no_grad():
            for idx, (*x, y) in enumerate(texts_dataloader):
                output = self.forward(x)
                outs.append(output)
                # outs.append(F.softmax(output, dim=-1)) forward里做了softmax
        return torch.cat(outs, dim=0)

    # Enhanced by FPP
    # texts：list of list, word
    def text_pred_Enhance(self, orig_texts, texts):
        perturbed_texts = perturb_texts(self.args, orig_texts, texts, self.args.tf_vocabulary, change_ratio=0.2)
        Samples_x = gen_sample_multiTexts(self.args, orig_texts, perturbed_texts, self.args.sample_num, change_ratio=0.25)
        Sample_probs = self.text_pred_org(None, Samples_x)
        lable_mum=Sample_probs.size()[-1]
        Sample_probs=Sample_probs.view(len(texts), self.args.sample_num, lable_mum)
        probs_boost = []
        for l in range(lable_mum):
            num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)  # 获得预测值的比例作为对应标签的概率
            prob = num.float() / float(self.args.sample_num)
            probs_boost.append(prob.view(len(texts), 1))

        probs_boost_all = torch.cat(probs_boost, dim=1)
        return probs_boost_all

    def text_pred_SAFER(self, orig_texts, texts):
        Samples_x = gen_sample_multiTexts(self.args, orig_texts, texts, self.args.sample_num, change_ratio=1)
        Sample_probs = self.text_pred_org(None, Samples_x)
        lable_mum=Sample_probs.size()[-1]
        Sample_probs=Sample_probs.view(len(texts), self.args.sample_num, lable_mum)
        probs_boost = []
        for l in range(lable_mum):
            num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)
            prob = num.float() / float(self.args.sample_num)
            probs_boost.append(prob.view(len(texts), 1))
        probs_boost_all = torch.cat(probs_boost, dim=1)
        return probs_boost_all

    def text_pred_FGWS(self, orig_texts, texts):
        gamma1=0.5
        perturbed_texts = perturb_FGWS(self.args, orig_texts, texts, self.args.tf_vocabulary)
        pre_prob=self.text_pred_org(None, perturbed_texts)
        ori_prob=self.text_pred_org(None, texts)
        lable=torch.argmax(ori_prob, dim=1)
        index=torch.arange(len(texts)).cuda()
        D=ori_prob.gather(1, lable.view(-1, 1))-pre_prob.gather(1, lable.view(-1, 1))-gamma1

        probs_boost_all=torch.where(D > 0, pre_prob, ori_prob)
        #D=D.view(-1)
        #probs_boost_all=torch.ones_like(ori_prob)
        #probs_boost_all.index_put_((index,lable),0.5 - D)
        #probs_boost_all.index_put_((index,1-lable), 0.5+D)
        return probs_boost_all


class BERT_new(nn.Module):
    def __init__(self, args, tokenizer):
        super(BERT_new, self).__init__()
        self.args = args
        device = "cuda:0"
        self.model = BertForSequenceClassification.from_pretrained(args.target_model_path, num_labels=args.num_labels).to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.trainer = Trainer(model=self.model)
        self.dataset = Dataset_BERT(args)

    # def text_pred(self, orig_texts, texts):
    #     texts = [' '.join(x) for x in texts]
    #     labels = [0] * len(texts)
    #     dataset_test = Dataset.from_dict({'text': texts, 'label': labels})
    #     dataset_test = dataset_test.map(lambda e: self.tokenizer(e['text'], truncation=True, padding='max_length',
    #                                                         max_length=self.args.max_seq_length), batched=True)
    #     dataset_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    #
    #     predictions = self.trainer.predict(dataset_test)[0]
    #     predictions = torch.tensor(predictions).cuda()
    #
    #     return predictions

    def text_pred(self, orig_texts, texts):
        device = "cuda:0"
        all_data, all_data_syn = self.dataset.transform_text(texts, labels=[0] * len(texts))
        sampler = SequentialSampler(all_data)
        texts_dataloader = DataLoader(all_data, sampler=sampler, batch_size=self.args.batch_size)

        outs = []
        with torch.no_grad():
            for idx, (*x, y) in enumerate(texts_dataloader):
                input_ids, attention_mask, token_type_ids, label = x[0].to(device), x[1].to(device), x[2].to(device), y.to(device)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                    labels=label)[1]
                outs.append(output)
        return torch.cat(outs, dim=0)


class BERT(BaseModel):
    """BERT for mr/imdb"""

    def __init__(self, args):
        super(BERT, self).__init__(args)
        self.args = args
        self.dataset = Dataset_BERT(args)

        # self.parall()
        # if self.args.prompt_generate:  # 若希望用prompt生成对抗样本，需要的bert模型是BertForMaskedLM
        #     self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()
        #     if args.load_prompt_trained:
        #         self.model_gen = BertForMaskedLM.from_pretrained(pretrained_dir+'_prompt_10').cuda() ##############
        #     else:
        #         self.model_gen = BertForMaskedLM.from_pretrained('bert-base-uncased').cuda()
        # else:

        self.model = BertForSequenceClassification.from_pretrained(args.target_model_path, num_labels=args.num_labels).cuda()
        # self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.num_labels).cuda()

    def text_to_emb(self, *inputs):
        """
        :param inputs: 0 ids: [batch_size, seq_len], 1:segment_ids
        :return:
        """
        input_ids,  segment_ids = inputs

        embs = self.model.bert.embeddings(input_ids, segment_ids)
        return embs

    def emb_to_logit(self, *inputs):
        """
        :param inputs: 0: embs, 1, attention/input_mask
        :return:
        """
        """BertModel forward"""
        embs, input_mask = inputs
        input_mask = input_mask.cuda(self.args.rank)
        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers = self.model.bert.encoder(embs, extended_attention_mask, head_mask = [None] * 12)
        sequence_output = encoded_layers[-1]
        pooled_output = self.model.bert.pooler(sequence_output)
        """BertForSequenceClassification forward"""
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return logits

    def sent_emb(self, inputs):
        input_ids, input_mask, segment_ids = inputs[:3]
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        outputs = self.model.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        _, pooled_output = outputs
        pooled_output = self.model.dropout(pooled_output)
        return pooled_output

    """Used for train"""
    def forward(self, inputs):
        input_ids, input_mask, segment_ids = inputs[:3]
        input_ids = input_ids.cuda(self.args.rank)
        input_mask = input_mask.cuda(self.args.rank)
        segment_ids = segment_ids.cuda(self.args.rank)
        logits = self.model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        probs = nn.functional.softmax(logits, dim=-1)

        return probs

    # """Used for test"""
    # def text_pred(self):
    #     """
    #     org: original predictor
    #     Enhance: Enhancement version
    #     adv:adv trained
    #     SAFER：SAFER
    #     FGWS：FGWS
    #     """
    #     call_func = getattr(self, 'text_pred_%s' % self.args.kind, 'Didn\'t find your text_pred_*.')
    #     return call_func
    #
    #
    # # orig_texts, texts : a batch
    # def text_pred_org(self, orig_texts, texts):
    #     texts_dataloader = self.dataset.transform_text(texts, labels=[0]*len(texts))  # labels are only given for call
    #     outs = []
    #     with torch.no_grad():
    #         for *x, y in texts_dataloader:
    #             output = self.forward(x)
    #             outs.append(nn.functional.softmax(output, dim=-1))
    #     return torch.cat(outs, dim=0)
    #
    #
    #
    # def text_pred_adv(self, orig_texts, texts):
    #     texts_dataloader = self.dataset.transform_text(texts, labels=[0]*len(texts))  # labels are only given for call
    #     outs = []
    #     with torch.no_grad():
    #         for *x, y in texts_dataloader:
    #             output = self.forward(x)
    #             outs.append(F.softmax(output, dim=-1))
    #     return torch.cat(outs, dim=0)
    #
    # def text_pred_Enhance(self, orig_texts, texts):
    #     perturbed_texts = perturb_texts(self.args, orig_texts, texts, self.args.tf_vocabulary, change_ratio=0.2)
    #     Samples_x = gen_sample_multiTexts(self.args, orig_texts, perturbed_texts, self.args.sample_num, change_ratio=0.25)
    #     Sample_probs = self.text_pred_org(None, Samples_x)
    #     lable_mum=Sample_probs.size()[-1]
    #     Sample_probs=Sample_probs.view(len(texts), self.args.sample_num, lable_mum)
    #     probs_boost = []
    #     for l in range(lable_mum):
    #         num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)  # 获得预测值的比例作为对应标签的概率
    #         prob = num.float() / float(self.args.sample_num)
    #         probs_boost.append(prob.view(len(texts), 1))
    #
    #     probs_boost_all = torch.cat(probs_boost, dim=1)
    #     return probs_boost_all
    #
    # def text_pred_SAFER(self,orig_texts, texts):
    #     Samples_x = gen_sample_multiTexts(self.args, orig_texts, texts, self.args.sample_num, change_ratio=1)
    #     Sample_probs = self.text_pred_org(None, Samples_x)
    #     lable_mum=Sample_probs.size()[-1]
    #     Sample_probs=Sample_probs.view(len(texts), self.args.sample_num, lable_mum)
    #     probs_boost = []
    #     for l in range(lable_mum):
    #         num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)
    #         prob = num.float() / float(self.args.sample_num)
    #         probs_boost.append(prob.view(len(texts), 1))
    #     probs_boost_all = torch.cat(probs_boost, dim=1)
    #     return probs_boost_all
    #
    # def text_pred_FGWS(self, orig_texts, texts):
    #     gamma1=0.5
    #     perturbed_texts = perturb_FGWS(self.args, orig_texts, texts, self.args.tf_vocabulary)
    #     pre_prob=self.text_pred_org(None, perturbed_texts)
    #     ori_prob=self.text_pred_org(None, texts)
    #     lable=torch.argmax(ori_prob, dim=1)
    #     index=torch.arange(len(texts)).cuda()
    #     D=ori_prob.gather(1, lable.view(-1, 1))-pre_prob.gather(1, lable.view(-1, 1))-gamma1
    #
    #     probs_boost_all=torch.where(D > 0, pre_prob, ori_prob)
    #     #D=D.view(-1)
    #     #probs_boost_all=torch.ones_like(ori_prob)
    #     #probs_boost_all.index_put_((index,lable),0.5 - D)
    #     #probs_boost_all.index_put_((index,1-lable), 0.5+D)
    #     return probs_boost_all


class BERT_snli(BaseModel):
    """BERT for snli"""

    def __init__(self, args):
        super(BERT_snli, self).__init__(args)
        self.args = args
        self.dataset = Dataset_BERT(args)
        self.hidden_size = 768
        self.model = BertModel.from_pretrained('bert-base-uncased').cuda(args.rank)
        self.decoder = nn.Linear(self.hidden_size, args.num_labels)
        # if args.target_model_path != 'bert-base-uncased':
        #     checkpoint = torch.load(args.target_model_path, map_location=args.device)
        #     self.load_state_dict(checkpoint)

    """Used for train"""
    def forward(self, inputs):
        input_ids, input_mask, segment_ids = inputs[:3]
        input_ids = input_ids.cuda(self.args.rank)
        input_mask = input_mask.cuda(self.args.rank)
        segment_ids = segment_ids.cuda(self.args.rank)

        outputs = self.model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        encoded_layers = outputs[0]
        encode_out = encoded_layers[:, 0, :]
        logits = self.decoder(encode_out)
        probs = nn.functional.softmax(logits[:, [1, 0, 2]])

        return probs


    # def text_pred_Enhance(self, orig_texts, texts):
    #     orig_s2s, orig_s1s, s2s = [], [], []
    #     for orig_text, text in zip(orig_texts, texts):
    #         orig_s1 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[0]
    #         orig_s2 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[1]
    #         s2 = ' '.join(text).split(' %s ' % self.args.pad_token)[1]
    #         orig_s1s.append(orig_s1)
    #         orig_s2s.append(orig_s2)
    #         s2s.append(s2)
    #
    #     perturbed_s2s = perturb_texts(self.args, orig_s2s, s2s, self.args.tf_vocabulary, change_ratio=0.2)
    #     Samples_s2s = gen_sample_multiTexts(self.args, orig_s2s, perturbed_s2s, self.args.sample_num, change_ratio=0.25)
    #     Samples_s1s = []
    #     for s1 in orig_s1s:
    #         Samples_s1s.extend([s1]*self.args.sample_num)
    #     Samples_x = [s1 + [self.args.pad_token] + s2 for s1, s2 in zip(Samples_s1s, Samples_s2s)]
    #
    #     Sample_probs = self.text_pred_org(None, Samples_x)
    #     lable_mum = Sample_probs.size()[-1]
    #     Sample_probs = Sample_probs.view(len(texts), self.args.sample_num, lable_mum)
    #     probs_boost = []
    #     for l in range(lable_mum):
    #         num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)  # 获得预测值的比例作为对应标签的概率
    #         prob = num.float() / float(self.args.sample_num)
    #         probs_boost.append(prob.view(len(texts), 1))
    #
    #     probs_boost_all = torch.cat(probs_boost, dim=1)
    #     return probs_boost_all
    #
    # def text_pred_SAFER(self, orig_texts, texts):
    #     orig_s2s, orig_s1s, s2s = [], [], []
    #     for orig_text, text in zip(orig_texts, texts):
    #         orig_s1 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[0]
    #         orig_s2 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[1]
    #         s2 = ' '.join(text).split(' %s ' % self.args.pad_token)[1]
    #         orig_s1s.append(orig_s1)
    #         orig_s2s.append(orig_s2)
    #         s2s.append(s2)
    #     Samples_s2s = gen_sample_multiTexts(self.args, orig_s2s, s2s, self.args.sample_num, change_ratio=1)
    #     Samples_s1s = []
    #     for s1 in orig_s1s:
    #         Samples_s1s.extend([s1]*self.args.sample_num)
    #     Samples_x = [s1 + [self.args.pad_token] + s2 for s1, s2 in zip(Samples_s1s, Samples_s2s)]
    #
    #     Sample_probs = self.text_pred_org(None, Samples_x)
    #     lable_mum = Sample_probs.size()[-1]
    #     Sample_probs = Sample_probs.view(len(texts), self.args.sample_num, lable_mum)
    #     probs_boost = []
    #     for l in range(lable_mum):
    #         num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)
    #         prob = num.float() / float(self.args.sample_num)
    #         probs_boost.append(prob.view(len(texts), 1))
    #     probs_boost_all = torch.cat(probs_boost, dim=1)
    #     return probs_boost_all
    #
    # def text_pred_FGWS(self, orig_texts, texts):
    #     orig_s2s, orig_s1s, s2s = [], [], []
    #     for orig_text, text in zip(orig_texts, texts):
    #         orig_s1 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[0]
    #         orig_s2 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[1]
    #         s2 = ' '.join(text).split(' %s ' % self.args.pad_token)[1]
    #         orig_s1s.append(orig_s1)
    #         orig_s2s.append(orig_s2)
    #         s2s.append(s2)
    #
    #     gamma1 = 0.5
    #     perturbed_s2s = perturb_FGWS(self.args, orig_s2s, s2s, self.args.tf_vocabulary)
    #     Samples_x = [s1 + [self.args.pad_token] + s2 for s1, s2 in zip(orig_s1s, perturbed_s2s)]
    #     pre_prob = self.text_pred_org(None, Samples_x)
    #     ori_prob = self.text_pred_org(None, texts)
    #     lable = torch.argmax(ori_prob, dim=1)
    #     index = torch.arange(len(texts)).cuda()
    #     D = ori_prob.gather(1, lable.view(-1, 1))-pre_prob.gather(1, lable.view(-1, 1))-gamma1
    #
    #     probs_boost_all = torch.where(D > 0, pre_prob, ori_prob)
    #     #D=D.view(-1)
    #     #probs_boost_all=torch.ones_like(ori_prob)
    #     #probs_boost_all.index_put_((index,lable),0.5 - D)
    #     #probs_boost_all.index_put_((index,1-lable), 0.5+D)
    #     return probs_boost_all


class EnsembleBERT(BaseModel):
    def __init__(self, args):
        super(EnsembleBERT, self).__init__(args)
        self.dataset = Dataset_BERT(args)
        self.args = args
        # self.models = []
        if args.mode == 'train':
            # for i in range(args.num_models):
            #     self.model = BertForSequenceClassification.from_pretrained(args.target_model_path, num_labels=args.num_labels, output_attentions=True).cuda()
            #     self.models.append(self.model)
            if 'esb' in args.target_model_path:
                self.model0 = BertForSequenceClassification.from_pretrained(args.target_model_path + '_%d' % 0, num_labels=args.num_labels, output_attentions=True).cuda()
                self.model1 = BertForSequenceClassification.from_pretrained(args.target_model_path + '_%d' % 1, num_labels=args.num_labels, output_attentions=True).cuda()
                self.model2 = BertForSequenceClassification.from_pretrained(args.target_model_path + '_%d' % 2, num_labels=args.num_labels, output_attentions=True).cuda()
            else:
                self.model0 = BertForSequenceClassification.from_pretrained(args.target_model_path, num_labels=args.num_labels, output_attentions=True).cuda()
                self.model1 = BertForSequenceClassification.from_pretrained(args.target_model_path, num_labels=args.num_labels, output_attentions=True).cuda()
                self.model2 = BertForSequenceClassification.from_pretrained(args.target_model_path, num_labels=args.num_labels, output_attentions=True).cuda()
        else:
            # self.model0 = BertForSequenceClassification.from_pretrained(args.target_model_path+'_%d' % 0, num_labels=args.num_labels).cuda()
            # self.model1 = BertForSequenceClassification.from_pretrained(args.target_model_path + '_%d' % 1, num_labels=args.num_labels).cuda()
            # self.model2 = BertForSequenceClassification.from_pretrained(args.target_model_path + '_%d' % 2, num_labels=args.num_labels).cuda()

            self.model0 = BertForSequenceClassification.from_pretrained(args.target_model_path + '_%d' % 0, num_labels=args.num_labels, output_attentions=True).cuda()
            self.model1 = BertForSequenceClassification.from_pretrained(args.target_model_path + '_%d' % 1, num_labels=args.num_labels, output_attentions=True).cuda()
            self.model2 = BertForSequenceClassification.from_pretrained(args.target_model_path + '_%d' % 2, num_labels=args.num_labels, output_attentions=True).cuda()
            # for i in range(args.num_models):
            #     self.model = BertForSequenceClassification.from_pretrained(args.target_model_path + '_%d' % i, num_labels=args.num_labels, output_attentions=True).cuda()
            #     self.models.append(self.model)

        self.models = [self.model0, self.model1, self.model2]
        # self.models = [self.model0, self.model1]

    def forward(self, inputs):
        """Used for test"""
        input_ids, input_mask, segment_ids = inputs[:3]
        input_ids = input_ids.cuda(self.args.rank)
        input_mask = input_mask.cuda(self.args.rank)
        segment_ids = segment_ids.cuda(self.args.rank)
        """Type 1. 用多个模型logits的均值作为最终的预测"""
        # logits = 0
        # for i in range(self.args.num_models):
        #     _, lg = self.models[i](input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        #     logits += lg
        # logits = logits / float(self.args.num_models)
        # # if self.args.task == 'snli':
        # #     probs = nn.functional.softmax(logits[:, [1, 0, 2]], dim=-1)  # Attention: label transpose for snli
        # # else:
        # #     probs = nn.functional.softmax(logits, dim=-1)
        # probs = nn.functional.softmax(logits, dim=-1)

        """Type 2. 用多个模型预测label的投票作为最终label"""
        logits_all = []
        for i in range(self.args.num_models):
            _, logits = self.models[i](input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            logits_all.append(logits)
        logits_all = torch.cat(logits_all, dim=1)
        logits_all = logits_all.view(input_ids.size()[0], self.args.num_models, self.args.num_labels)  # batch_size,_,_

        probs_boost = []
        for l in range(self.args.num_labels):
            num = torch.sum(torch.eq(torch.argmax(logits_all, dim=2), l), dim=1)  # 获得几个集成模型的预测标签的比例作为对应标签的概率
            prob = num.float() / float(self.args.num_models)
            probs_boost.append(prob.view(input_ids.size()[0], 1))
        probs = torch.cat(probs_boost, dim=1)

        return probs

    # def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,labels=None):
    #     all_preds = []
    #     all_labels = []
    #     input_ids = input_ids.to(device)
    #     attention_mask = attention_mask.to(device)
    #     token_type_ids = token_type_ids.to(device)
    #     labels = labels.to(device)
    #     print(input_ids)
    #     for i in range(self.args.num_models):
    #         print(i)
    #         logits = self.model[str(i)](input_ids, attention_mask, token_type_ids)
    #         all_preds.append(logits)
    #         all_labels.append(labels)
    #     all_preds = torch.cat(all_preds, dim=0)
    #     all_labels = torch.cat(all_labels, dim=0)
    #     print(all_preds)
    #     return all_preds, all_labels


class AuxModel(nn.Module):
    def __init__(self):
        super(AuxModel, self).__init__()
        # self.dense = nn.Linear(args.max_seq_length+args.num_models, args.num_param)
        # self.dense = nn.Linear(1, args.num_param)
        self.dense = nn.Linear(1+768, 768)

    def forward(self, x):  # [b_size, seq_len, hid_size] --> [b_size, seq_len, hid_size+1]
        # input_tensor = torch.cat((input_ids, model_nos), axis=1)
        # input_tensor = model_nos.float()
        # input_tensor = torch.cat([model_nos.float(), hidden_states], -1)
        # print(input_tensor)
        hidden_states = self.dense(x)
        # outs = torch.nn.GELU()(hidden_states)  # 调试替换成别的
        outs = torch.nn.functional.sigmoid(hidden_states)
        return outs


class EnsembleBERT_comp(BaseModel):
    def __init__(self, args):
        super(EnsembleBERT_comp, self).__init__(args)
        self.dataset = Dataset_BERT(args)
        self.args = args
        self.bert = BertForSequenceClassification.from_pretrained(args.target_model_path, num_labels=args.num_labels, output_attentions=True).cuda()
        # self.aux = AuxModel().cuda()

    def forward(self, inputs):
        """Used for test"""
        input_ids, input_mask, segment_ids, _ = inputs[:4]
        input_ids = input_ids.cuda(self.args.rank)
        input_mask = input_mask.cuda(self.args.rank)
        segment_ids = segment_ids.cuda(self.args.rank)

        ensemble = 'avg'  # 'vote'
        if ensemble == 'avg':
            """Type 1. 用多个模型logits的均值作为最终的预测"""
            logits = 0
            for i in range(self.args.num_models):
                _, lg = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, model_ids=i)
                logits += lg
            logits = logits / float(self.args.num_models)
            probs = nn.functional.softmax(logits, dim=-1)
        elif ensemble == 'vote':
            """Type 2. 用多个模型预测label的投票作为最终label"""
            logits = []
            for i in range(self.args.num_models):
                # model_no = torch.tensor([i], dtype=torch.float).cuda(self.args.rank)
                # params_aux = self.aux(model_no)
                # s_index = 0  # 在辅助网络输出中的索引
                # param_org_bert = {}
                # for name, param in self.bert.named_parameters():
                #     # print('name', name)
                #     param_org = param.clone()  # param_org与param不共享内存
                #     param_org_bert[name] = param_org
                #     if self.args.modify_attentions:  # 只修改第0层self-attention部分
                #         if 'encoder.layer.0.attention' not in name:
                #             continue
                #     param_num = param.numel()  # 参数个数
                #     param.data.add_(self.args.aux_weight * params_aux[s_index: s_index + param_num].reshape(param.shape))  # bert的参数随之改变
                #     s_index += param_num
                # _, lg = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                _, lg = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, model_ids=i)
                logits.append(lg)
                # for name, param in self.bert.named_parameters():
                #     param.data = param_org_bert[name]
                #     param.data.requires_grad = False
            logits = torch.cat(logits, dim=1)  # 注意这里和训练阶段不同，是竖着拼接 [batch_size, num_classes*num_models]
            logits = logits.view(input_ids.size()[0], self.args.num_models, self.args.num_labels)  # batch_size, num_classes, num_models

            probs_boost = []
            for l in range(self.args.num_labels):
                num = torch.sum(torch.eq(torch.argmax(logits, dim=2), l), dim=1)  # 获得几个集成模型的预测标签的比例作为对应标签的概率
                prob = num.float() / float(self.args.num_models)
                probs_boost.append(prob.view(input_ids.size()[0], 1))
            probs = torch.cat(probs_boost, dim=1)

        return probs


class LSTM(BaseModel):
    """LSTM for mr/imdb"""

    def __init__(self, args):
        super(LSTM, self).__init__(args)
        # self.args = args
        self.dataset = Dataset_LSTM(args)
        self.norm = True

        self.drop = nn.Dropout(args.dropout)
        self.emb_layer = modules.EmbeddingLayer(embs=dataloader.load_embedding(args.word_embeddings_path), fix_emb=False)
        if self.norm:
            self.norm1 = nn.LayerNorm(normalized_shape=args.emb_dim)  # 按时间片归一化
            self.norm2 = nn.LayerNorm(normalized_shape=args.hidden_size)  # 按时间片归一化
        self.word2id = self.emb_layer.word2id
        self.id2word = self.emb_layer.id2word
        self.encoder = nn.LSTM(self.emb_layer.n_d, args.hidden_size//2, args.depth, dropout=args.dropout, bidirectional=True)  # for input , seq_len first ]
        d_out = args.hidden_size
        self.out = nn.Linear(d_out, args.nclasses)

    def text_to_emb(self, *inputs):
        """
        :param inputs: tensor, id [batch_size, seq_len]
        :return: [seq_len, batch_size, emb_dim]
        """
        # if torch.cuda.device_count() > 1:
        #     x = inputs[0].cuda(self.args.rank)
        # else:
        #     x = inputs[0].cuda()
        if torch.cuda.device_count() > 1:
            x = inputs[0].cuda(self.args.rank)
        else:
            x = inputs[0].cuda()
        if x.shape[1] == self.args.max_seq_length:  # if for x, batch first, permute and let seq_len first
            x = x.permute([1, 0])
        emb = self.emb_layer(x)
        if self.norm:
            emb = self.norm1(emb)

        return emb  # []

    def emb_to_logit(self, *inputs):
        """
        :param inputs: [seq_len, batch_size, emb_dim]
        :return: [batch_size, n_class]
        """
        if inputs.shape[0] != self.args.max_seq_length:  # let seq_len first
            inputs = inputs.permute([1, 0, 2])
        emb = self.drop(inputs)
        output, hidden = self.encoder(emb)
        if self.norm:
            output = self.norm2(output)
        output = torch.max(output, dim=0)[0]
        output = self.drop(output)
        output = self.out(output)
        return output


    def forward(self, inputs):
        """Used for train
            inputs[0]: tensor, id  [batch_size, seq_len]
        """
        if torch.cuda.device_count() > 1:
            x = inputs[0].cuda(self.args.rank)
        else:
            x = inputs[0].cuda()
        if x.shape[1] == self.args.max_seq_length:  # if for x, batch first, permute and let seq_len first
            x = x.permute([1, 0])

        emb = self.emb_layer(x)  # emb[seq_len, batch_size, emb_dim]
        if self.norm:
            emb = self.norm1(emb)
        emb = self.drop(emb)
        output, hidden = self.encoder(emb)  # output[seq_len, batch_size, out_dim]
        if self.norm:
            output = self.norm2(output)
        output = torch.max(output, dim=0)[0]
        output = self.drop(output)
        output = self.out(output)
        return output


class ESIM(BaseModel):
    """ESIM: https://aclanthology.org/P17-1152.pdf"""

    def __init__(self, args, depth=1, dropout=0.):
        """no emb layer"""
        super(ESIM, self).__init__(args)
        # self.args = args
        self.dataset = Dataset_LSTM_snli(args)
        self.word_emb_dim = 300
        self.enc_lstm_dim = 2048
        self.inputdim = 4*2*self.enc_lstm_dim  # 原始的输入有长度信息
        self.inputdim = int(self.inputdim)
        self.fc_dim = 512

        self.encoder = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, depth, dropout=dropout, bidirectional=True)  # for input , seq_len first ]
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.inputdim, self.fc_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(self.fc_dim, args.nclasses),
        )


    def encode(self, *input):  # input: [ids, lens]
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: (seqlen x bsize x worddim)
        sent, sent_len = input

        # Sort by length (keep idx)
        if torch.is_tensor(sent_len):
            sent_len = sent_len.cpu().numpy()
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.encoder(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output = sent_output.index_select(1, idx_unsort)

        # Pooling
        emb = torch.max(sent_output, dim=0)[0]

        return emb

    """Used for train"""
    def forward(self, inputs):
        s1_embs, s1_lens, s2_embs, s2_lens = inputs[:4]
        s1_embs = s1_embs.cuda()
        s1_lens = s1_lens.cuda()
        s2_embs = s2_embs.cuda()
        s2_lens = s2_lens.cuda()
        if s1_embs.shape[1] == self.args.max_seq_length:  # if batch first, permute and let seq_len first
            s1_embs = s1_embs.permute([1, 0, 2])
        if s2_embs.shape[1] == self.args.max_seq_length:
            s2_embs = s2_embs.permute([1, 0, 2])

        output_s1 = self.encode(s1_embs, s1_lens)
        output_s2 = self.encode(s2_embs, s2_lens)
        features = torch.cat((output_s1, output_s2, torch.abs(output_s1-output_s2), output_s1*output_s2), 1)

        output = self.classifier(features)

        return output


    def text_pred_org(self, orig_texts, texts):
        orig_s2s, orig_s1s, s2s = [], [], []
        for orig_text, text in zip(orig_texts, texts):
            orig_s1 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[0].split(' ')
            orig_s2 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[1].split(' ')
            s2 = ' '.join(text).split(' %s ' % self.args.pad_token)[1].split(' ')
            orig_s1s.append(orig_s1)
            orig_s2s.append(orig_s2)
            s2s.append(s2)

    def text_pred_Enhance(self, orig_texts, texts):
        orig_s2s, orig_s1s, s2s = [], [], []
        for orig_text, text in zip(orig_texts, texts):
            orig_s1 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[0].split(' ')
            orig_s2 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[1].split(' ')
            s2 = ' '.join(text).split(' %s ' % self.args.pad_token)[1].split(' ')
            orig_s1s.append(orig_s1)
            orig_s2s.append(orig_s2)
            s2s.append(s2)
        perturbed_s2s = perturb_texts(self.args, orig_s2s, s2s, self.args.tf_vocabulary, change_ratio=0.2)
        Samples_s2s = gen_sample_multiTexts(self.args, orig_s2s, perturbed_s2s, self.args.sample_num, change_ratio=0.25)
        Samples_s1s = []
        for s1 in orig_s1s:
            Samples_s1s.extend([s1]*self.args.sample_num)
        Samples_x = [s1 + [self.args.pad_token] + s2 for s1, s2 in zip(Samples_s1s, Samples_s2s)]

        Sample_probs = self.text_pred_org(None, Samples_x)
        lable_mum = Sample_probs.size()[-1]
        Sample_probs = Sample_probs.view(len(texts), self.args.sample_num, lable_mum)
        probs_boost = []
        for l in range(lable_mum):
            num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)  # 获得预测值的比例作为对应标签的概率
            prob = num.float() / float(self.args.sample_num)
            probs_boost.append(prob.view(len(texts), 1))

        probs_boost_all = torch.cat(probs_boost, dim=1)
        return probs_boost_all

    def text_pred_SAFER(self, orig_texts, texts):
        orig_s2s, orig_s1s, s2s = [], [], []
        for orig_text, text in zip(orig_texts, texts):
            orig_s1 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[0].split(' ')
            orig_s2 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[1].split(' ')
            s2 = ' '.join(text).split(' %s ' % self.args.pad_token)[1].split(' ')
            orig_s1s.append(orig_s1)
            orig_s2s.append(orig_s2)
            s2s.append(s2)
        Samples_s2s = gen_sample_multiTexts(self.args, orig_s2s, s2s, self.args.sample_num, change_ratio=1)
        Samples_s1s = []
        for s1 in orig_s1s:
            Samples_s1s.extend([s1]*self.args.sample_num)
        Samples_x = [s1 + [self.args.pad_token] + s2 for s1, s2 in zip(Samples_s1s, Samples_s2s)]

        Sample_probs = self.text_pred_org(None, Samples_x)
        lable_mum = Sample_probs.size()[-1]
        Sample_probs = Sample_probs.view(len(texts), self.args.sample_num, lable_mum)
        probs_boost = []
        for l in range(lable_mum):
            num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)
            prob = num.float() / float(self.args.sample_num)
            probs_boost.append(prob.view(len(texts), 1))
        probs_boost_all = torch.cat(probs_boost, dim=1)
        return probs_boost_all

    def text_pred_FGWS(self, orig_texts, texts):
        orig_s2s, orig_s1s, s2s = [], [], []
        for orig_text, text in zip(orig_texts, texts):
            orig_s1 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[0].split(' ')
            orig_s2 = ' '.join(orig_text).split(' %s ' % self.args.pad_token)[1].split(' ')
            s2 = ' '.join(text).split(' %s ' % self.args.pad_token)[1].split(' ')
            orig_s1s.append(orig_s1)
            orig_s2s.append(orig_s2)
            s2s.append(s2)

        gamma1 = 0.5
        perturbed_s2s = perturb_FGWS(self.args, orig_s2s, s2s, self.args.tf_vocabulary)
        Samples_x = [s1 + [self.args.pad_token] + s2 for s1, s2 in zip(orig_s1s, perturbed_s2s)]
        pre_prob = self.text_pred_org(None, Samples_x)
        ori_prob = self.text_pred_org(None, texts)
        lable = torch.argmax(ori_prob, dim=1)
        index = torch.arange(len(texts)).cuda()
        D = ori_prob.gather(1, lable.view(-1, 1))-pre_prob.gather(1, lable.view(-1, 1))-gamma1

        probs_boost_all = torch.where(D > 0, pre_prob, ori_prob)
        #D=D.view(-1)
        #probs_boost_all=torch.ones_like(ori_prob)
        #probs_boost_all.index_put_((index,lable),0.5 - D)
        #probs_boost_all.index_put_((index,1-lable), 0.5+D)
        return probs_boost_all


class ROBERTA(BaseModel):
    """ROBERTA for mr/imdb"""

    def __init__(self, args):
        super(ROBERTA, self).__init__(args)
        self.args = args
        self.dataset = Dataset_ROBERTA(args)
        self.config = RobertaConfig.from_pretrained(args.model_path + 'roberta-base')
        self.encoder = RobertaModel(config=self.config)
        # self.encoder = RobertaModel.from_pretrained(args.target_model_path, config=self.config).cuda(args.rank)
        self.drop = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, args.nclasses)
        # self.model = RobertaForSequenceClassification.from_pretrained(args.target_model_path, config=self.config).cuda(args.rank) #
        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_path +'roberta-base')
        # self.tokenizer = RobertaTokenizer.from_pretrained(args.target_model_path+'/vocab.json', args.target_model_path+'/merges.json')

    def text_to_emb(self, *inputs):
        """
        :param inputs: 0 ids: [batch_size, seq_len], 1:segment_ids
        :return:
        """
        input_ids,  segment_ids = inputs

        embs = self.encoder.embeddings(input_ids)
        return embs

    def emb_to_logit(self, *inputs):
        """
        :param inputs: 0: embs, 1, attention/input_mask
        :return:
        """
        """robertamodel forward"""
        embs, input_mask = inputs

        input_mask = input_mask.cuda(self.args.rank)
        encoder_outputs = self.encoder.encoder(embs)
        sequence_output = encoder_outputs[0]
        pooled_output = self.encoder.pooler(sequence_output)

        """BertForSequenceClassification forward"""
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def forward(self, inputs):
        input_ids, input_mask, segment_ids = inputs
        # print(torch.max(input_ids))
        # print(input_ids)
        input_ids = input_ids.cuda(self.args.rank)
        input_mask = input_mask.cuda(self.args.rank)
        segment_ids = segment_ids.cuda(self.args.rank)

        hidden_state, pooled_output = self.encoder(input_ids, attention_mask=input_mask, return_dict=False)
        output = self.drop(pooled_output)
        logits = self.classifier(output)
        probs = nn.functional.softmax(logits, dim=-1)

        # logits = self.model(input_ids, input_mask, segment_ids)

        # print(logits)
        # if self.args.target_model == 'roberta':
        #     if self.args.task == 'snli':
        #         probs = nn.functional.softmax(logits.logits[:, [1, 0, 2]], dim=-1)
        #     else:
        #         probs = nn.functional.softmax(logits.logits, dim=-1)
        # else:
        #     if self.args.task == 'snli':
        #         probs = nn.functional.softmax(logits[:, [1, 0, 2]], dim=-1)  # Attention: label transpose for snli
        #     else:
        #         probs = nn.functional.softmax(logits, dim=-1)
        return probs


"""为之前集成多次对抗训练的方法写的"""
# class EnsembleBERT(BERT):
#     def __init__(self, args):
#         super(EnsembleBERT, self).__init__(args)
#
#
#         # Load multiple trained models
#         all_adv_models = []
#         num_model = 4
#         # 对抗样本训练得到的
#         # saved_files = ['/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/imdb_adv0',
#         #                '/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/imdb_adv1',
#         #                '/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/imdb_adv2']
#         # 置信度下降样本训练得到的
#         # saved_files = ['/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/imdb_adv0',
#         #                '/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/imdb_adv0_ag1_ep50',
#         #                '/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/imdb_adv0_ag1_ep50_ag2_ep10']
#         # saved_files = ['/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/imdb_ag0_ep90',
#         #                '/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/imdb_adv0_ag1_ep90',
#         #                '/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/imdb_ag0_ep90_ag1_ep90_ag2_ep90']
#         if self.args.task == 'imdb' and self.args.target_model == 'bert':
#             saved_files = ['/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/imdb_ag0',
#                            '/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/imdb_ag1',
#                            '/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/imdb_ag2']
#         elif self.args.task == 'mr' and self.args.target_model == 'bert':
#             saved_files = ['/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/mr_ag0',
#                            '/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/mr_ag1',
#                            '/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/mr_ag2']
#         for i in range(num_model - 1):
#             model = BERT(self.args).cuda(self.args.rank)
#             checkpoint = torch.load(saved_files[i] + '/pytorch_model.bin', map_location=self.args.device)
#             model.model.load_state_dict(checkpoint)
#             all_adv_models.append(model)
#
#         self.adv_models = all_adv_models
