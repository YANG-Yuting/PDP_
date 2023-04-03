from __future__ import print_function
import json

# from models import LSTM, ESIM, BERT, BERT_snli, ROBERTA, EnsembleBERT
from dataset import *
from config import args

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, BertTokenizer
from datasets import Dataset, load_dataset, load_metric
import tensorflow as tf
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models import EnsembleBERT_new

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

hp_lamda = 0
hp_log_det_lamda = 0
log_offset = 1e-20
det_offset = 1e-6
CEloss = nn.CrossEntropyLoss()

## Function ##
def Entropy(input):
    #input shape is batch_size X num_class
    return tf.reduce_sum(-tf.multiply(input, tf.log(input + log_offset)), axis=-1)


def Ensemble_Entropy(y_true, y_pred, num_model=args.num_models):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_p_all = 0
    for i in range(num_model):
        y_p_all += y_p[i]
    Ensemble = Entropy(y_p_all / num_model)
    return Ensemble

def log_det(y_true, y_pred, num_model=args.num_models):
    bool_R_y_true = tf.not_equal(tf.ones_like(y_true) - y_true, 0) # batch_size X (num_class X num_models), 2-D
    mask_non_y_pred = tf.boolean_mask(y_pred, bool_R_y_true) # batch_size X (num_class-1) X num_models, 1-D
    mask_non_y_pred = tf.reshape(mask_non_y_pred, [-1, num_model, args.num_classes-1]) # batch_size X num_model X (num_class-1), 3-D
    mask_non_y_pred = mask_non_y_pred / tf.norm(mask_non_y_pred, axis=2, keepdims=True) # batch_size X num_model X (num_class-1), 3-D
    matrix = tf.matmul(mask_non_y_pred, tf.transpose(mask_non_y_pred, perm=[0, 2, 1])) # batch_size X num_model X num_model, 3-D
    all_log_det = tf.linalg.logdet(matrix+det_offset*tf.expand_dims(tf.eye(num_model),0)) # batch_size X 1, 1-D
    return all_log_det

def Loss_withEE_DPP(y_true, y_pred, num_model=args.num_models):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_t = tf.split(y_true, num_model, axis=-1)
    CE_all = 0
    for i in range(num_model):
        CE_all += CEloss(y_t[i], y_p[i])
    if hp_lamda == 0 and hp_log_det_lamda == 0:
        print('This is original ECE!')
        return CE_all
    else:
        EE = Ensemble_Entropy(y_true, y_pred, num_model)
        log_dets = log_det(y_true, y_pred, num_model)
        return CE_all - hp_lamda * EE - hp_log_det_lamda * log_dets


# class MyTrainer(Trainer):
#     """Customized trainer"""
#
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#
#         # my loss
#         pred_labels = 0
#         loss = Loss_withEE_DPP(labels, pred_labels)
#
#         return (loss, outputs) if return_outputs else loss

class MyTrainer(Trainer):
    """Customized trainer"""

    def compute_loss(self, model, inputs, return_outputs=False):
        all_preds, all_labels = model(**inputs)
        # labels = inputs.get("labels")
        # all_preds = []
        # all_labels = []
        # for m in model.values():
        #     print(inputs)
        #     outputs = m(**inputs)
        #     print(outputs)
        #     print(labels)
        #     exit(0)
        #     logits = outputs.get("logits")
        #     all_preds.append(logits)
        #     all_labels.append(labels)
        # all_preds = torch.cat(all_preds, dim=0)
        # all_labels = torch.cat(all_labels, dim=0)

        loss = Loss_withEE_DPP(all_labels, all_preds)

        return (loss, all_preds) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main(args):
    metric = load_metric("accuracy")

    """Build test set"""
    # tokenizer = AutoTokenizer.from_pretrained('/data/yangyuting/TextFooler/models/bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    test_x = args.datasets.test_seqs2
    test_x = [' '.join([args.inv_full_dict[w] for w in x]) for x in test_x]
    test_y = args.datasets.test_y
    dataset_test = Dataset.from_dict({'text': test_x[:100], 'label': test_y[:100]})
    dataset_test = dataset_test.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length',
                                                          max_length=args.max_seq_length), batched=True)
    dataset_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    print(dataset_test)

    """Build model"""
    # model = BertForSequenceClassification.from_pretrained(args.target_model_path, num_labels=args.num_labels)
    model = EnsembleBERT_new(args)

    if args.mode == 'train':
        """Build train set"""
        train_x = args.datasets.train_seqs2
        train_x = [' '.join([args.inv_full_dict[w] for w in x]) for x in train_x]
        train_y = list(args.datasets.train_y)
        dataset_train = Dataset.from_dict({'text': train_x[:100], 'label': train_y[:100]})
        dataset_train = dataset_train.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length',
                                                              max_length=args.max_seq_length), batched=True)
        dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        print(dataset_train)
        """Build trainer"""
        train_args = TrainingArguments(
            output_dir=args.save_path,
            overwrite_output_dir=True,
            evaluation_strategy="steps",
            eval_steps=10,
            logging_strategy="steps",
            logging_steps=10,
            logging_dir=args.save_path+'/log',
            save_strategy="steps",
            save_steps=10,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            do_train=True,
            do_eval=True,
            num_train_epochs=args.max_epoch,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="tensorboard",
            save_total_limit=3,
            local_rank=int(os.environ.get('LOCAL_RANK', -1)),
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,

        )
        print(trainer.is_model_parallel)
        """Training..."""
        trainer.train()

        print('-----test-----')
        predictions = trainer.predict(dataset_test)
        test_predictions_argmax = np.argmax(predictions[0], axis=1)
        test_references = np.array(dataset_test["label"])
        result = metric.compute(predictions=test_predictions_argmax, references=test_references)
        print(result)

    elif args.mode == 'test':
        trainer = Trainer(model=model)
        predictions = trainer.predict(dataset_test)
        test_predictions_argmax = np.argmax(predictions[0], axis=1)
        test_references = np.array(dataset_test["label"])
        result = metric.compute(predictions=test_predictions_argmax, references=test_references)
        print(result)


if __name__ == '__main__':
    main(args)

