import torch
from transformers import AutoConfig, AutoTokenizer, AdamW

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime

import TRC.utils
import EMD.utils

from TRC.models import *
from EMD.models import *

import common_utils


saved_model_path_TRC = '/home/cc/rora_tesi/log_rora_tesi/log-tweet-classification/xlm-roberta-large/bertweet-seq/14_epoch/data/True_weight/42_seed/saved-model'
config_path_TRC = '/home/cc/rora_tesi/log_rora_tesi/log-tweet-classification/xlm-roberta-large/bertweet-seq/14_epoch/data/True_weight/42_seed/saved-model/config.json'

saved_model_path_EMD = '/home/cc/rora_tesi/log_rora_tesi/log-token-classification/xlm-roberta-large/bertweet-token-crf/entity_detection/14_epoch/data/True_weight/42_seed/saved-model'
config_path_EMD = '/home/cc/rora_tesi/log_rora_tesi/log-token-classification/xlm-roberta-large/bertweet-token-crf/entity_detection/14_epoch/data/True_weight/42_seed/saved-model/config.json'


config_TRC = AutoConfig.from_pretrained(config_path_TRC)

config_EMD = AutoConfig.from_pretrained(config_path_EMD)

# model_TRC = TRC.utils.load_model('bertweet-seq', saved_model_path_TRC, config_TRC)
model_TRC = TRC.utils.load_model(saved_model_path_TRC, config_TRC)


model_EMD = EMD.utils.load_model('bertweet-token-crf', saved_model_path_EMD, config_EMD)

sd_TRC = model_TRC.state_dict()

classifier_weight_TRC = sd_TRC['classifier.weight']
classifier_bias_TRC = sd_TRC['classifier.bias']


# for k in sd_TRC.keys():
#     print(k, sd_TRC[k].shape)

sd_EMD = model_EMD.state_dict()

classifier_weight_EMD = sd_EMD['classifier.weight']
classifier_bias_EMD = sd_EMD['classifier.bias']
crf_start_EMD = sd_EMD['crf.start_transitions']
crf_end_EMD = sd_EMD['crf.end_transitions']
crf_transitions_EMD = sd_EMD['crf.transitions']


# for k in sd_EMD.keys():
#     print(k,sd_EMD[k].shape)

sd_EMD_new = sd_EMD

del sd_EMD_new['classifier.weight']
del sd_EMD_new['classifier.bias']
del sd_EMD_new['crf.start_transitions']
del sd_EMD_new['crf.end_transitions']
del sd_EMD_new['crf.transitions']

sd_EMD_new['classifier.weight'] = classifier_weight_TRC
sd_EMD_new['classifier.bias'] = classifier_bias_TRC

print(model_EMD.config.num_labels)
model_EMD.config.num_labels = 2
print(model_EMD.config.num_labels)

# PRENDO I PESI DI EMD, LEVO GLI ULTIMO DEL CLASSIFICATORE E DI CRF, CI FICCO QUELLI DI TRC 
# E POI CON IL MODELLO NUOVO TESTO SU TRC
# MODEL_NEW = FROM EMD TO TRC

config_EMD.update({'num_labels': 2})
#print(config_TRC.num_labels)
# model_new = TRC.utils.load_model('bertweet-seq', saved_model_path_EMD, config_EMD)
model_new = TRC.utils.load_model(saved_model_path_EMD, config_EMD)
print(model_new.config.num_labels)

model_new.load_state_dict(sd_EMD_new)
print('NEW',model_new.config.num_labels)


test_data = pd.read_pickle('/home/cc/rora_tesi/data/test.p')
val_data = pd.read_pickle('/home/cc/rora_tesi/data/dev.p')
train_data = pd.read_pickle('/home/cc/rora_tesi/data/train.p')

need_columns = ['tweet_tokens', 'sentence_class']

X_train_raw, Y_train = common_utils.extract_from_dataframe(train_data, need_columns)
X_dev_raw, Y_dev = common_utils.extract_from_dataframe(val_data, need_columns)
X_test_raw, Y_test = common_utils.extract_from_dataframe(test_data, need_columns)

# with open('/home/cc/rora_tesi/data/label_map.json', 'r') as fp:
#     label_map = json.load(fp)

# labels = list(label_map.keys())

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large', normalization=True)

X_train, masks_train = TRC.utils.tokenize_with_new_mask(
            X_train_raw, 128, tokenizer)
X_dev, masks_dev = TRC.utils.tokenize_with_new_mask(
    X_dev_raw, 128, tokenizer)
X_test, masks_test = TRC.utils.tokenize_with_new_mask(
    X_test_raw, 128, tokenizer)

class_weight = None
# default True
class_weight = [np.array(Y_train).shape[0] / (np.array(Y_train) == i).sum() for i in range(len(set(Y_train)))]
class_weight = torch.FloatTensor(class_weight)

param_optimizer = list(model_new.named_parameters())

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_new = model_new.to(device)

best_valid_acc = 0
best_valid_auc = 0
best_valid_tn, best_valid_fp, best_valid_fn, best_valid_tp, best_valid_precision, best_valid_recall = 0, 0, 0, 0, 0, 0
train_losses = []
eval_losses = []
train_acc_list, eval_acc_list = [], []

early_stop_sign = 0

for epoch in range(6):

    print(f'########## EPOCH {epoch+1} ##########')

    train_batch_generator = common_utils.mask_batch_generator(X_train, Y_train, masks_train, 32)
    num_batches = X_train.shape[0] // 32
    train_loss, train_auc, train_acc, train_tn, train_fp, train_fn, train_tp, train_precision, train_recall = TRC.utils.train(model_new, optimizer,
                                                                                        train_batch_generator,
                                                                                        num_batches, device,
                                                                                        class_weight)
    train_losses.append(train_loss)
    train_acc_list.append(train_acc)

    # eval
    dev_batch_generator = common_utils.mask_batch_seq_generator(X_dev, Y_dev, masks_dev,
                                                    min(X_dev.shape[0], 412))
    num_batches = X_dev.shape[0] // min(X_dev.shape[0], 412)
    valid_loss, valid_auc, valid_acc, valid_tn, valid_fp, valid_fn, valid_tp, valid_precision, valid_recall, valid_s_pred = TRC.utils.evaluate(model_new,
                                                                                                                    dev_batch_generator,
                                                                                                                    num_batches,
                                                                                                                    device,
                                                                                                                    class_weight)
    eval_losses.append(valid_loss)
    eval_acc_list.append(valid_acc)

    if best_valid_acc < valid_acc or epoch == 0:
        best_valid_acc = valid_acc
        best_valid_auc = valid_auc
        best_valid_tn = valid_tn
        best_valid_fp = valid_fp
        best_valid_fn = valid_fn
        best_valid_tp = valid_tp

        best_valid_precision = valid_precision
        best_valid_recall = valid_recall

        best_train_auc = train_auc
        best_train_acc = train_acc
        best_train_tn = train_tn
        best_train_fp = train_fp
        best_train_fn = train_fn
        best_train_tp = train_tp

        best_train_precision = train_precision
        best_train_recall = train_recall


    print(f'Train Acc: {train_acc * 100:.2f}%')
    print(f'Train Precision: {train_precision * 100:.2f}%')
    print(f'Train Recall: {train_recall * 100:.2f}%')

    print(f'Val. Acc: {valid_acc * 100:.2f}%')
    print(f'Val. Precision: {valid_precision * 100:.2f}%')
    print(f'Val. Recall: {valid_recall * 100:.2f}%')

print(f"After {epoch + 1} epoch, Best valid accuracy: {best_valid_acc}, Best valid precision: {best_valid_precision* 100:.2f}%, Best valid recall: {best_valid_recall* 100:.2f}%")
    
# Plot training classification loss
epoch_count = np.arange(1, epoch + 2)
fig, axs = plt.subplots(2, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
axs[0].plot(epoch_count, train_losses, 'b--')
axs[0].plot(epoch_count, eval_losses, 'b-')
axs[0].legend(['Training Loss', 'Valid Loss'], fontsize=14)
axs[0].set_ylabel('Loss', fontsize=16)
axs[0].tick_params(axis='y', labelsize=14, labelcolor='b')
axs[0].tick_params(axis='x', labelsize=14)

axs[1].plot(epoch_count, train_acc_list, 'y--')
axs[1].plot(epoch_count, eval_acc_list, 'y-')
axs[1].legend(['Training Acc', 'Valid Acc'], fontsize=14)
axs[1].set_ylabel('Acc', fontsize=16)
axs[1].set_xlabel('Epoch', fontsize=16)
axs[1].tick_params(axis='y', labelsize=14, labelcolor='y')
axs[1].tick_params(axis='x', labelsize=14)

figure_filename = 'fig.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.','-') + '.png'

log_directory = '/home/cc/rora_tesi/log_rora_tesi/from_EMD_to_TRC/'                                                          
figfullname = log_directory + figure_filename
plt.savefig(figfullname, dpi=fig.dpi)

num_batches = X_test.shape[0] // 412
test_batch_generator = common_utils.mask_batch_seq_generator(X_test, Y_test, masks_test, 412)
        


test_loss, test_auc, test_acc, test_tn, test_fp, test_fn, test_tp, test_precision, test_recall, test_s_pred = TRC.utils.evaluate(model_new,
                                                                                              test_batch_generator,
                                                                                              num_batches, device,
                                                                                              class_weight)

content = f'Test Acc: {test_acc * 100:.2f}%, AUC: {test_auc * 100:.2f}%, TN: {test_tn}, FP: {test_fp}, FN: {test_fn}, TP: {test_tp}, Precision: {test_precision* 100:.2f}%, Recall: {test_recall* 100:.2f}%'
print(content)