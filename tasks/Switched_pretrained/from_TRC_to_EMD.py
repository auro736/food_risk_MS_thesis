import torch
from transformers import AutoConfig, AutoTokenizer, AdamW

import pandas as pd
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt

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

model_TRC = TRC.utils.load_model('bertweet-seq', saved_model_path_TRC, config_TRC)

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

sd_TRC_new = sd_TRC

del sd_TRC_new['classifier.weight']
del sd_TRC_new['classifier.bias']

sd_TRC_new['classifier.weight'] = classifier_weight_EMD
sd_TRC_new['classifier.bias'] = classifier_bias_EMD
sd_TRC_new['crf.start_transitions'] = crf_start_EMD
sd_TRC_new['crf.end_transitions'] = crf_end_EMD
sd_TRC_new['crf.transitions'] = crf_transitions_EMD

print(model_TRC.config.num_labels)
model_TRC.config.num_labels = 9
print(model_TRC.config.num_labels)

# PRENDO I PESI DI TRC, LEVO GLI ULTIMO DEL CLASSIFICATORE, CI FICCO QUELLI DI EMD E IL CRF DI EMD
# E POI CON IL MODELLO NUOVO TESTO SU EMD
# MODEL_NEW = FROM TRC TO EMD

config_TRC.update({'num_labels': 9})
#print(config_TRC.num_labels)
model_new = EMD.utils.load_model('bertweet-token-crf', saved_model_path_TRC, config_TRC)
print(model_new.config.num_labels)

model_new.load_state_dict(sd_TRC_new)
print('NEW',model_new.config.num_labels)


test_data = pd.read_pickle('/home/cc/rora_tesi/data/test.p')
val_data = pd.read_pickle('/home/cc/rora_tesi/data/dev.p')
train_data = pd.read_pickle('/home/cc/rora_tesi/data/train.p')

need_columns = ['tweet_tokens', 'entity_label','sentence_class']

X_train_raw, Y_train_raw, seq_train = common_utils.extract_from_dataframe(train_data, need_columns)
X_dev_raw, Y_dev_raw, seq_dev = common_utils.extract_from_dataframe(val_data, need_columns)
X_test_raw, Y_test_raw, seq_test = common_utils.extract_from_dataframe(test_data, need_columns)

with open('/home/cc/rora_tesi_new/data/label_map.json', 'r') as fp:
    label_map = json.load(fp)

labels = list(label_map.keys())

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large', normalization=True)

X_train, masks_train, Y_train = EMD.utils.tokenize_with_new_mask(
            X_train_raw, 128, tokenizer, Y_train_raw, label_map)
X_dev, masks_dev, Y_dev = EMD.utils.tokenize_with_new_mask(
            X_dev_raw, 128, tokenizer, Y_dev_raw, label_map)
X_test, masks_test, Y_test = EMD.utils.tokenize_with_new_mask(
            X_test_raw, 128, tokenizer, Y_test_raw, label_map)

class_weight = None
# default True
class_weight = [Y_train.shape[0] / (Y_train == i).sum() for i in range(len(labels))]
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
best_valid_P, best_valid_R, best_valid_F = 0, 0, 0
train_losses = []
eval_losses = []
train_F_list, eval_F_list = [], []

for epoch in range(10):

    print('########## EPOCH: ', epoch+1, '##########')

    # train

    train_batch_generator = common_utils.mask_batch_generator(X_train, Y_train, masks_train, 32)
    print(train_batch_generator)
    num_batches = X_train.shape[0] // 32
    args=None
    train_loss, train_acc, train_results, train_results_by_tag, train_CR = EMD.utils.train(model_new, optimizer,
                                                                                    train_batch_generator,
                                                                                    num_batches,
                                                                                    device, args, label_map,
                                                                                    class_weight)
    train_losses.append(train_loss)
    train_F = train_results['strict']['f1']
    train_P = train_results['strict']['precision']
    train_R = train_results['strict']['recall']
    train_F_list.append(train_F)

    # eval
    dev_batch_generator = common_utils.mask_batch_seq_generator(X_dev, Y_dev, masks_dev,
                                                    min(X_dev.shape[0], 412))
    num_batches = X_dev.shape[0] // min(X_dev.shape[0], 412)
    valid_loss, valid_acc, valid_results, valid_results_by_tag, valid_t_pred, valid_CR = EMD.utils.evaluate(model_new,
                                                                                                    dev_batch_generator,
                                                                                                    num_batches,
                                                                                                    device,
                                                                                                    label_map,
                                                                                                    class_weight)
    eval_losses.append(valid_loss)
    valid_F = valid_results['strict']['f1']
    valid_P = valid_results['strict']['precision']
    valid_R = valid_results['strict']['recall']
    eval_F_list.append(valid_F)

    if best_valid_F < valid_F or epoch == 0:
        best_valid_acc = valid_acc
        best_valid_P = valid_P
        best_valid_R = valid_R
        best_valid_F = valid_F
        best_valid_results = valid_results
        best_valid_results_by_tag = valid_results_by_tag
        best_valid_CR = valid_CR

        best_train_acc = train_acc
        best_train_P = train_P
        best_train_R = train_R
        best_train_F = train_F
        best_train_results = train_results
        best_train_results_by_tag = train_results_by_tag
        best_train_CR = train_CR

        

    print(f'Train Acc: {train_acc * 100:.2f}%')
    print(f'Train P: {train_P * 100:.2f}%')
    print(f'Train R: {train_R * 100:.2f}%')
    print(f'Train F1: {train_F * 100:.2f}%')
    print(f'Val. Acc: {valid_acc * 100:.2f}%')
    print(f'Val. P: {valid_P * 100:.2f}%')
    print(f'Val. R: {valid_R * 100:.2f}%')
    print(f'Val. F1: {valid_F * 100:.2f}%')

content = f"After {epoch + 1} epoch, Best valid F1: {best_valid_F}, accuracy: {best_valid_acc}, Recall: {best_valid_R}, Precision: {best_valid_P}"
print(content)

 # Plot training classification loss
epoch_count = np.arange(1, epoch + 2)
fig, axs = plt.subplots(2, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
axs[0].plot(epoch_count, train_losses, 'b--')
axs[0].plot(epoch_count, eval_losses, 'b-')
axs[0].legend(['Training Loss', 'Valid Loss'], fontsize=14)
axs[0].set_ylabel('Loss', fontsize=16)
axs[0].tick_params(axis='y', labelsize=14, labelcolor='b')
axs[0].tick_params(axis='x', labelsize=14)
axs[1].plot(epoch_count, train_F_list, 'r--')
axs[1].plot(epoch_count, eval_F_list, 'r-')
axs[1].legend(['Training F1', 'Valid F1'], fontsize=14)
axs[1].set_ylabel('F1', fontsize=16)
axs[1].set_xlabel('Epoch', fontsize=16)
axs[1].tick_params(axis='y', labelsize=14, labelcolor='r')
axs[1].tick_params(axis='x', labelsize=14)

figure_filename = 'fig.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.',
                                                                                                        '-') + '.png'
log_directory = '/home/cc/rora_tesi/log_rora_tesi/from_TRC_to_EMD/'
figfullname = log_directory + figure_filename
plt.savefig(figfullname, dpi=fig.dpi)

num_batches = X_test.shape[0] // 412
test_batch_generator = common_utils.mask_batch_seq_generator(X_test, Y_test, masks_test, 412)


test_loss, test_acc, test_results, test_results_by_tag, test_t_pred, test_CR = EMD.utils.predict(model_new,
                                                                                           test_batch_generator,
                                                                                           num_batches, device,
                                                                                           label_map, class_weight)

test_F = test_results['strict']['f1']
test_P = test_results['strict']['precision']
test_R = test_results['strict']['recall']
print(f'Test Acc: {test_acc * 100:.2f}%')
print(f'Test P Strict: {test_P * 100:.2f}%')
print(f'Test R Strict: {test_R * 100:.2f}%')
print(f'Test F1 Strict: {test_F * 100:.2f}%')

test_F_2 = test_results['ent_type']['f1']
test_P_2 = test_results['ent_type']['precision']
test_R_2 = test_results['ent_type']['recall']
print(f'Test P Entity_type: {test_P_2 * 100:.2f}%')
print(f'Test R Entity_type: {test_R_2 * 100:.2f}%')
print(f'Test F1 Entity_type: {test_F_2 * 100:.2f}%')