import os
import json
import random
import shutil
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoConfig, AdamW

from common_utils import extract_from_dataframe, multi_batch_generator, multi_batch_seq_generator

from JOINT.custom_parser import my_parser
from JOINT.utils import tokenize_with_new_mask, load_model, train, evaluate

NOTE = 'Task: Multi'

def main():

    args = my_parser()

    assert args.task_type in ['entity_detection', 'relevant_entity_detection', 'entity_relevance_classification']

    print("cuda is available:", torch.cuda.is_available())

    log_directory = args.log_dir + '/' + str(args.bert_model).split('/')[-1] + '/' + args.model_type + '/' + \
                args.task_type + '/' + str(args.n_epochs) + \
                '_epoch/' + args.data.split('/')[-1] + '/' + str(args.assign_token_weight) + \
                '_token_weight/' + str(args.assign_seq_weight) + '_seq_weight/' + str(args.token_lambda) + \
                '_token_lambda/' + str(args.seed) + '_seed/'
    log_filename = 'log.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.', '-') + '.txt'

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    model_dir = 'saved-model'
    modeldir = log_directory + model_dir

    if os.path.exists(modeldir) and os.listdir(modeldir):
        print(f"modeldir {modeldir} already exists and it is not empty")
    else:
        os.makedirs(modeldir, exist_ok=True)
        print(f"Create modeldir: {modeldir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = pd.read_pickle(os.path.join(args.data, args.train_file))
    val_data = pd.read_pickle(os.path.join(args.data, args.val_file))
    test_data = pd.read_pickle(os.path.join(args.data, args.test_file))

    need_columns = ['tweet_tokens']
    if args.task_type == 'entity_detection':
        need_columns.append('entity_label')
    elif args.task_type == 'relevant_entity_detection':
        need_columns.append('relevant_entity_label')
    elif args.task_type == 'entity_relevance_classification':
        need_columns.append('relevance_entity_class_label')
    need_columns.append('sentence_class')

    X_train_raw, token_label_train_raw, Y_train = extract_from_dataframe(train_data, need_columns)
    X_dev_raw, token_label_dev_raw, Y_dev = extract_from_dataframe(val_data, need_columns)
    X_test_raw, token_label_test_raw, Y_test = extract_from_dataframe(test_data, need_columns)

    with open(os.path.join(args.data, args.label_map), 'r') as fp:
            label_map = json.load(fp)

    labels = list(label_map.keys())

    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, normalization=True)

    X_train, masks_train, token_label_train = tokenize_with_new_mask(X_train_raw, args.max_length, tokenizer, token_label_train_raw, label_map)
    X_dev, masks_dev, token_label_dev = tokenize_with_new_mask(X_dev_raw, args.max_length, tokenizer, token_label_dev_raw, label_map)
    X_test, masks_test, token_label_test = tokenize_with_new_mask(X_test_raw, args.max_length, tokenizer, token_label_test_raw, label_map)
    
    # weight of each class in loss function
    token_weight = None
    if args.assign_token_weight:
        token_weight = [token_label_train.shape[0] / (token_label_train == i).sum() for i in range(len(labels))]
        token_weight = torch.FloatTensor(token_weight)

    y_weight = None
    if args.assign_seq_weight:
        y_weight = [np.array(Y_train).shape[0] / (np.array(Y_train) == i).sum() for i in range(len(set(Y_train)))]
        y_weight = torch.FloatTensor(y_weight)

    config = AutoConfig.from_pretrained(args.bert_model)
    config.update({'num_token_labels': len(labels), 'num_labels': len(set(Y_train)), 'token_label_map': label_map, })

    model = load_model(args.model_type, args.bert_model, config)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    model = model.to(device)

    best_valid_s_auc, best_valid_s_acc = 0, 0

    best_valid_t_acc, best_valid_t_F = 0, 0
    best_valid_s_tuple, best_valid_t_tuple = None, None
    train_losses = []
    eval_losses = []
    train_s_acc_list, eval_s_acc_list = [], []
    train_t_F_list, eval_t_F_list = [], []

    early_stop_sign = 0

    for epoch in range(args.n_epochs):

        print('############# EPOCH ', epoch + 1, '##############')

        # train

        train_batch_generator = multi_batch_generator(X_train, Y_train, token_label_train,
                                                      masks_train, args.batch_size)
        num_batches = X_train.shape[0] // args.batch_size
        train_outputs = train(model, optimizer, train_batch_generator, num_batches, device,
                              args, label_map, token_weight, y_weight)
        train_s_tuple, train_t_tuple = train_outputs[:2]
        if len(train_outputs) > 2:
            train_bt_tuple = train_outputs[2]
        train_losses.append(train_s_tuple[0])
        train_s_acc = train_s_tuple[2]
        train_t_F = train_t_tuple[1]['strict']['f1']
        train_s_acc_list.append(train_s_acc)
        train_t_F_list.append(train_t_F)

        # eval
        dev_batch_generator = multi_batch_seq_generator(X_dev, Y_dev, token_label_dev, masks_dev,
                                                        min(X_dev.shape[0], args.eval_batch_size))
        num_batches = X_dev.shape[0] // min(X_dev.shape[0], args.eval_batch_size)
        valid_outputs = evaluate(model, dev_batch_generator, num_batches, device,
                                 args, label_map, token_weight, y_weight)
        valid_s_tuple, valid_t_tuple, valid_t_pred, valid_s_pred = valid_outputs[:4]

        eval_losses.append(valid_s_tuple[0])
        valid_s_acc = valid_s_tuple[2]
        valid_t_F = valid_t_tuple[1]['strict']['f1']
        eval_s_acc_list.append(valid_s_acc)
        eval_t_F_list.append(valid_t_F)

        good_cond_s = best_valid_s_acc < valid_s_acc
        normal_cond_s = (abs(best_valid_s_acc - valid_s_acc) < 0.03) or good_cond_s
        good_cond_t = best_valid_t_F < valid_t_F
        normal_cond_t = abs(best_valid_t_F - valid_t_F) < 0.03 or good_cond_t

        if (good_cond_s and normal_cond_t) or (good_cond_t and normal_cond_s) or epoch == 0:
            best_valid_s_auc, best_valid_s_acc, best_valid_s_tn, best_valid_s_fp, best_valid_s_fn, best_valid_s_tp = valid_s_tuple[
                                                                                                                     1:]
            best_valid_t_F = valid_t_F
            best_valid_s_tuple, best_valid_t_tuple = valid_s_tuple, valid_t_tuple
            best_train_t_tuple, best_train_s_tuple = train_t_tuple, train_s_tuple

            
            model.save_pretrained(modeldir)
            
            if args.early_stop:
                early_stop_sign = 0
        elif args.early_stop:
            early_stop_sign += 1

        content = f'Train Seq Acc: {train_s_acc * 100:.2f}%, Token F1: {train_t_F * 100:.2f}%. ' \
                  f'Val Seq Acc: {valid_s_acc * 100:.2f}%, Token F1: {valid_t_F * 100:.2f}%'
        print(content)
        if args.early_stop and early_stop_sign >= 5:
            break

    content = f"After {epoch + 1} epoch, Best valid token F1: {best_valid_t_F}, seq accuracy: {best_valid_s_acc}"
    print(content)

    performance_dict = vars(args)
    performance_dict['S_best_train_AUC'], performance_dict['S_best_train_ACC'], \
    performance_dict['S_best_train_TN'], performance_dict['S_best_train_FP'], \
    performance_dict['S_best_train_FN'], performance_dict['S_best_train_TP'] = best_train_s_tuple[1:]

    performance_dict['T_best_train_ACC'], performance_dict['T_best_train_results'], \
    performance_dict['T_best_train_results_by_tag'], performance_dict['T_best_train_CR'] = best_train_t_tuple

    performance_dict['S_best_valid_AUC'], performance_dict['S_best_valid_ACC'], \
    performance_dict['S_best_valid_TN'], performance_dict['S_best_valid_FP'], \
    performance_dict['S_best_valid_FN'], performance_dict['S_best_valid_TP'] = best_valid_s_tuple[1:]

    performance_dict['T_best_valid_ACC'], performance_dict['T_best_valid_results'], \
    performance_dict['T_best_valid_results_by_tag'], performance_dict['T_best_valid_CR'] = best_valid_t_tuple

    # Plot training classification loss
    epoch_count = np.arange(1, epoch + 2)
    fig, axs = plt.subplots(3, figsize=(10, 18), sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
    axs[0].plot(epoch_count, train_losses, 'b--')
    axs[0].plot(epoch_count, eval_losses, 'b-')
    axs[0].legend(['Training Loss', 'Valid Loss'], fontsize=14)
    axs[0].set_ylabel('Loss', fontsize=16)
    axs[0].tick_params(axis='y', labelsize=14, labelcolor='b')
    axs[0].tick_params(axis='x', labelsize=14)

    axs[1].plot(epoch_count, train_t_F_list, 'r--')
    axs[1].plot(epoch_count, eval_t_F_list, 'r-')
    axs[1].legend(['Training Token F1', 'Valid Token F1'], fontsize=14)
    axs[1].set_ylabel('F1', fontsize=16)
    axs[1].tick_params(axis='y', labelsize=14, labelcolor='r')
    axs[1].tick_params(axis='x', labelsize=14)

    axs[2].plot(epoch_count, train_s_acc_list, 'y--')
    axs[2].plot(epoch_count, eval_s_acc_list, 'y-')
    axs[2].legend(['Training Seq Acc', 'Valid Seq Acc'], fontsize=14)
    axs[2].set_ylabel('Acc', fontsize=16)
    axs[2].set_xlabel('Epoch', fontsize=16)
    axs[2].tick_params(axis='y', labelsize=14, labelcolor='y')
    axs[2].tick_params(axis='x', labelsize=14)

    figure_filename = 'fig.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.',
                                                                                                         '-') + '.png'
    figfullname = log_directory + figure_filename
    plt.savefig(figfullname, dpi=fig.dpi)

    num_batches = X_test.shape[0] // args.test_batch_size
    test_batch_generator = multi_batch_seq_generator(X_test, Y_test, token_label_test, masks_test,
                                                     args.test_batch_size)
    
    test_outputs = evaluate(model, test_batch_generator, num_batches, device, args, label_map, token_weight, y_weight) 

    test_s_tuple, test_t_tuple, test_t_pred, test_s_pred = test_outputs[0:4]
    test_t_F = test_t_tuple[1]['strict']['f1']
    content = f'Test Seq Acc: {test_s_tuple[2] * 100:.2f}%, Token F1: {test_t_F * 100:.2f}%'
    print(content)
    
    token_pred_dir = log_directory + 'token_prediction.npy'
    np.save(token_pred_dir, test_t_pred)
    seq_pred_dir = log_directory + 'seq_prediction.npy'
    np.save(seq_pred_dir, test_s_pred)

    performance_dict['S_best_test_AUC'], performance_dict['S_best_test_ACC'], \
    performance_dict['S_best_test_TN'], performance_dict['S_best_test_FP'], \
    performance_dict['S_best_test_FN'], performance_dict['S_best_test_TP'] = test_s_tuple[1:]

    performance_dict['T_best_test_ACC'], performance_dict['T_best_test_results'], \
    performance_dict['T_best_test_results_by_tag'], performance_dict['T_best_test_CR'] = test_t_tuple

    performance_dict['script_file'] = os.path.basename(__file__)
    performance_dict['log_directory'] = log_directory
    performance_dict['log_filename'] = log_filename
    performance_dict['note'] = NOTE
    performance_dict['Time'] = str(datetime.datetime.now())
    performance_dict['device'] = torch.cuda.get_device_name(device)

    for key, value in performance_dict.items():
        if type(value) is np.int64:
            performance_dict[key] = int(value)
    with open(args.performance_file, 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')
    if not args.save_model:
        shutil.rmtree(modeldir)

  
if __name__ == '__main__':
    main()