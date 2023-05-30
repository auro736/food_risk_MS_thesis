import os
import json
import shutil
import logging
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from transformers import AutoTokenizer, AutoConfig, AdamW

from TRC.custom_parser import my_parser
from TRC.utils import tokenize_with_new_mask, load_model, train, evaluate, calibration_plot

from common_utils import extract_from_dataframe, mask_batch_generator, mask_batch_seq_generator

NOTE = 'V1.0.0: Initial Public Version'

def main():

    args = my_parser()

    print("cuda is available:", torch.cuda.is_available())

    log_directory = args.log_dir + '/' + str(args.bert_model).split('/')[-1] + '/' + args.model_type + '/' \
                    + str(args.n_epochs) + '_epoch/' + args.data.split('/')[-1] + '/' + \
                    str(args.assign_weight) + '_weight/' + str(args.seed) + '_seed/'
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
    need_columns = ['tweet_tokens', 'sentence_class']

    X_train_raw, Y_train = extract_from_dataframe(train_data, need_columns)
    X_dev_raw, Y_dev = extract_from_dataframe(val_data, need_columns)
    X_test_raw, Y_test = extract_from_dataframe(test_data, need_columns)
    args.eval_batch_size = Y_dev.shape[0]
    args.test_batch_size = Y_test.shape[0]

    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, normalization=True)
    X_train, masks_train = tokenize_with_new_mask(X_train_raw, args.max_length, tokenizer)
    X_dev, masks_dev = tokenize_with_new_mask(X_dev_raw, args.max_length, tokenizer)
    X_test, masks_test = tokenize_with_new_mask(X_test_raw, args.max_length, tokenizer)

    # weight of each class in loss function
    class_weight = None
    if args.assign_weight:
        class_weight = [np.array(Y_train).shape[0] / (np.array(Y_train) == i).sum() for i in range(len(set(Y_train)))]
        class_weight = torch.FloatTensor(class_weight)

    config = AutoConfig.from_pretrained(args.bert_model)
    config.update({'num_labels': len(set(Y_train))})
    
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

    best_valid_acc = 0
    best_valid_auc = 0
    best_valid_tn, best_valid_fp, best_valid_fn, best_valid_tp, best_valid_precision, best_valid_recall = 0, 0, 0, 0, 0, 0
    train_losses = []
    eval_losses = []
    train_acc_list, eval_acc_list = [], []

    early_stop_sign = 0

    for epoch in range(args.n_epochs):

        print(f'########## EPOCH {epoch+1} ##########')

        # train
        train_batch_generator = mask_batch_generator(X_train, Y_train, masks_train, args.batch_size)
        num_batches = X_train.shape[0] // args.batch_size
        train_loss, train_auc, train_acc, train_tn, train_fp, train_fn, train_tp, train_precision, train_recall = train(model, optimizer,
                                                                                         train_batch_generator,
                                                                                         num_batches, device,
                                                                                         class_weight)
        train_losses.append(train_loss)
        train_acc_list.append(train_acc)

        # eval
        dev_batch_generator = mask_batch_seq_generator(X_dev, Y_dev, masks_dev,
                                                       min(X_dev.shape[0], args.eval_batch_size))
        num_batches = X_dev.shape[0] // min(X_dev.shape[0], args.eval_batch_size)
        _, _, valid_loss, valid_auc, valid_acc, valid_tn, valid_fp, valid_fn, valid_tp, valid_precision, valid_recall, valid_s_pred = evaluate(model,
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

            model.save_pretrained(modeldir)
            
            if args.early_stop:
                early_stop_sign = 0
        elif args.early_stop:
            early_stop_sign += 1

        print(f'Train Acc: {train_acc * 100:.2f}%')
        print(f'Train Precision: {train_precision * 100:.2f}%')
        print(f'Train Recall: {train_recall * 100:.2f}%')

        print(f'Val. Acc: {valid_acc * 100:.2f}%')
        print(f'Val. Precision: {valid_precision * 100:.2f}%')
        print(f'Val. Recall: {valid_recall * 100:.2f}%')

        if args.early_stop and early_stop_sign >= 5:
            break

    print(f"After {epoch + 1} epoch, Best valid accuracy: {best_valid_acc}, Best valid precision: {best_valid_precision* 100:.2f}%, Best valid recall: {best_valid_recall* 100:.2f}%")

    performance_dict = vars(args)
    performance_dict['S_best_train_AUC'] = best_train_auc
    performance_dict['S_best_train_ACC'] = best_train_acc
    performance_dict['S_best_train_TN'] = best_train_tn
    performance_dict['S_best_train_FP'] = best_train_fp
    performance_dict['S_best_train_FN'] = best_train_fn
    performance_dict['S_best_train_TP'] = best_train_tp

    performance_dict['S_best_train_precision'] = best_train_precision
    performance_dict['S_best_train_recall'] = best_train_recall

    performance_dict['S_best_valid_AUC'] = best_valid_auc
    performance_dict['S_best_valid_ACC'] = best_valid_acc
    performance_dict['S_best_valid_TN'] = best_valid_tn
    performance_dict['S_best_valid_FP'] = best_valid_fp
    performance_dict['S_best_valid_FN'] = best_valid_fn
    performance_dict['S_best_valid_TP'] = best_valid_tp

    performance_dict['S_best_valid_precision'] = best_valid_precision
    performance_dict['S_best_valid_recall'] = best_valid_recall

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

    figure_filename = 'fig.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.',
                                                                                                         '-') + '.png'
    figfullname = log_directory + figure_filename
    plt.savefig(figfullname, dpi=fig.dpi)

    num_batches = X_test.shape[0] // args.test_batch_size
    test_batch_generator = mask_batch_seq_generator(X_test, Y_test, masks_test, args.test_batch_size)

    logits, y_batch, test_loss, test_auc, test_acc, test_tn, test_fp, test_fn, test_tp, test_precision, test_recall, test_s_pred = evaluate(model,
                                                                                              test_batch_generator,
                                                                                              num_batches, device,
                                                                                              class_weight)
    
    
    figure_name = 'calibration_curve.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.',
                                                                                                         '-')+ '.png'
    path = log_directory+figure_name

    calibration_plot(logits=logits, y = y_batch, img_path=path)

    content = f'Test Acc: {test_acc * 100:.2f}%, AUC: {test_auc * 100:.2f}%, TN: {test_tn}, FP: {test_fp}, FN: {test_fn}, TP: {test_tp}, Precision: {test_precision* 100:.2f}%, Recall: {test_recall* 100:.2f}%'
    print(content)

    seq_pred_dir = log_directory + 'seq_prediction.npy'
    np.save(seq_pred_dir, test_s_pred)

    performance_dict['S_best_test_AUC'] = test_auc
    performance_dict['S_best_test_ACC'] = test_acc
    performance_dict['S_best_test_TN'] = test_tn
    performance_dict['S_best_test_FP'] = test_fp
    performance_dict['S_best_test_FN'] = test_fn
    performance_dict['S_best_test_TP'] = test_tp

    performance_dict['S_best_test_precision'] = test_precision
    performance_dict['S_best_test_recall'] = test_recall

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