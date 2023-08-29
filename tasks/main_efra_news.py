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

from TRC.utils import train, evaluate, load_local_TRC_model

from common_utils import extract_from_dataframe, mask_batch_generator, mask_batch_seq_generator

from TRC.utils import load_local_TRC_model, load_model
from EFRA.utils import tokenize_with_new_mask_news
from EFRA.custom_parser import my_parser

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# def split_df(data):
#     random_indices = np.random.permutation(data.index)

#     train_ratio = 0.6
#     val_ratio = 0.2
#     test_ratio = 0.2

#     num_samples = len(data)
#     num_train = int(num_samples * train_ratio)
#     num_val = int(num_samples * val_ratio)

#     train_indices = random_indices[:num_train]
#     val_indices = random_indices[num_train:num_train + num_val]
#     test_indices = random_indices[num_train + num_val:]

#     train_set = data.loc[train_indices]
#     val_set = data.loc[val_indices]
#     test_set = data.loc[test_indices]
    
#     return train_set, val_set, test_set

def main():

    args = my_parser()

    if args.from_finetuned:
        log_directory = args.log_dir +'/news/' + str(args.bert_model).split('/')[-1] + '/from_finetuned' + '/' + args.model_type + '/' \
                    + str(args.n_epochs) + '_epoch/' + args.data.split('/')[-1] + '/' + \
                    str(args.assign_weight) + '_weight/' + str(args.seed) + '_seed/'
    else:
        log_directory = args.log_dir + '/news/' + str(args.bert_model).split('/')[-1] + '/no_finetuned' + '/' + args.model_type + '/' \
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


    # data_path = '/home/cc/rora_tesi_new/data/SampleAgroknow/news_updated.p'
    
    # news = pd.read_pickle(data_path)
    # train_news, val_news, test_news = split_df(news)

    data_path = '/home/agensale/rora_tesi_new/data/SampleAgroknow/News/'

    train_news = pd.read_pickle(data_path + 'train_news.p')
    val_news = pd.read_pickle(data_path + 'val_news.p')
    test_news = pd.read_pickle(data_path + 'test_news.p')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.bert_model

    print(args.max_length)

    #device = "cpu"
  
    need_columns = ['words', 'sentence_class']

    X_train_raw, Y_train = extract_from_dataframe(train_news, need_columns)
    X_dev_raw, Y_dev = extract_from_dataframe(val_news, need_columns)
    X_test_raw, Y_test = extract_from_dataframe(test_news, need_columns)
    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization = True)

    if args.from_finetuned:
        print('USING FINETUNED MODEL')

        # model_path = '/home/cc/rora_tesi_new/log/log_TRC/roberta-large/bertweet-seq/24_epoch/Tweet-Fid/True_weight/42_seed/saved-model/pytorch_model.bin'
        # config_path = '/home/cc/rora_tesi_new/log/log_TRC/roberta-large/bertweet-seq/24_epoch/Tweet-Fid/True_weight/42_seed/saved-model/config.json'
        
        # SE USI HPC
        model_path = '/home/agensale/rora_tesi/log_rora_tesi/log-tweet-classification/roberta-large/bertweet-seq/24_epoch/data/True_weight/42_seed/saved-model/pytorch_model.bin'
        config_path = '/home/agensale/rora_tesi/log_rora_tesi/log-tweet-classification/roberta-large/bertweet-seq/24_epoch/data/True_weight/42_seed/saved-model/config.json'

        model = load_local_TRC_model(model_path, config_path, device, model_name)
        
    else: 
        print('NO FINETUNED')
        config = AutoConfig.from_pretrained(args.bert_model)
        model = load_model(args.model_type, args.bert_model, config)


    model = model.to(device)
    # X_train_raw, Y_train = X_train_raw[:5], Y_train[:5]
    # X_dev_raw, Y_dev = X_dev_raw[:5], Y_dev[:5]
    # X_test_raw, Y_test = X_test_raw[:5], Y_test[:5]
    

    X_train, masks_train = tokenize_with_new_mask_news(X_train_raw, args.max_length, tokenizer)
    X_dev, masks_dev = tokenize_with_new_mask_news(X_dev_raw, args.max_length, tokenizer)
    X_test, masks_test = tokenize_with_new_mask_news(X_test_raw, args.max_length, tokenizer)

    # weight of each class in loss function
    assign_weight = True
    class_weight = None
    if assign_weight:
        class_weight = [np.array(Y_train).shape[0] / (np.array(Y_train) == i).sum() for i in range(len(set(Y_train)))]
        class_weight = torch.FloatTensor(class_weight)
        
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6)

    best_valid_acc = 0
    best_valid_auc = 0
    best_valid_tn, best_valid_fp, best_valid_fn, best_valid_tp, best_valid_precision, best_valid_recall = 0, 0, 0, 0, 0, 0
    train_losses = []
    eval_losses = []
    train_acc_list, eval_acc_list = [], []


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
            

        print(f'Train Acc: {train_acc * 100:.2f}%')
        print(f'Train Precision: {train_precision * 100:.2f}%')
        print(f'Train Recall: {train_recall * 100:.2f}%')

        print(f'Val. Acc: {valid_acc * 100:.2f}%')
        print(f'Val. Precision: {valid_precision * 100:.2f}%')
        print(f'Val. Recall: {valid_recall * 100:.2f}%')

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
    performance_dict['note'] = 'ciaone'
    performance_dict['Time'] = str(datetime.datetime.now())
    performance_dict['device'] = torch.cuda.get_device_name(device)
    
    for key, value in performance_dict.items():
        if type(value) is np.int64:
            performance_dict[key] = int(value)

    performance_file = 'performance/performance_EFRA_news.txt'
    with open(performance_file, 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')

    if not args.save_model:
        shutil.rmtree(modeldir)

if __name__ == '__main__':
    main()