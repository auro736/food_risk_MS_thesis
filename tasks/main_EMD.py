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

from EMD.custom_parser import my_parser
from EMD.utils import tokenize_with_new_mask, load_model, train, evaluate, predict

from common_utils import extract_from_dataframe, mask_batch_generator, mask_batch_seq_generator

NOTE = 'hello'

def main():

    args = my_parser()

    assert args.task_type in ['entity_detection', 'relevant_entity_detection', 'entity_relevance_classification']

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

    need_columns = ['tweet_tokens']
    need_columns.append('entity_label')
    # if args.task_type == 'entity_detection':
    #     need_columns.append('entity_label')
    # elif args.task_type == 'relevant_entity_detection':
    #     need_columns.append('relevant_entity_label')
    # elif args.task_type == 'entity_relevance_classification':
    #     need_columns.append('relevance_entity_class_label')
    need_columns.append('sentence_class')

    X_train_raw, Y_train_raw, seq_train = extract_from_dataframe(train_data, need_columns)
    X_dev_raw, Y_dev_raw, seq_dev = extract_from_dataframe(val_data, need_columns)
    X_test_raw, Y_test_raw, seq_test = extract_from_dataframe(test_data, need_columns)
    args.eval_batch_size = seq_dev.shape[0]
    args.test_batch_size = seq_test.shape[0]

    with open(os.path.join(args.data, args.label_map), 'r') as fp:
        label_map = json.load(fp)

    labels = list(label_map.keys())

    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, normalization=True)
    
    # X_train = embedding dei token 
    # masks_train = maschera con 0 e 1, 
    # in masks_train[i] ho 1 dove in X_train[i] ho un umbedding con valore > 0
    # Y_train[i] = label numeriche per ogni token in X_train[i]

    X_train, masks_train, Y_train = tokenize_with_new_mask(X_train_raw, args.max_length, tokenizer, Y_train_raw, label_map)
    X_dev, masks_dev, Y_dev = tokenize_with_new_mask(X_dev_raw, args.max_length, tokenizer, Y_dev_raw, label_map)
    X_test, masks_test, Y_test = tokenize_with_new_mask(X_test_raw, 128, tokenizer, Y_test_raw, label_map)
    
    # weight of each class in loss function
    class_weight = None
    if args.assign_weight: # default True
        class_weight = [Y_train.shape[0] / (Y_train == i).sum() for i in range(len(labels))]
        class_weight = torch.FloatTensor(class_weight)


    config = AutoConfig.from_pretrained(args.bert_model)
    config.update({'num_labels': len(labels), })

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
    best_valid_P, best_valid_R, best_valid_F = 0, 0, 0
    train_losses = []
    eval_losses = []
    train_F_list, eval_F_list = [], []

    early_stop_sign = 0

    for epoch in range(args.n_epochs):

        print('########## EPOCH: ', epoch+1, '##########')

        # train

        train_batch_generator = mask_batch_generator(X_train, Y_train, masks_train, args.batch_size)
        num_batches = X_train.shape[0] // args.batch_size
           
        train_loss, train_acc, train_results, train_results_by_tag, train_CR = train(model, optimizer,
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
        dev_batch_generator = mask_batch_seq_generator(X_dev, Y_dev, masks_dev,
                                                       min(X_dev.shape[0], args.eval_batch_size))
        num_batches = X_dev.shape[0] // min(X_dev.shape[0], args.eval_batch_size)
        valid_loss, valid_acc, valid_results, valid_results_by_tag, valid_t_pred, valid_CR = evaluate(model,
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

            model.save_pretrained(modeldir)
            
            if args.early_stop:
                early_stop_sign = 0
        elif args.early_stop:
            early_stop_sign += 1

        print(f'Train Acc: {train_acc * 100:.2f}%')
        print(f'Train P: {train_P * 100:.2f}%')
        print(f'Train R: {train_R * 100:.2f}%')
        print(f'Train F1: {train_F * 100:.2f}%')
        print(f'Val. Acc: {valid_acc * 100:.2f}%')
        print(f'Val. P: {valid_P * 100:.2f}%')
        print(f'Val. R: {valid_R * 100:.2f}%')
        print(f'Val. F1: {valid_F * 100:.2f}%')
        
        if args.early_stop and early_stop_sign >= 5:
            break

    content = f"After {epoch + 1} epoch, Best valid F1: {best_valid_F}, accuracy: {best_valid_acc}, Recall: {best_valid_R}, Precision: {best_valid_P}"
    print(content)

    performance_dict = vars(args)
    performance_dict['T_best_train_F'] = best_train_F
    performance_dict['T_best_train_ACC'] = best_train_acc
    performance_dict['T_best_train_R'] = best_train_R
    performance_dict['T_best_train_P'] = best_train_P
    performance_dict['T_best_train_CR'] = best_train_CR
    performance_dict['T_best_train_results'] = best_train_results
    performance_dict['T_best_train_results_by_tag'] = best_train_results_by_tag

    performance_dict['T_best_valid_F'] = best_valid_F
    performance_dict['T_best_valid_ACC'] = best_valid_acc
    performance_dict['T_best_valid_R'] = best_valid_R
    performance_dict['T_best_valid_P'] = best_valid_P
    performance_dict['T_best_valid_CR'] = best_valid_CR
    performance_dict['T_best_valid_results'] = best_valid_results
    performance_dict['T_best_valid_results_by_tag'] = best_valid_results_by_tag


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
    figfullname = log_directory + figure_filename
    plt.savefig(figfullname, dpi=fig.dpi)

    num_batches = X_test.shape[0] // args.test_batch_size
    test_batch_generator = mask_batch_seq_generator(X_test, Y_test, masks_test, args.test_batch_size)

    _, test_loss, test_acc, test_results, test_results_by_tag, test_t_pred, test_CR = predict(model,
                                                                                           test_batch_generator,
                                                                                           num_batches, device,
                                                                                           label_map, class_weight)
    test_F = test_results['strict']['f1']
    test_P = test_results['strict']['precision']
    test_R = test_results['strict']['recall']
    print(f'Test Acc: {test_acc * 100:.2f}%')
    print(f'Test P: {test_P * 100:.2f}%')
    print(f'Test R: {test_R * 100:.2f}%')
    print(f'Test F1: {test_F * 100:.2f}%')
    token_pred_dir = log_directory + 'token_prediction.npy'
    np.save(token_pred_dir, test_t_pred)

    performance_dict['T_best_test_F'] = test_F
    performance_dict['T_best_test_ACC'] = test_acc
    performance_dict['T_best_test_R'] = test_R
    performance_dict['T_best_test_P'] = test_P
    performance_dict['T_best_test_CR'] = test_CR
    performance_dict['T_best_test_results'] = test_results
    performance_dict['T_best_test_results_by_tag'] = test_results_by_tag

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
