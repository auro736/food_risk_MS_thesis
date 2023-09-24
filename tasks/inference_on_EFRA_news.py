import torch
from transformers import AutoTokenizer

import os
import json
import random
import datetime
import numpy as np
import pandas as pd

from common_utils import extract_from_dataframe, mask_batch_seq_generator

from TRC.utils import load_local_TRC_model, evaluate

from EFRA.custom_parser import my_parser
from EFRA.utils import tokenize_with_new_mask_news


NOTE = 'Inferenza modelli tweet su news'

def main():

    #INFERENCE ON NEWS WITH MODEL ON TWITTER TRC 

    args = my_parser()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.bert_model

    train_news = pd.read_pickle('/home/agensale/rora_tesi_new/data/SampleAgroknow/News/news_train_EN.p')
    test_news = pd.read_pickle('/home/agensale/rora_tesi_new/data/SampleAgroknow/News/news_test_EN.p')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model_name = args.bert_model

    print(args.max_length)

    need_columns = ['words', 'sentence_class']

    _, Y_train = extract_from_dataframe(train_news, need_columns)
    X_test_raw, Y_test = extract_from_dataframe(test_news, need_columns)

    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization = True)

    model_path = args.saved_model_path + 'pytorch_model.bin'
    config_path = args.saved_model_path + 'config.json'
    
    args.from_finetuned = 'True'
    model = load_local_TRC_model(model_path, config_path, device, model_name)

    model = model.to(device)

    # X_train_raw, Y_train = X_train_raw[:100], Y_train[:100]
    # X_test_raw, Y_test = X_test_raw[:100], Y_test[:100]

    X_test, masks_test = tokenize_with_new_mask_news(X_test_raw, args.max_length, tokenizer)

    class_weight = None
    if args.assign_weight:
        class_weight = [np.array(Y_train).shape[0] / (np.array(Y_train) == i).sum() for i in range(len(set(Y_train)))]
        class_weight = torch.FloatTensor(class_weight)
    
    num_batches = X_test.shape[0] // args.test_batch_size
    test_batch_generator = mask_batch_seq_generator(X_test, Y_test, masks_test, args.test_batch_size)

    logits, y_batch, test_loss, test_auc, test_acc, test_tn, test_fp, test_fn, test_tp, test_precision, test_recall, test_s_pred = evaluate(model,
                                                                                              test_batch_generator,
                                                                                              num_batches, device,
                                                                                              class_weight)
    
    content = f'Test Acc: {test_acc * 100:.2f}%, AUC: {test_auc * 100:.2f}%, TN: {test_tn}, FP: {test_fp}, FN: {test_fn}, TP: {test_tp}, Precision: {test_precision* 100:.2f}%, Recall: {test_recall* 100:.2f}%'
    print(content)
    
    performance_dict = vars(args)

    performance_dict['S_best_test_AUC'] = test_auc
    performance_dict['S_best_test_ACC'] = test_acc
    performance_dict['S_best_test_TN'] = test_tn
    performance_dict['S_best_test_FP'] = test_fp
    performance_dict['S_best_test_FN'] = test_fn
    performance_dict['S_best_test_TP'] = test_tp

    performance_dict['S_best_test_precision'] = test_precision
    performance_dict['S_best_test_recall'] = test_recall

    performance_dict['script_file'] = os.path.basename(__file__)
    performance_dict['note'] = NOTE
    performance_dict['Time'] = str(datetime.datetime.now())
    #performance_dict['device'] = torch.cuda.get_device_name(device)
    
    for key, value in performance_dict.items():
        if type(value) is np.int64:
            performance_dict[key] = int(value)

    performance_file = '/home/agensale/rora_tesi_new/performance/inferenze_EFRA_news.txt'
    with open(performance_file, 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')

if __name__ == '__main__':
    main()

