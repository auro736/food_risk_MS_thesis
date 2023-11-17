import torch
import torch.nn as nn
from transformers import AutoTokenizer

import random
import numpy as np
import pandas as pd

from TRC.utils import tokenize_with_new_mask, evaluate, load_local_TRC_model, create_weight, calibration_plot, confusion_matrix_display, create_analysis_csv
# from TRC.utils_ea import *
from common_utils import mask_batch_seq_generator, extract_from_dataframe

from EFRA.utils import tokenize_with_new_mask_news

# TRC 

MAX_LENGTH = 128
ASSIGN_WEIGHT = True
SEED = 42


def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model_name = 'microsoft/deberta-v3-large'
    #log_directory = '/home/agensale/rora_tesi_new/log/log_EFRA/news/roberta-large/from_finetuned/10_epoch/SampleAgroknow/True_weight/42_seed/'
    log_directory = '/mnt/c/Users/auror/Desktop/rora_tesi_new/log/log_TRC/deberta-v3-large/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #model_path = '/home/agensale/rora_tesi_new/log/log_EFRA/news/roberta-large/from_finetuned/10_epoch/SampleAgroknow/True_weight/42_seed/saved-model/pytorch_model.bin'
    #config_path = '/home/agensale/rora_tesi_new/log/log_EFRA/news/roberta-large/from_finetuned/10_epoch/SampleAgroknow/True_weight/42_seed/saved-model/config.json'

    model_path = '/mnt/c/Users/auror/Desktop/deberta-v3-large-tweet-fid-TRC/pytorch_model.bin'
    config_path = '/mnt/c/Users/auror/Desktop/deberta-v3-large-tweet-fid-TRC/config.json'

    model = load_local_TRC_model(model_path, config_path, device, model_name)
    
    model = model.to(device)

    #train_data_path = '/home/agensale/rora_tesi_new/data/SampleAgroknow/News/news_train_EN.p'
    #test_data_path = '/home/agensale/rora_tesi_new/data/SampleAgroknow/News/news_test_EN.p'
    
    train_data_path = '/mnt/c/Users/auror/Desktop/rora_tesi_new/data/Tweet-Fid/train.p'
    test_data_path = '/mnt/c/Users/auror/Desktop/rora_tesi_new/data/Tweet-Fid/test.p'

    need_columns = ['tweet','tweet_tokens', 'sentence_class']

    #need_columns = ['description','words', 'sentence_class']

    train_ds = pd.read_pickle(train_data_path)
    test_ds = pd.read_pickle(test_data_path)

    descr_train ,_, Y_train = extract_from_dataframe(train_ds, need_columns)
    descr_test, X_test_raw, Y_test = extract_from_dataframe(test_ds, need_columns)

    # descr_train, Y_train = descr_train[:100], Y_train[:100]
    # descr_test, X_test_raw, Y_test = descr_test[:100], X_test_raw[:100], Y_test[:100]

    # _, _, Y_train = extract_from_dataframe(train_ds, need_columns)
    # tweet_test, X_test_raw, Y_test = extract_from_dataframe(test_ds, need_columns)

    # _, _, Y_train = prepare_data(train_data_path, need_columns)
    # tweet_test, X_test_raw, Y_test = prepare_data(test_data_path, need_columns)

    # test_batch_size = Y_test.shape[0]
    test_batch_size = 32


    class_weight = None
    if ASSIGN_WEIGHT:
        class_weight = create_weight(Y_train)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization=True)
    X_test, masks_test = tokenize_with_new_mask(X_test_raw, MAX_LENGTH, tokenizer)
    #X_test, masks_test = tokenize_with_new_mask_news(X_test_raw, MAX_LENGTH, tokenizer)


    num_batches = X_test.shape[0] // test_batch_size
    test_batch_generator = mask_batch_seq_generator(X_test, Y_test, masks_test, test_batch_size)

    _, y_true, test_loss, test_auc, test_acc, test_tn, test_fp, test_fn, test_tp, test_precision, test_recall, test_s_pred = evaluate(model,
                                                                                                test_batch_generator,
                                                                                                num_batches, device,
                                                                                                class_weight )

    content = f'Test Acc: {test_acc * 100:.2f}%, AUC: {test_auc * 100:.2f}%, TN: {test_tn}, FP: {test_fp}, FN: {test_fn}, TP: {test_tp}, Precision: {test_precision* 100:.2f}%, Recall: {test_recall* 100:.2f}%'
    print(content)

    softmax = nn.Softmax(dim=1)
    tensor_preds = torch.from_numpy(test_s_pred)
    probabilities = softmax(tensor_preds)

    cal_path = log_directory + 'calibration_curve.png'
    calibration_plot(probabilities, y_true = y_true, img_path=cal_path, model_name=model_name)

    conf_path = log_directory + 'confusion_matrix.png'
    confusion_matrix_display(probabilities, y_true=y_true, model_name =model_name, path = conf_path)

    df_errati, df_corretti = create_analysis_csv(probabilities=probabilities, tweet_test=descr_test, tweet_token = X_test_raw, y_true=y_true, data_type = 'tweets')

    #df_errati, df_corretti = create_analysis_csv(probabilities=probabilities, tweet_test=descr_test, tweet_token = X_test_raw, y_true=y_true, data_type = 'news')
    df_errati.to_csv(log_directory+'tweet_errati.csv', header= True, index = True)
    #df_errati.to_csv(log_directory+'news_errate.csv', header= True, index = True)
    df_errati.head()

    df_corretti.to_csv(log_directory+'tweet_corretti.csv', header= True, index = True)
    #df_corretti.to_csv(log_directory+'news_corrette.csv', header= True, index = True)
    df_corretti.head()

    # tokens_1 = df_corretti['Tweet tok'][0]
    # print('aaaaaa', type(tokens_1))

    # df_errati.to_pickle(log_directory+'tweet_errati.pkl')
    # df_corretti.to_pickle(log_directory+'tweet_corretti.pkl')

if __name__ == '__main__':
    main()
