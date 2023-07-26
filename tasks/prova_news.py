import numpy as np
import pandas as pd

import torch

from transformers import AutoTokenizer, AutoConfig, AdamW

# from TRC.custom_parser import my_parser
from TRC.utils import tokenize_with_new_mask, train, evaluate, load_local_TRC_model

from common_utils import extract_from_dataframe, mask_batch_generator, mask_batch_seq_generator

def split_df(data):
    random_indices = np.random.permutation(data.index)

    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    num_samples = len(data)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)

    train_indices = random_indices[:num_train]
    val_indices = random_indices[num_train:num_train + num_val]
    test_indices = random_indices[num_train + num_val:]

    train_set = data.loc[train_indices]
    val_set = data.loc[val_indices]
    test_set = data.loc[test_indices]
    
    return train_set, val_set, test_set

def main():

    data_path = '/home/agensale/rora_tesi_new/data/SampleAgroknow/mixed_news.p'
    news = pd.read_pickle(data_path)
    train_news, val_news, test_news = split_df(news)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    model_name = 'roberta-large'

    model_path = '/home/agensale/rora_tesi/log_rora_tesi/log-tweet-classification/roberta-large/bertweet-seq/24_epoch/data/True_weight/42_seed/saved-model/pytorch_model.bin'
    config_path = '/home/agensale/rora_tesi/log_rora_tesi/log-tweet-classification/roberta-large/bertweet-seq/24_epoch/data/True_weight/42_seed/saved-model/config.json'

    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization = True)
    model = load_local_TRC_model(model_path, config_path, device, model_name)
    model = model.to(device)

    need_columns = ['tokens_clean', 'sentence_class']

    X_train_raw, Y_train = extract_from_dataframe(train_news, need_columns)
    X_dev_raw, Y_dev = extract_from_dataframe(val_news, need_columns)
    X_test_raw, Y_test = extract_from_dataframe(test_news, need_columns)
    
    eval_batch_size = 32
    test_batch_size = 32

    max_length = 128

    X_train, masks_train = tokenize_with_new_mask(X_train_raw, max_length, tokenizer)
    X_dev, masks_dev = tokenize_with_new_mask(X_dev_raw, max_length, tokenizer)
    X_test, masks_test = tokenize_with_new_mask(X_test_raw, max_length, tokenizer)

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

    batch_size = 32
    n_epochs = 6

    for epoch in range(n_epochs):

        print(f'########## EPOCH {epoch+1} ##########')
        

        # train
        train_batch_generator = mask_batch_generator(X_train, Y_train, masks_train, batch_size)
        num_batches = X_train.shape[0] // batch_size
        train_loss, train_auc, train_acc, train_tn, train_fp, train_fn, train_tp, train_precision, train_recall = train(model, optimizer,
                                                                                         train_batch_generator,
                                                                                         num_batches, device,
                                                                                         class_weight)
        train_losses.append(train_loss)
        train_acc_list.append(train_acc)

        # eval
        dev_batch_generator = mask_batch_seq_generator(X_dev, Y_dev, masks_dev,
                                                       min(X_dev.shape[0], eval_batch_size))
        num_batches = X_dev.shape[0] // min(X_dev.shape[0], eval_batch_size)
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

        print(f'Train Acc: {train_acc * 100:.2f}%')
        print(f'Train Precision: {train_precision * 100:.2f}%')
        print(f'Train Recall: {train_recall * 100:.2f}%')

        print(f'Val. Acc: {valid_acc * 100:.2f}%')
        print(f'Val. Precision: {valid_precision * 100:.2f}%')
        print(f'Val. Recall: {valid_recall * 100:.2f}%')

    print(f"After {epoch + 1} epoch, Best valid accuracy: {best_valid_acc}, Best valid precision: {best_valid_precision* 100:.2f}%, Best valid recall: {best_valid_recall* 100:.2f}%")



    num_batches = X_test.shape[0] // test_batch_size
    test_batch_generator = mask_batch_seq_generator(X_test, Y_test, masks_test, test_batch_size)

    logits, y_batch, test_loss, test_auc, test_acc, test_tn, test_fp, test_fn, test_tp, test_precision, test_recall, test_s_pred = evaluate(model,
                                                                                              test_batch_generator,
                                                                                              num_batches, device,
                                                                                              class_weight)
    



    content = f'Test Acc: {test_acc * 100:.2f}%, AUC: {test_auc * 100:.2f}%, TN: {test_tn}, FP: {test_fp}, FN: {test_fn}, TP: {test_tp}, Precision: {test_precision* 100:.2f}%, Recall: {test_recall* 100:.2f}%'
    print(content)

if __name__ == '__main__':
    main()