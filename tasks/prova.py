import torch
import random
import numpy as np
import pandas as pd

from transformers import AutoConfig, AutoTokenizer
from TRC.utils import load_model, tokenize_with_new_mask, evaluate
from common_utils import extract_from_dataframe, mask_batch_seq_generator
from TRC.custom_parser import my_parser

# PROVA TRC 
BERT_MODEL = 'cardiffnlp/twitter-roberta-large-2022-154m'
MAX_LENGTH = 128
ASSIGN_WEIGHT = True
SEED = 42

def main():

    args = my_parser()
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_model_path = '/home/cc/rora_tesi_new/log/log_TRC/twitter-roberta-large-2022-154m/bertweet-seq/20_epoch/data/True_weight/42_seed/saved-model/pytorch_model.bin'
    config_path = '/home/cc/rora_tesi_new/log/log_TRC/twitter-roberta-large-2022-154m/bertweet-seq/20_epoch/data/True_weight/42_seed/saved-model/config.json'

    config = AutoConfig.from_pretrained(config_path)
    model = load_model('bertweet-seq', saved_model_path, config)

    print(model.__class__.__name__)

    print(model.config.label2id)
    labels = sorted(model.config.label2id, key=model.config.label2id.get)
    print(labels)

    model = model.to(device)

    train_data_path = '/home/cc/rora_tesi_new/data/train.p'
    test_data_path = '/home/cc/rora_tesi_new/data/test.p'
    train_data = pd.read_pickle(train_data_path)
    test_data = pd.read_pickle(test_data_path)

    need_columns = ['tweet_tokens', 'sentence_class']
    X_test_raw, Y_test = extract_from_dataframe(test_data, need_columns)
    test_batch_size = Y_test.shape[0]
    test_batch_size = 16
    print(test_batch_size)

    _, Y_train = extract_from_dataframe(train_data, need_columns)
    # weight of each class in loss function
    class_weight = None
    if ASSIGN_WEIGHT:
        class_weight = [np.array(Y_train).shape[0] / (np.array(Y_train) == i).sum() for i in range(len(set(Y_train)))]
        class_weight = torch.FloatTensor(class_weight)

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, normalization=True)
    X_test, masks_test = tokenize_with_new_mask(X_test_raw, MAX_LENGTH, tokenizer)

    num_batches = X_test.shape[0] // test_batch_size
    test_batch_generator = mask_batch_seq_generator(X_test, Y_test, masks_test, test_batch_size)

    test_loss, test_auc, test_acc, test_tn, test_fp, test_fn, test_tp, test_precision, test_recall, test_s_pred = evaluate(model,
                                                                                                test_batch_generator,
                                                                                                num_batches, device,
                                                                                                class_weight, 
                                                                                                split = 'test')

    content = f'Test Acc: {test_acc * 100:.2f}%, AUC: {test_auc * 100:.2f}%, TN: {test_tn}, FP: {test_fp}, FN: {test_fn}, TP: {test_tp}, Precision: {test_precision* 100:.2f}%, Recall: {test_recall* 100:.2f}%'
    print(content)

if __name__ == '__main__':
    main()