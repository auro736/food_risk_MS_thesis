import pandas as pd

import torch
from transformers import AutoTokenizer

# from error_analysis_TRC import load_local_TRC_model
from TRC.utils import tokenize_with_new_mask, evaluate, load_local_TRC_model
from common_utils import extract_from_dataframe, mask_batch_seq_generator

# PRENDI NEWS, METTI COLONNA 1, E TESTA CON MODELLO FINETUNED SU TRC 

SEED = 42
MAX_LENGTH = 128

def main():

    data_path = '/home/cc/rora_tesi_new/data/SampleAgroknow/news.csv'
    news = pd.read_csv(data_path, index_col=0)
    news['year'] = news.apply(lambda x: x['date'].split('-')[0], axis = 1)
    news['month'] = news.apply(lambda x: x['date'].split('-')[1], axis = 1)
    news['relevance'] = 1


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    model_name = 'cardiffnlp/twitter-roberta-large-2022-154m'

    model_path = '/home/cc/rora_tesi_new/log/log_TRC/roberta-large/bertweet-seq/20_epoch/data/True_weight/42_seed/saved-model/pytorch_model.bin'
    config_path = '/home/cc/rora_tesi_new/log/log_TRC/roberta-large/bertweet-seq/20_epoch/data/True_weight/42_seed/saved-model/config.json'

    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization = True)
    model = load_local_TRC_model(model_path, config_path, device, model_name)
    model = model.to(device)

    need_colums = ['description', 'relevance']
    
    X_test_raw, Y_test = extract_from_dataframe(news, need_colums)
    test_batch_size = 4

    class_weight = None

    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization=True)
    X_test, masks_test = tokenize_with_new_mask(X_test_raw, MAX_LENGTH, tokenizer)

    num_batches = X_test.shape[0] // test_batch_size
    test_batch_generator = mask_batch_seq_generator(X_test, Y_test, masks_test, test_batch_size)

    
    logits, y_true, test_loss, test_auc, test_acc, test_tn, test_fp, test_fn, test_tp, test_precision, test_recall, test_s_pred = evaluate(model,
                                                                                                test_batch_generator,
                                                                                                num_batches, device,
                                                                                                class_weight )

    content = f'Test Acc: {test_acc * 100:.2f}%, AUC: {test_auc * 100:.2f}%, TN: {test_tn}, FP: {test_fp}, FN: {test_fn}, TP: {test_tp}, Precision: {test_precision* 100:.2f}%, Recall: {test_recall* 100:.2f}%'
    print(content)


if __name__ == '__main__':
    main()