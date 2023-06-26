import torch
from transformers import AutoTokenizer

import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

from TRC.utils_ea import load_local_model

SEED = 42

def preprocess(df):

    pass

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    news = pd.read_csv('/home/cc/rora_tesi_new/data/SampleAgroknow/news.csv', index_col = 0)
    print(len(news))

    news['year'] = news.apply(lambda x: x['date'].split('-')[0], axis = 1)
    news['month'] = news.apply(lambda x: x['date'].split('-')[1], axis = 1)

    # print(news.head())
    news['relevance'] = 1
    # print(news.head())

    ASSIGN_WEIGHT = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    model_name = 'cardiffnlp/twitter-roberta-large-2022-154m'

    model_path = '/home/cc/rora_tesi_new/log/log_TRC/twitter-roberta-large-2022-154m/bertweet-seq/20_epoch/data/True_weight/42_seed/saved-model/pytorch_model.bin'
    config_path = '/home/cc/rora_tesi_new/log/log_TRC/twitter-roberta-large-2022-154m/bertweet-seq/20_epoch/data/True_weight/42_seed/saved-model/config.json'

    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization = True)
    model = load_local_model(model_path, config_path, device, model_name)
    model = model.to(device)

    news_train, news_test = train_test_split(news, test_size=0.25)


if __name__ == '__main__':
    main()