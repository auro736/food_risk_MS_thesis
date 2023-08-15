import numpy as np
import pandas as pd

np.random.seed(42)

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

    data_path_news = '/home/cc/rora_tesi_new/data/SampleAgroknow/news_updated.p'
    data_path_incidents = '/home/cc/rora_tesi_new/data/SampleAgroknow/incidents_words.p'

    news = pd.read_pickle(data_path_news)
    train_news, val_news, test_news = split_df(news)

    train_news.to_pickle('/home/cc/rora_tesi_new/data/SampleAgroknow/News/train_news.p')
    val_news.to_pickle('/home/cc/rora_tesi_new/data/SampleAgroknow/News/val_news.p')
    test_news.to_pickle('/home/cc/rora_tesi_new/data/SampleAgroknow/News/test_news.p')

    incidents = pd.read_pickle(data_path_incidents)
    train_inc, val_inc, test_inc = split_df(incidents)

    train_inc.to_pickle('/home/cc/rora_tesi_new/data/SampleAgroknow/Incidents/train_inc.p')
    val_inc.to_pickle('/home/cc/rora_tesi_new/data/SampleAgroknow/Incidents/val_inc.p')
    test_inc.to_pickle('/home/cc/rora_tesi_new/data/SampleAgroknow/Incidents/test_inc.p')

if __name__ == '__main__':
    main()
