import torch
from transformers import AutoTokenizer

import nltk
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split


SEED = 42

def create_entity(x):
    entity_list = list()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    for i,t in enumerate(x['token']):
        prod = [x.lower() for x in x['product'].split(' ')]
        suppl =  [x.lower() for x in x['supplier'].split(' ')]
#         haz = [x.lower() for x in x['hazard'].split(' ')]
        haz = x['hazard'].lower()
        t = t.lower()
        if t in prod and x['token'][i-1] not in prod  :
            entity_list.append('B-food')
        elif t in prod and x['token'][i-1] in prod:
            entity_list.append('I-food')
        elif t in suppl  and x['token'][i-1] not in suppl :
            entity_list.append('B-loc')
        elif t in suppl and x['token'][i-1] in suppl:
            entity_list.append('I-loc')
        elif t == haz:
            entity_list.append('B-haz')
#         elif t in haz and x['token'][i-1] in haz and t not in stop_words:
#             entity_list.append('B-haz')
#         elif t in haz and x['token'][i-1] in haz and t not in stop_words:
#             entity_list.append('I-haz')
        else:
            entity_list.append('O')
    return entity_list

def preprocess_incidents(df):

    df['year'] = df.apply(lambda x: x['date'].split('-')[0], axis = 1)
    df['month'] = df.apply(lambda x: x['date'].split('-')[1], axis = 1)

    df['description'] = df.apply(lambda x: ''.join(x['description'].splitlines()), axis = 1)
    df['description'] = df.apply(lambda x: " ".join(x['description'].split()) , axis = 1)
    df['token'] = [x.split(' ') for x in df['description'].values.tolist()]

    
    df['entity_label'] = df.apply(lambda x: create_entity(x), axis = 1)

    # ho annotato B/I food, B/I suppl, solo B hazards

    return df 

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    incidents_0 = pd.read_csv('/home/cc/rora_tesi_new/data/SampleAgroknow/incidents.csv', index_col = 0)
    print(len(incidents_0))

    incidents_annotated = preprocess_incidents(incidents_0)

    # incidents_annotated.to_csv('/home/cc/rora_tesi_new/data/SampleAgroknow/incidents_annotated.csv', header=True)

    incidents_annotated.to_pickle('/home/cc/rora_tesi_new/data/SampleAgroknow/incidents_annotated.pickle')


    # ASSIGN_WEIGHT = True

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # #device = "cpu"
    # model_name = 'cardiffnlp/twitter-roberta-large-2022-154m'

    # model_path = '/home/cc/rora_tesi_new/log/log_TRC/twitter-roberta-large-2022-154m/bertweet-seq/20_epoch/data/True_weight/42_seed/saved-model/pytorch_model.bin'
    # config_path = '/home/cc/rora_tesi_new/log/log_TRC/twitter-roberta-large-2022-154m/bertweet-seq/20_epoch/data/True_weight/42_seed/saved-model/config.json'

    # tokenizer = AutoTokenizer.from_pretrained(model_name, normalization = True)
    # model = load_local_model(model_path, config_path, device, model_name)
    # model = model.to(device)


if __name__ == '__main__':
    main()