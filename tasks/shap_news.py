import torch
from transformers import AutoTokenizer

import shap
import nltk
import pickle
import string
import pandas as pd

from TRC.utils import  load_local_TRC_model


from better_profanity import profanity
from copy import deepcopy
from ast import literal_eval

ASSIGN_WEIGHT = True

def flatten_lists(row):
    # Utilizziamo una list comprehension per appiattire le liste di liste mantenendo le stringhe intatte
    return [item for sublist in row for item in sublist]

# come input prende una lista di tweet


def clean_strings(big_tokens_list):
    tags = ['@USER','USER', 'HTTPURL', 'HTTP', 'URL']
    nltk.download('stopwords')
    profanity.load_censor_words()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    clean_list = []
    for t_list in big_tokens_list:
        tokens = deepcopy(t_list)
        for token, i in zip(t_list, range(len(t_list))):
            tmp = deepcopy(token)
            tmp = tmp.strip()
            tmp = tmp.lower()
            if tmp in string.punctuation or \
            tmp.upper() in tags or \
            tmp == '...' or  tmp == '....' or \
            tmp in stop_words or \
            profanity.contains_profanity(tmp) or \
            len(tmp) < 3:
                tokens.remove(token)
        clean_list.append(' '.join(tokens))
    return clean_list

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    model_name = 'roberta-large'

    model_path = '/home/agensale/rora_tesi_new/log/log_EFRA/news/roberta-large/from_finetuned/10_epoch/SampleAgroknow/True_weight/42_seed/saved-model/pytorch_model.bin'
    config_path = '/home/agensale/rora_tesi_new/log/log_EFRA/news/roberta-large/from_finetuned/10_epoch/SampleAgroknow/True_weight/42_seed/saved-model/config.json'

    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization = True)
    model = load_local_TRC_model(model_path, config_path, device, model_name)
    model = model.to(device)

    def f(x):
    # x = [tweet_test[0]]
        tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=128, truncation=True) for v in x]).to(device)
        # print('tv',tv.shape)
        attention_mask = (tv!=0).type(torch.int64).to(device)
        # print('atte',attention_mask.shape)
        model.eval()
        with torch.no_grad():
            outputs = model(tv, attention_mask, class_weight=None)
        # print(outputs)
        return outputs['logits']

    explainer = shap.Explainer(f, masker = tokenizer)
    print(explainer)

    news_corrette = pd.read_csv('/home/agensale/rora_tesi_new/log/log_EFRA/news/roberta-large/from_finetuned/10_epoch/SampleAgroknow/True_weight/42_seed/news_corrette.csv', index_col = 0, converters={'News tok':literal_eval})
    news_errate = pd.read_csv('/home/agensale/rora_tesi_new/log/log_EFRA/news/roberta-large/from_finetuned/10_epoch/SampleAgroknow/True_weight/42_seed/news_errate.csv', index_col = 0, converters={'News tok':literal_eval})
    news_total = pd.concat([news_corrette, news_errate])
    news_total = news_total.sample(n=1500, random_state=42, ignore_index= True)

    news_total['lista_piatta'] = news_total['News tok'].apply(flatten_lists)

    all_tweets_tokens = news_total['lista_piatta'].tolist()
    cleaned_all = clean_strings(all_tweets_tokens)

    # CREAZIONE SHAP_VALUES E SALVATAGGIO IN PICKLE


    save_path = '/home/agensale/rora_tesi_new/log/log_EFRA/news/roberta-large/from_finetuned/10_epoch/SampleAgroknow/True_weight/42_seed/shap/'
    shap_values = explainer(cleaned_all)
    print(shap_values.shape)
    print(shap_values.output_names)
    with open(save_path + 'shap_values_all_test.pickle', 'wb') as handle:
        pickle.dump(shap_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':
    main()