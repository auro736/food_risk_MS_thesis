import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import transformers
import datasets
import shap

from common_utils import extract_from_dataframe

# load the emotion dataset
# dataset  = datasets.load_dataset("emotion", split = "train")

test_data = pd.read_pickle('/home/cc/rora_tesi_new/data/test.p')

# data = pd.DataFrame({'text':dataset['text'],'emotion':dataset['label']})

need_columns = ['tweet_tokens', 'sentence_class']
X_test_raw, Y_test = extract_from_dataframe(test_data, need_columns)

# load the model and tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion", use_fast=True)
model = transformers.AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion").cuda()
labels = sorted(model.config.label2id, key=model.config.label2id.get)

# this defines an explicit python function that takes a list of strings and outputs scores for each class
def f(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=128, truncation=True) for v in x]).cuda()
    attention_mask = (tv!=0).type(torch.int64).cuda()
    outputs = model(tv,attention_mask=attention_mask)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores)
    return val

explainer = shap.Explainer(f, tokenizer, output_names=labels)

shap_values = explainer(data['text'][:3])

shap.plots.text(shap_values)