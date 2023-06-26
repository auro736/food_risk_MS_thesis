import os
import json
import random
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoConfig, AdamW

from EMD.models import *
from EMD.utils import tokenize_with_new_mask

from EFRA.custom_parser import my_parser

from common_utils import extract_from_dataframe


SEED = 42
ASSIGN_WEIGHT = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_local_model(model_path, config_path, device, model_name):

    config = config = AutoConfig.from_pretrained(config_path)
    if 'deberta' in model_name:
        print('deberta')
        model = ModelForTokenClassificationWithCRFDeberta(model_name=model_name,config=config)
    else:
        model = ModelForTokenClassificationWithCRF(model_name=model_name,config=config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    return model

def main():

    args = my_parser()
    
    train_inc = pd.read_pickle('/home/cc/rora_tesi_new/data/SampleAgroknow/train_inc.p')
    val_inc = pd.read_pickle('/home/cc/rora_tesi_new/data/SampleAgroknow/val_inc.p')
    test_inc = pd.read_pickle('/home/cc/rora_tesi_new/data/SampleAgroknow/test_inc.p')

    # print(train_inc.head())

    need_columns = ['tokens']
    if args.task_type == 'entity_detection':
        need_columns.append('entity_label')
    # elif args.task_type == 'relevant_entity_detection':
    #     need_columns.append('relevant_entity_label')
    # elif args.task_type == 'entity_relevance_classification':
    #     need_columns.append('relevance_entity_class_label')
    need_columns.append('sentence_class')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    model_name = 'microsoft/deberta-v3-large'

    model_path = '/home/cc/rora_tesi_new/log/log_EMD/deberta-v3-large/bertweet-token-crf/20_epoch/data/True_weight/42_seed/saved-model/pytorch_model.bin'
    config_path = '/home/cc/rora_tesi_new/log/log_EMD/deberta-v3-large/bertweet-token-crf/20_epoch/data/True_weight/42_seed/saved-model/config.json'

    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization = True)
    model = load_local_model(model_path, config_path, device, model_name)
    model = model.to(device)

    X_train_raw, Y_train_raw, seq_train = extract_from_dataframe(train_inc, need_columns)
    X_dev_raw, Y_dev_raw, seq_dev = extract_from_dataframe(val_inc, need_columns)
    X_test_raw, Y_test_raw, seq_test = extract_from_dataframe(test_inc, need_columns)
    args.eval_batch_size = seq_dev.shape[0]
    args.test_batch_size = seq_test.shape[0]

    with open(os.path.join(args.data, args.label_map), 'r') as fp:
        label_map = json.load(fp)

    labels = list(label_map.keys())

    X_train, masks_train, Y_train = tokenize_with_new_mask(X_train_raw, args.max_length, tokenizer, Y_train_raw, label_map)
    X_dev, masks_dev, Y_dev = tokenize_with_new_mask(X_dev_raw, args.max_length, tokenizer, Y_dev_raw, label_map)
    X_test, masks_test, Y_test = tokenize_with_new_mask(X_test_raw, 128, tokenizer, Y_test_raw, label_map)

    print('qua')

    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization = True)
    config = AutoConfig.from_pretrained(config_path)
    config.update({'num_labels': len(labels), })
    model = load_local_model(model_path, config, device, model_name)
    model = model.to(device)

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

if __name__ == '__main__':
    main()