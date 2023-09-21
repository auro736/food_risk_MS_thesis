import os
import json
import random
import shutil
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoConfig, AdamW

from EMD.models import *
from EMD.utils import train, evaluate, predict, load_model, load_local_EMD_model

from EFRA.custom_parser import my_parser
from EFRA.utils import tokenize_with_new_mask_inc, tokenize_with_new_mask_inc_train

from common_utils import extract_from_dataframe, mask_batch_generator, mask_batch_seq_generator



SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():

    args = my_parser()

    incidents_train = pd.read_pickle('/home/agensale/rora_tesi_new/data/SampleAgroknow/Incidents/inc_train_EN_annotati.p')
    incidents_test = pd.read_pickle('/home/agensale/rora_tesi_new/data/SampleAgroknow/Incidents/inc_test_EN_annotati.p')

    need_columns = ['words', 'entity_label']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(str(device).upper())

    args.from_finetuned = 'True'
    model_name = args.bert_model

    X_train_raw, Y_train_raw = extract_from_dataframe(incidents_train, need_columns)
    X_test_raw, Y_test_raw = extract_from_dataframe(incidents_test, need_columns)

    with open(os.path.join(args.data, args.label_map), 'r') as fp:
        label_map = json.load(fp)

    labels = list(label_map.keys())

    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization = True)

    model_path = args.saved_model_path + 'pytorch_model.bin'
    config_path = args.saved_model_path + 'config.json'

    model, config = load_local_EMD_model(model_path, config_path, device, model_name)
    model.config.update({'num_labels': len(labels), })
    model.num_labels = config.num_labels
    model.classifier = nn.Linear(config.hidden_size, config.num_labels)
    model.crf = CRF(num_tags=config.num_labels, batch_first=True)
    print(model.num_labels)
    print(model.classifier)
    print(model.crf)

    model = model.to(device)

    # X_train_raw, Y_train_raw = X_train_raw[:10], Y_train_raw[:10]
    # X_test_raw, Y_test_raw = X_test_raw[:10], Y_test_raw[:10]

    X_train, masks_train, Y_train = tokenize_with_new_mask_inc_train(X_train_raw, args.max_length, tokenizer, Y_train_raw, label_map)
    X_test, masks_test, Y_test = tokenize_with_new_mask_inc(X_test_raw, args.max_length, tokenizer, Y_test_raw, label_map)

    # weight of each class in loss function
    class_weight = None
    args.assign_weight = 'True'
    if args.assign_weight: # default True
        class_weight = [Y_train.shape[0] / (Y_train == i).sum() for i in range(len(labels))]
        class_weight = torch.FloatTensor(class_weight)

    num_batches = X_test.shape[0] // args.test_batch_size
    test_batch_generator = mask_batch_seq_generator(X_test, Y_test, masks_test, args.test_batch_size)

    _, test_loss, test_acc, test_results, test_results_by_tag, test_t_pred, test_CR = predict(model,
                                                                                           test_batch_generator,
                                                                                           num_batches, device,
                                                                                           label_map, class_weight)
    
    test_F = test_results['strict']['f1']
    test_P = test_results['strict']['precision']
    test_R = test_results['strict']['recall']
    print(f'Test Acc: {test_acc * 100:.2f}%')
    print(f'Test P: {test_P * 100:.2f}%')
    print(f'Test R: {test_R * 100:.2f}%')
    print(f'Test F1 Strict: {test_F * 100:.2f}%')
    print('Test F1 Ent Type:' , test_results['ent_type']['f1'] )


    performance_dict = vars(args)


    performance_dict['T_best_test_F'] = test_F
    performance_dict['T_best_test_ACC'] = test_acc
    performance_dict['T_best_test_R'] = test_R
    performance_dict['T_best_test_P'] = test_P
    performance_dict['T_best_test_CR'] = test_CR
    performance_dict['T_best_test_results'] = test_results
    performance_dict['T_best_test_results_by_tag'] = test_results_by_tag

    performance_dict['script_file'] = os.path.basename(__file__)
    performance_dict['note'] = 'elo'
    performance_dict['Time'] = str(datetime.datetime.now())
    # performance_dict['device'] = torch.cuda.get_device_name(device)
    for key, value in performance_dict.items():
        if type(value) is np.int64:
            performance_dict[key] = int(value)
    
    performance_file = 'performance/inferenze_EFRA_incidents.txt'
    with open(performance_file, 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')

if __name__ == '__main__':
    main()


    