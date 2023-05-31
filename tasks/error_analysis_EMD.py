import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay

from EMD.models import ModelForTokenClassificationWithCRF, ModelForWeightedTokenClassification
from common_utils import extract_from_dataframe, mask_batch_seq_generator
from EMD.utils import tokenize_with_new_mask, predict

# EMD 

MAX_LENGTH = 128
ASSIGN_WEIGHT = True
SEED = 42

def load_local_model(model_path, config_path, device, model_name):

    config = config = AutoConfig.from_pretrained(config_path)

    model = ModelForTokenClassificationWithCRF(model_name=model_name,config=config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    return model

def prepare_data(data_path, need_columns):

    data = pd.read_pickle(data_path)
    X, Y, seq = extract_from_dataframe(data, need_columns)
    return X, Y, seq

def create_weight(Y_train, labels):
    # weight of each class in loss function
    class_weight = [Y_train.shape[0] / (Y_train == i).sum() for i in range(len(labels))]
    class_weight = torch.FloatTensor(class_weight)
    return class_weight

def calibration_plot(probabilities, y_true, img_path, model_name):
    # probabilities = probabilità per classe
    # y = batch ground truth

    # per creare calibration plot from_predictions la doc dice che vuole (y_true, prob)
    # dove prob sono le prob della classe positiva, ossia in probabilities
    # gli el sono liste [P(0), P(1)] devo prendermi solo P(1)

    pos_probs = probabilities[:,1] 
    
    disp = CalibrationDisplay.from_predictions(y_true, pos_probs, pos_label=1)
    plt.title(model_name)
    plt.savefig(img_path)

def confusion_matrix(probabilities, y_true, model_name, path):

    y_values, indices = torch.max(probabilities, 1)
    # dato che la posizione nella lista di 2 elementi della probabilità 
    # rappresenta la prob della label
    y_pred = indices
    conf_mat = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], colorbar=False)
    plt.title(model_name)
    plt.savefig(path)

def to_binary(probabilities):
    lista = []
  
       


def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model_name = 'roberta-large'
    log_directory = '/home/cc/rora_tesi_new/log/log_EMD/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = '/home/cc/rora_tesi/log_rora_tesi/log-token-classification/roberta-large/bertweet-token-crf/entity_detection/20_epoch/data/True_weight/42_seed/saved-model/pytorch_model.bin'
    config_path = '/home/cc/rora_tesi/log_rora_tesi/log-token-classification/roberta-large/bertweet-token-crf/entity_detection/20_epoch/data/True_weight/42_seed/saved-model/config.json'

    model = load_local_model(model_path, config_path, device, model_name)
    
    model = model.to(device)

    train_data_path = '/home/cc/rora_tesi_new/data/train.p'
    test_data_path = '/home/cc/rora_tesi_new/data/test.p'

    label_path = '/home/cc/rora_tesi_new/data/label_map.json'
    
    need_columns = ['tweet_tokens', 'entity_label','sentence_class']

    X_train_raw, Y_train_raw, seq_train = prepare_data(train_data_path, need_columns)
    X_test_raw, Y_test_raw, seq_test = prepare_data(test_data_path, need_columns)

    test_batch_size = seq_test.shape[0]

    with open(label_path, 'r') as fp:
        label_map = json.load(fp)

    labels = list(label_map.keys())

    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization = True)

    X_train, masks_train, Y_train = tokenize_with_new_mask(X_train_raw, MAX_LENGTH, tokenizer, Y_train_raw, label_map)
    X_test, masks_test, Y_test = tokenize_with_new_mask(X_test_raw, 128, tokenizer, Y_test_raw, label_map)

    class_weight = None
    if ASSIGN_WEIGHT:
        class_weight = create_weight(Y_train, labels)

    num_batches = X_test.shape[0] // test_batch_size
    test_batch_generator = mask_batch_seq_generator(X_test, Y_test, masks_test, test_batch_size)

    logits, y_true, test_loss, test_acc, test_results, test_results_by_tag, test_t_pred, test_CR = predict(model,
                                                                                           test_batch_generator,
                                                                                           num_batches, device,
                                                                                           label_map, class_weight)
    test_F = test_results['strict']['f1']
    test_P = test_results['strict']['precision']
    test_R = test_results['strict']['recall']
    print(f'Test Acc: {test_acc * 100:.2f}%')
    print(f'Test P Strict: {test_P * 100:.2f}%')
    print(f'Test R Strict: {test_R * 100:.2f}%')
    print(f'Test F1 Strict: {test_F * 100:.2f}%')

    print(len(logits))
    print(logits.shape)

    softmax = nn.Softmax(dim=1)
    probabilities = softmax(logits)

    to_binary(probabilities)

    # print(probabilities[0][0])
    # print(len(probabilities[0][0]))
    print(len(y_true))

    # print(probabilities[0], y_true[0])

    # cal_path = log_directory + 'calibration_curve.png'

    
    # # rendere il problema binario per calibration curve
    # # 1 classe vs le altre, prendo prob della classe considerata e calcolo prob delle altre classi come 1-probclasse
    # calibration_plot(y_true=y_true, probabilities=probabilities, img_path=cal_path, model_name=model_name)

if __name__ == '__main__':
    main()