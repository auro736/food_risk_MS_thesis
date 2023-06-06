import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay

from TRC.utils import tokenize_with_new_mask, evaluate
from common_utils import extract_from_dataframe, mask_batch_seq_generator
from TRC.models import ModelForWeightedSequenceClassification

# TRC 

MAX_LENGTH = 128
ASSIGN_WEIGHT = True
SEED = 42


def load_local_model(model_path, config_path, device, model_name):

    config = config = AutoConfig.from_pretrained(config_path)

    model = ModelForWeightedSequenceClassification(model_name=model_name,config=config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    return model

def prepare_data(data_path, need_columns):

    data = pd.read_pickle(data_path)
    list = extract_from_dataframe(data, need_columns)
    return list

def create_weight(Y_train):
    # weight of each class in loss function
    class_weight = [np.array(Y_train).shape[0] / (np.array(Y_train) == i).sum() for i in range(len(set(Y_train)))]
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


def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model_name = 'cardiffnlp/twitter-roberta-large-2022-154m'
    log_directory = '/home/cc/rora_tesi_new/log/log_TRC/twitter-roberta-large-2022-154m/bertweet-seq/20_epoch/data/True_weight/42_seed/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = '/home/cc/rora_tesi_new/log/log_TRC/twitter-roberta-large-2022-154m/bertweet-seq/20_epoch/data/True_weight/42_seed/saved-model/pytorch_model.bin'
    config_path = '/home/cc/rora_tesi_new/log/log_TRC/twitter-roberta-large-2022-154m/bertweet-seq/20_epoch/data/True_weight/42_seed/saved-model/config.json'

    model = load_local_model(model_path, config_path, device, model_name)
    
    model = model.to(device)

    train_data_path = '/home/cc/rora_tesi_new/data/train.p'
    test_data_path = '/home/cc/rora_tesi_new/data/test.p'
    
    need_columns = ['tweet_tokens', 'sentence_class']

    _, Y_train = prepare_data(train_data_path, need_columns)
    X_test_raw, Y_test = prepare_data(test_data_path, need_columns)

    test_batch_size = Y_test.shape[0]

    class_weight = None
    if ASSIGN_WEIGHT:
        class_weight = create_weight(Y_train)
    
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

    softmax = nn.Softmax(dim=1)
    probabilities = softmax(logits)

    cal_path = log_directory + 'calibration_curve.png'
    calibration_plot(probabilities, y_true = y_true, img_path=cal_path, model_name=model_name)

    conf_path = log_directory + 'confusion_matrix.png'
    confusion_matrix(probabilities, y_true=y_true, model_name =model_name, path = conf_path)


if __name__ == '__main__':
    main()
