import torch
import torch.nn as nn
from transformers import AutoConfig

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay

from common_utils import extract_from_dataframe
from TRC.models import ModelForWeightedSequenceClassification

def load_local_model(model_path, config_path, device, model_name):

    config = config = AutoConfig.from_pretrained(config_path)

    model = ModelForWeightedSequenceClassification(model_name=model_name,config=config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    return model

def create_weight(Y_train):
    # weight of each class in loss function
    class_weight = [np.array(Y_train).shape[0] / (np.array(Y_train) == i).sum() for i in range(len(set(Y_train)))]
    class_weight = torch.FloatTensor(class_weight)
    return class_weight

def prepare_data(data_path, need_columns):

    data = pd.read_pickle(data_path)
    list = extract_from_dataframe(data, need_columns)
    return list

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

def tweet_errati(probabilities, tweet_test, y_true):

    y_values, indices = torch.max(probabilities, 1)
    # dato che la posizione nella lista di 2 elementi della probabilità 
    # rappresenta la prob della label
    y_pred = indices.detach().cpu().numpy()

    # error is a list of list [indice dell'errore,tweet_errato, y_true, y_pred]
    error = [[i, tweet_test[i], y_true[i], y_pred[i]] for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    error = np.array(error)
    to_save = error[:,1:]

    cols = ['Tweet', 'True label', 'Pred label']
    df = pd.DataFrame(to_save, columns=cols)
    return df 