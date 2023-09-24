import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers import AutoConfig

import nltk
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from better_profanity import profanity
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from common_utils import pad_sequences, extract_from_dataframe
from TRC.models import ModelForWeightedSequenceClassification, ModelForWeightedSequenceClassificationDeberta


def load_local_TRC_model(model_path, config_path, device, model_name):

    config = AutoConfig.from_pretrained(config_path)
    
    if 'deberta' in model_name.lower():
        model = ModelForWeightedSequenceClassificationDeberta(model_name=model_name, config=config)
    else:
    # print(model_name)
        model = ModelForWeightedSequenceClassification(model_name=model_name,config=config)
    # print(model)
    checkpoint = torch.load(model_path, map_location=device)
    # print(heckpoint)
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

def confusion_matrix_display(probabilities, y_true, model_name, path):

    y_values, indices = torch.max(probabilities, 1)
    # dato che la posizione nella lista di 2 elementi della probabilità 
    # rappresenta la prob della label
    y_pred = indices
    conf_mat = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], colorbar=False)
    plt.title(model_name)
    plt.savefig(path)


def create_token_dict(shap_values, ind_to_get):
    print('indice di shap preso: ', ind_to_get)
    token_dict = {}
    nltk.download('stopwords')
    profanity.load_censor_words()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    val = [shap_values[i] for i in range(len(shap_values))]
    for el in val:
        for i in range(len(el)-1):
            token = el[i].data
            # print(token)
            token = token.strip()
            if token not in string.punctuation and token != 'USER' and token != 'HTTPURL' and token != 'HTTP' and token!='URL' and token != '...' and len(token) >= 3:
                # print(token)
                shap_val = el[i].values
                pos = abs(shap_val[ind_to_get])
                token = token.lower()
                if token not in stop_words:
                    if not profanity.contains_profanity(token):
                        if token not in token_dict.keys():
                            token_dict[token] = pos
                        else:
                            token_dict[token] += pos
    return token_dict

def get_top_n(n,token_dict):
    tmp = {}
    for i,k,v in zip(range(n), token_dict.keys(), token_dict.values()):
        tmp[k] = v
    return tmp

def create_analysis_csv(probabilities, tweet_test,tweet_token, y_true):

    y_values, indices = torch.max(probabilities, 1)
    # dato che la posizione nella lista di 2 elementi della probabilità 
    # rappresenta la prob della label
    y_pred = indices.detach().cpu().numpy()

    # error is a list of list [indice dell'errore,tweet_errato, y_true, y_pred]
    print(tweet_token[0])
    print(type(tweet_token[0]))

    errors = [[i, tweet_test[i], tweet_token[i], y_true[i], y_pred[i]] for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    errors = np.array(errors)

    corr = [[i, tweet_test[i], tweet_token[i], y_true[i], y_pred[i]] for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    corr = np.array(corr)

    cols = ['Tweet', 'Tweet tok', 'True label', 'Pred label']

    errors_df = pd.DataFrame(errors[:,1:], columns=cols)
    corr_df = pd.DataFrame(corr[:,1:], columns=cols)
    
    return errors_df, corr_df

def eval_metrics(preds, y):

    '''
    Returns performance metrics of predictor
    :param y: ground truth label
    :param preds: predicted logits
    :return: auc, acc, tn, fp, fn, tp
    '''
    m = nn.Softmax(dim=1)
    probabilities = m(preds)
    # torna il valore più alto e il rispettivo indice 
    # y_values la probabilità maggiore per ogni tweet
    y_values, indices = torch.max(probabilities, 1)
    # dato che la posizione nella lista di 2 elementi della probabilità 
    # rappresenta la prob della label
    y_pred = indices

    try:
        auc = roc_auc_score(y, y_values)
    except:
        auc = np.array(0)
    acc = accuracy_score(y, y_pred)
    conf_mat = confusion_matrix(y, y_pred, labels=[0, 1])
    tn = conf_mat[0, 0]
    fp = conf_mat[0, 1]
    fn = conf_mat[1, 0]
    tp = conf_mat[1, 1]
    return auc, acc, tn, fp, fn, tp


def simple_tokenize(orig_tokens, tokenizer):
    """
    tokenize a array of raw text
    """
    bert_tokens = [tokenizer.cls_token]
    for x in orig_tokens:
        bert_tokens.extend(tokenizer.tokenize(x))
    bert_tokens.append(tokenizer.sep_token)
    return bert_tokens


def tokenize_with_new_mask(orig_text, max_length, tokenizer):
    """
    tokenize a array of raw text and generate corresponding
    attention labels array and attention masks array
    """
    # tokenizza semplice
    bert_tokens = [simple_tokenize(t, tokenizer) for t in orig_text]
    
    # embedding
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    
    # padding delle sequenze in modo da averle tutte della stessa lunghezza
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")

    # creo attention masks, metto 1 dove embeddings in input_ids > 0
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)
    
    return input_ids, attention_masks


def train(model, optimizer, train_batch_generator, num_batches, device, class_weight):
    """
    Main training routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_auc = 0
    epoch_cl = 0
    epoch_al = 0
    epoch_tn, epoch_fp, epoch_fn, epoch_tp, epoch_precision, epoch_recall = 0, 0, 0, 0, 0, 0

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        x_batch, y_batch, masks_batch = next(train_batch_generator)
        if len(x_batch.shape) == 3:
            x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
        else:
            x_batch = Variable(torch.LongTensor(x_batch)).to(device)
        y_batch = y_batch.astype(np.float64)
        y_batch = Variable(torch.LongTensor(y_batch)).to(device)
        masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
        class_weight = class_weight.to(device) if class_weight is not None else None
        optimizer.zero_grad()
        outputs = model(x_batch, masks_batch, labels=y_batch, class_weight=class_weight)
        
        loss, logits = outputs[:2]

        loss.backward()
        optimizer.step()

        auc, acc, tn, fp, fn, tp = eval_metrics(logits.detach().cpu(),
                                                y_batch.detach().cpu())

        epoch_loss += loss.item()
        epoch_auc += auc.item()
        epoch_acc += acc.item()
        epoch_tn += tn.item()
        epoch_fp += fp.item()
        epoch_fn += fn.item()
        epoch_tp += tp.item()

    if (epoch_tp + epoch_fp) != 0:
        epoch_precision = epoch_tp / (epoch_tp + epoch_fp)
    if  (epoch_tp + epoch_fn) != 0:
        epoch_recall = epoch_tp / (epoch_tp + epoch_fn)

    print(f'\tClassification Loss: {epoch_loss / num_batches:.3f}')
    return epoch_loss / num_batches, epoch_auc / num_batches, epoch_acc / num_batches, epoch_tn, epoch_fp, epoch_fn, epoch_tp, epoch_precision, epoch_recall


def evaluate(model, test_batch_generator, num_batches, device, class_weight):
    """
    Main evaluation routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_auc = 0
    epoch_tn, epoch_fp, epoch_fn, epoch_tp, epoch_precision, epoch_recall = 0, 0, 0, 0, 0, 0

    output_s_pred = None

    model.eval()

    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, y_batch, masks_batch = next(test_batch_generator)
            if len(x_batch.shape) == 3:
                x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
            else:
                x_batch = Variable(torch.LongTensor(x_batch)).to(device)
            y_batch = y_batch.astype(np.float64)
            y_batch = Variable(torch.LongTensor(y_batch)).to(device)
            masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
            class_weight = class_weight.to(device) if class_weight is not None else None
            outputs = model(x_batch, masks_batch, labels=y_batch, class_weight=class_weight)

            loss, logits = outputs[:2]

            

            #logits escono da rete
            #se gli applico softmax ottengo probabilità
            auc, acc, tn, fp, fn, tp  = eval_metrics(logits.detach().cpu(),
                                                    y_batch.detach().cpu()) #y_true
            
            epoch_loss += loss.item()
            epoch_auc += auc.item()
            epoch_acc += acc.item()
            epoch_tn += tn.item()
            epoch_fp += fp.item()
            epoch_fn += fn.item()
            epoch_tp += tp.item()

            if output_s_pred is None:
                output_s_pred = logits.detach().cpu().numpy()
            else:
                output_s_pred = np.concatenate([output_s_pred, logits.detach().cpu().numpy()], axis=0)

        if (epoch_tp + epoch_fp) != 0:
            epoch_precision = epoch_tp / (epoch_tp + epoch_fp)
        if (epoch_tp + epoch_fn) != 0:
            epoch_recall = epoch_tp / (epoch_tp + epoch_fn)

    return logits.detach().cpu(), y_batch.detach().cpu().numpy(), epoch_loss / num_batches, epoch_auc / num_batches, epoch_acc / num_batches, epoch_tn, epoch_fp, epoch_fn, epoch_tp, epoch_precision, epoch_recall, output_s_pred


def load_model(model_path, config):
    """
        
        model_path = from args bert-model, the name of huggingface of the model used
    """
        
    if "deberta" in model_path:
        model = ModelForWeightedSequenceClassificationDeberta(model_name = model_path, config = config)
    else:
        model = ModelForWeightedSequenceClassification(model_name = model_path, config = config)


    return model




