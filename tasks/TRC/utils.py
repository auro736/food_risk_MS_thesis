import torch
import torch.nn as nn
from torch.autograd import Variable

import datetime
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay

import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from common_utils import pad_sequences
from TRC.custom_parser import my_parser
from TRC.models import ModelForWeightedSequenceClassification, ModelForWeightedSequenceClassificationDeberta

args = my_parser()

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
    # bert_tokens = ["[CLS]"]
    bert_tokens = [tokenizer.cls_token]
    for x in orig_tokens:
        bert_tokens.extend(tokenizer.tokenize(x))
    # bert_tokens.append("[SEP]")
    bert_tokens.append(tokenizer.sep_token)
    return bert_tokens


def tokenize_with_new_mask(orig_text, max_length, tokenizer):
    """
    tokenize a array of raw text and generate corresponding
    attention labels array and attention masks array
    """
    bert_tokens = [simple_tokenize(t, tokenizer) for t in orig_text]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
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

def calibration_plot(logits, y):
    # logits = output dell'ultimo layer del classificatore senza softmax
    # y = batch ground truth
    
    m = nn.Softmax(dim=1)
    probabilities = m(logits)

    # per creare calibration plot from_predictions la doc dice che vuole (y_true, prob)
    # dove prob sono le prob della classe positiva, ossia in probabilities
    # gli el sono liste [P(0), P(1)] devo prendermi solo P(1)

    pos_probs = probabilities[:,1] 
    # pos_probs = []
    # for probs in probabilities:
    #     pos_probs.append(probs[1])
    
    log_directory = args.log_dir + '/' + str(args.bert_model).split('/')[-1] + '/' + args.model_type + '/' \
                    + str(args.n_epochs) + '_epoch/' + args.data.split('/')[-1] + '/' + \
                    str(args.assign_weight) + '_weight/' + str(args.seed) + '_seed/'
    figure_name = 'calibration_curve.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.',
                                                                                                         '-')+ '.png'
    path = log_directory+figure_name
    disp = CalibrationDisplay.from_predictions(y, pos_probs, pos_label=1)
    plt.savefig(path)

def evaluate(model, test_batch_generator, num_batches, device, class_weight, split):
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
            if split.lower() == 'test':
                calibration_plot(logits=logits.detach().cpu(), y=y_batch.detach().cpu())

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

    return epoch_loss / num_batches, epoch_auc / num_batches, epoch_acc / num_batches, epoch_tn, epoch_fp, epoch_fn, epoch_tp, epoch_precision, epoch_recall, output_s_pred


def load_model(model_type, model_path, config):
    """
        model_type = from args, bertweet-seq or BiLSTM
        model_path = from args bert-model, the name of huggingface of the model used
    """
    if model_type == 'bertweet-seq':
        
        if "deberta" in model_path:
            model = ModelForWeightedSequenceClassificationDeberta(model_name = model_path, config = config)
        else:
            model = ModelForWeightedSequenceClassification(model_name = model_path, config = config)

    else:
        model = None
    return model




