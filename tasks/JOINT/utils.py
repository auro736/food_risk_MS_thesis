import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.autograd import Variable

from common_utils import pad_sequences, compute_crf_metrics, compute_metrics

from JOINT.models import ModelForJointClassificationWithCRF, ModelForJointClassificationWithCRFDeberta, ModelForJointClassification, ModelForJointClassificationDeberta


def eval_metrics(preds, y):
    '''
    Returns performance metrics of predictor
    :param y: ground truth label
    :param preds: predicted logits
    :return: auc, acc, tn, fp, fn, tp
    '''
    m = nn.Softmax(dim=1)
    probabilities = m(preds)
    y_values, indices = torch.max(probabilities, 1)
    y_pred = indices
    try:
        auc = roc_auc_score(y, y_values)
    except ValueError:
        auc = np.array(0)
    acc = accuracy_score(y, y_pred)
    conf_mat = confusion_matrix(y, y_pred, labels=[0, 1])
    tn = conf_mat[0, 0]
    fp = conf_mat[0, 1]
    fn = conf_mat[1, 0]
    tp = conf_mat[1, 1]
    return auc, acc, tn, fp, fn, tp


def simple_tokenize(orig_tokens, tokenizer, orig_labels, label_map, max_seq_length):
    """
    tokenize a array of raw text
    """
    # orig_tokens = orig_tokens.split()

    pad_token_label_id = -100
    tokens = []
    label_ids = []
    for word, label in zip(orig_tokens, orig_labels):
        word_tokens = tokenizer.tokenize(word)

        # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
        if len(word_tokens) > 0:
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = tokenizer.num_special_tokens_to_add()
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]

    bert_tokens = [tokenizer.cls_token]
    # bert_tokens = ["[CLS]"]

    bert_tokens.extend(tokens)
    label_ids = [pad_token_label_id] + label_ids

    bert_tokens.append(tokenizer.sep_token)
    # bert_tokens.append("[SEP]")
    label_ids += [pad_token_label_id]

    return bert_tokens, label_ids


def tokenize_with_new_mask(orig_text, max_length, tokenizer, orig_labels, label_map):
    """
    tokenize a array of raw text and generate corresponding
    attention labels array and attention masks array
    """
    pad_token_label_id = -100
    simple_tokenize_results = [list(tt) for tt in zip(
        *[simple_tokenize(orig_text[i], tokenizer, orig_labels[i], label_map, max_length) for i in
          range(len(orig_text))])]
    bert_tokens, label_ids = simple_tokenize_results[0], simple_tokenize_results[1]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    label_ids = pad_sequences(label_ids, maxlen=max_length, dtype="long", truncating="post", padding="post",
                              value=pad_token_label_id)
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)
    return input_ids, attention_masks, label_ids


def train(model, optimizer, train_batch_generator, num_batches, device, args, label_map, token_weight, y_weight):
    """
    Main training routine
    """
    epoch_loss = 0
    epoch_s_acc, epoch_s_auc, epoch_t_acc = 0, 0, 0
    epoch_t_results, epoch_t_results_by_tag = {}, {}
    epoch_s_tn, epoch_s_fp, epoch_s_fn, epoch_s_tp = 0, 0, 0, 0
    epoch_t_CR = ""

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        x_batch, y_batch_l, t_batch_l, masks_batch = next(train_batch_generator)
        if len(x_batch.shape) == 3:
            x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
        else:
            x_batch = Variable(torch.LongTensor(x_batch)).to(device)
        y_batch_l = y_batch_l.astype(np.float64)
        y_batch_l = torch.LongTensor(y_batch_l)
        y_batch = Variable(y_batch_l).to(device)
        t_batch_l = t_batch_l.astype(np.float64)
        t_batch_l = torch.LongTensor(t_batch_l)
        t_batch = Variable(t_batch_l).to(device)
        masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
        token_weight = token_weight.to(device) if token_weight is not None else None
        y_weight = y_weight.to(device) if y_weight is not None else None
        optimizer.zero_grad()
        outputs = model(input_ids=x_batch, attention_mask=masks_batch,
                        seq_labels=y_batch, token_labels=t_batch,
                        token_class_weight=token_weight, seq_class_weight=y_weight,
                        token_lambda=args.token_lambda, )

        loss, token_logits, seq_logits = outputs[:3]

        loss.backward()
        optimizer.step()

        s_auc, s_acc, s_tn, s_fp, s_fn, s_tp = eval_metrics(seq_logits.detach().cpu(),
                                                            y_batch_l)

        if type(model) in [ModelForJointClassificationWithCRF,ModelForJointClassificationWithCRFDeberta]:
            t_batch_filtered = [t_batch_l[i][t_batch_l[i] >= 0].tolist() for i in range(t_batch_l.shape[0])]
            t_eval_metrics = compute_crf_metrics(outputs[3], t_batch_filtered, label_map)
        else:
            t_eval_metrics = compute_metrics(token_logits.detach().cpu(), t_batch_l, label_map)

        epoch_loss += loss.item()
        epoch_s_auc += s_auc
        epoch_s_acc += s_acc
        epoch_s_tn += s_tn
        epoch_s_fp += s_fp
        epoch_s_fn += s_fn
        epoch_s_tp += s_tp
        epoch_t_acc += t_eval_metrics["accuracy_score"]
        epoch_t_results.update(t_eval_metrics["results"])
        epoch_t_results_by_tag.update(t_eval_metrics["results_by_tag"])
        epoch_t_CR = t_eval_metrics["CR"]

    print(f'\tClassification Loss: {epoch_loss / num_batches:.3f}')
    return_s_tuple = (epoch_loss / num_batches, epoch_s_auc / num_batches, epoch_s_acc / num_batches,
                      epoch_s_tn, epoch_s_fp, epoch_s_fn,
                      epoch_s_tp)
    return_t_tuple = (
        epoch_t_acc / num_batches, epoch_t_results, epoch_t_results_by_tag, epoch_t_CR)

    return_tuple = (return_s_tuple, return_t_tuple)
    return return_tuple


def evaluate(model, test_batch_generator, num_batches, device, args, label_map, token_weight, y_weight):
    """
    Main evaluation routine
    """
    epoch_loss = 0
    epoch_s_acc, epoch_s_auc, epoch_t_acc = 0, 0, 0
    epoch_t_results, epoch_t_results_by_tag = {}, {}
    epoch_s_tn, epoch_s_fp, epoch_s_fn, epoch_s_tp = 0, 0, 0, 0
    epoch_t_CR = ""

    output_t_pred, output_s_pred = None, None

    model.eval()

    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, y_batch, t_batch, masks_batch = next(test_batch_generator)
            if len(x_batch.shape) == 3:
                x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
            else:
                x_batch = Variable(torch.LongTensor(x_batch)).to(device)
            y_batch = y_batch.astype(np.float64)
            y_batch = Variable(torch.LongTensor(y_batch)).to(device)
            t_batch = t_batch.astype(np.float64)
            t_batch = Variable(torch.LongTensor(t_batch)).to(device)
            masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
            token_weight = token_weight.to(device) if token_weight is not None else None
            y_weight = y_weight.to(device) if y_weight is not None else None

            outputs = model(input_ids=x_batch, attention_mask=masks_batch,
                            seq_labels=y_batch, token_labels=t_batch,
                            token_class_weight=token_weight, seq_class_weight=y_weight,
                            token_lambda=args.token_lambda, )

            loss, token_logits, seq_logits = outputs[:3]

            s_auc, s_acc, s_tn, s_fp, s_fn, s_tp = eval_metrics(seq_logits.detach().cpu(),
                                                                y_batch.detach().cpu())

            if type(model) in [ModelForJointClassificationWithCRF, ModelForJointClassificationWithCRFDeberta]:
                t_batch_l = t_batch.detach().cpu()
                t_batch_filtered = [t_batch_l[i][t_batch_l[i] >= 0].tolist() for i in range(t_batch_l.shape[0])]
                t_eval_metrics = compute_crf_metrics(outputs[3], t_batch_filtered, label_map)
            else:
                t_eval_metrics = compute_metrics(token_logits.detach().cpu(), t_batch.detach().cpu(), label_map)

            epoch_loss += loss.item()
            epoch_s_auc += s_auc
            epoch_s_acc += s_acc
            epoch_s_tn += s_tn
            epoch_s_fp += s_fp
            epoch_s_fn += s_fn
            epoch_s_tp += s_tp
            epoch_t_acc += t_eval_metrics["accuracy_score"]
            epoch_t_results.update(t_eval_metrics["results"])
            epoch_t_results_by_tag.update(t_eval_metrics["results_by_tag"])
            epoch_t_CR = t_eval_metrics["CR"]
            if output_t_pred is None:
                output_t_pred = token_logits.detach().cpu().numpy()
                output_s_pred = seq_logits.detach().cpu().numpy()
            else:
                output_t_pred = np.concatenate([output_t_pred, token_logits.detach().cpu().numpy()], axis=0)
                output_s_pred = np.concatenate([output_s_pred, seq_logits.detach().cpu().numpy()], axis=0)

        print(f'\tClassification Loss: {epoch_loss / num_batches:.3f}')
        return_s_tuple = (epoch_loss / num_batches, epoch_s_auc / num_batches, epoch_s_acc / num_batches,
                          epoch_s_tn, epoch_s_fp, epoch_s_fn,
                          epoch_s_tp)
        return_t_tuple = (
            epoch_t_acc / num_batches, epoch_t_results, epoch_t_results_by_tag, epoch_t_CR)
        return_tuple = (return_s_tuple, return_t_tuple, output_t_pred, output_s_pred)
        return return_tuple
    

def load_model(model_type, model_path, config):
    if model_type.startswith('bertweet-multi') and not model_type.startswith('bertweet-multi-crf'):

        if "deberta" in model_path:
            model = ModelForJointClassificationDeberta(model_name = model_path, config=config)
        else:
            model = ModelForJointClassification(model_name = model_path, config=config)

    elif model_type == 'bertweet-multi-crf':
        if "deberta" in model_path:
            model = ModelForJointClassificationWithCRFDeberta(model_name = model_path, config=config)
        else:
            model = ModelForJointClassificationWithCRF(model_name = model_path, config=config)

    else:
        model = None
    return model