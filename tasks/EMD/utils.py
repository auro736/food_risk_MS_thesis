import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable

from common_utils import pad_sequences, compute_metrics, compute_crf_metrics
from EMD.models import ModelForTokenClassificationWithCRF, ModelForTokenClassificationWithCRFDeberta, ModelForWeightedTokenClassification, ModelForWeightedTokenClassificationDeberta

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


def train(model, optimizer, train_batch_generator, num_batches, device, args, label_map, class_weight):
    """
    Main training routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_results = {}
    epoch_results_by_tag = {}
    epoch_CR = ""

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

        outputs = model(input_ids=x_batch, attention_mask=masks_batch, labels=y_batch, class_weight=class_weight)

        loss, logits = outputs[:2]

        loss.backward()
        optimizer.step()

        if type(model) in [ModelForTokenClassificationWithCRF, ModelForTokenClassificationWithCRFDeberta]:
            y_batch = y_batch.detach().cpu()
            y_batch_filtered = [y_batch[i][y_batch[i] >= 0].tolist() for i in range(y_batch.shape[0])]
            eval_metrics = compute_crf_metrics(outputs[2], y_batch_filtered, label_map)
        else:
            eval_metrics = compute_metrics(logits.detach().cpu(), y_batch.detach().cpu(), label_map)

        #print('sono in train', eval_metrics)
        epoch_loss += loss.item()
        epoch_acc += eval_metrics["accuracy_score"]
        epoch_results.update(eval_metrics["results"])
        epoch_results_by_tag.update(eval_metrics["results_by_tag"])
        epoch_CR = eval_metrics["CR"]
    print(f'\tClassification Loss: {epoch_loss / num_batches:.3f}')
    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_results, \
           epoch_results_by_tag, epoch_CR


def evaluate(model, test_batch_generator, num_batches, device, label_map, class_weight):
    """
    Main evaluation routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_results = {}
    epoch_results_by_tag = {}
    epoch_CR = ""

    output_t_pred = None

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
            outputs = model(input_ids=x_batch, attention_mask=masks_batch, labels=y_batch,
                            class_weight=class_weight)

            loss, logits = outputs[:2]

            if type(model) in [ModelForTokenClassificationWithCRF,ModelForTokenClassificationWithCRFDeberta]:    
                y_batch = y_batch.detach().cpu()
                y_batch_filtered = [y_batch[i][y_batch[i] >= 0].tolist() for i in range(y_batch.shape[0])]
                eval_metrics = compute_crf_metrics(outputs[2], y_batch_filtered, label_map)
            else:
                eval_metrics = compute_metrics(logits.detach().cpu(), y_batch.detach().cpu(), label_map)

            #print(eval_metrics)
            epoch_loss += loss.item()
            epoch_acc += eval_metrics["accuracy_score"]
            epoch_results.update(eval_metrics["results"])
            epoch_results_by_tag.update(eval_metrics["results_by_tag"])
            epoch_CR = eval_metrics["CR"]
            if output_t_pred is None:
                output_t_pred = logits.detach().cpu().numpy()
            else:
                output_t_pred = np.concatenate([output_t_pred, logits.detach().cpu().numpy()], axis=0)

    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_results, \
           epoch_results_by_tag, output_t_pred, epoch_CR


def predict(model, test_batch_generator, num_batches, device, label_map, class_weight):
    """
    Main evaluation routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_results = {}
    epoch_results_by_tag = {}
    epoch_CR = ""

    output_t_pred = None

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

            outputs = model(input_ids=x_batch, attention_mask=masks_batch, labels=y_batch,
                            class_weight=class_weight)

            loss, logits = outputs[:2]
            # logits ha shape (412, 128, 9)
            # 412 lunghezza test ds
            # 128 lunghezza max della sequenza
            # 9 numero classi
            
            # print(logits[0][0])
            # print(logits[0][1])

            if type(model) in [ModelForTokenClassificationWithCRF, ModelForTokenClassificationWithCRFDeberta]:   
                # ha shape (412, 128)
                y_batch = y_batch.detach().cpu()
                # diventa una lista di liste di lunghezza 412
                # ogni elemento delle liste interne rappresentano la classe associata ai token del tweet
                y_batch_filtered = [y_batch[i][y_batch[i] >= 0].tolist() for i in range(y_batch.shape[0])]
                # outputs[2] è una lista di lista con le prediction
                eval_metrics = compute_crf_metrics(outputs[2], y_batch_filtered, label_map)
            else:
                eval_metrics = compute_metrics(logits.detach().cpu(), y_batch.detach().cpu(), label_map)

            epoch_loss += loss.item()
            epoch_acc += eval_metrics["accuracy_score"]
            epoch_results.update(eval_metrics["results"])
            epoch_results_by_tag.update(eval_metrics["results_by_tag"])
            epoch_CR = eval_metrics["CR"]
            if output_t_pred is None:
                output_t_pred = logits.detach().cpu().numpy()
            else:
                output_t_pred = np.concatenate([output_t_pred, logits.detach().cpu().numpy()], axis=0)

    return logits.detach().cpu(), epoch_loss / num_batches, epoch_acc / num_batches, epoch_results, \
           epoch_results_by_tag, output_t_pred, epoch_CR


def load_model(model_type, model_path, config):
    if model_type == 'bertweet-token':

        if "deberta" in model_path :
            model = ModelForWeightedTokenClassificationDeberta(model_name = model_path, config = config)
        else:
            model = ModelForWeightedTokenClassification(model_name = model_path, config = config)
            
    elif model_type == 'bertweet-token-crf':
        
        if "deberta" in model_path :
            model = ModelForTokenClassificationWithCRFDeberta(model_name = model_path, config=config)
        else:
            model = ModelForTokenClassificationWithCRF(model_name = model_path, config=config)

    else:
        model = None
    return model