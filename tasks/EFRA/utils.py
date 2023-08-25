import six
import numpy as np
from common_utils import pad_sequences
import random

random.seed(42)

def simple_tokenize_news(orig_tokens, tokenizer):
    """
    tokenize a array of raw text
    """
    bert_tokens = [tokenizer.cls_token]
    for x in orig_tokens:
        bert_tokens.extend(tokenizer.tokenize(x))
    bert_tokens.append(tokenizer.sep_token)
    return bert_tokens

def simple_tokenize_inc(orig_tokens, tokenizer, orig_labels, label_map, max_seq_length):
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

def tokenize_with_new_mask_inc_train(orig_text, max_length, tokenizer, orig_labels, label_map):

    pad_token_label_id = -100

    bert_tokens = []
    label_ids = []
    for t, labels in zip(orig_text, orig_labels):
        # t lista di sentences
        tmp = []
        for s, l in zip(t, labels):
            if len(s) > 3:
                if not all(elemento == "O" for elemento in l):
                    # print(s)
                    # print(l)
                    bert_tokens.append(simple_tokenize_inc(s, tokenizer, l, label_map, max_length)[0])
                    label_ids.append(simple_tokenize_inc(s, tokenizer, l, label_map, max_length)[1])
                else: # tutti i token senza label ner
                    # print(s)
                    # print(l)
                    rnd = random.uniform(0,1)
                    if rnd <= 0.3:
                        bert_tokens.append(simple_tokenize_inc(s, tokenizer, l, label_map, max_length)[0])
                        label_ids.append(simple_tokenize_inc(s, tokenizer, l, label_map, max_length)[1])
    
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    
    # print(len(input_ids))
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    label_ids = pad_sequences(label_ids, maxlen=max_length, dtype="long", truncating="post", padding="post",
                              value=pad_token_label_id)
    # print(len(input_ids))
    # print(len(input_ids[0]))
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)

    return input_ids, attention_masks, label_ids

def tokenize_with_new_mask_inc(orig_text, max_length, tokenizer, orig_labels, label_map):

    pad_token_label_id = -100

    bert_tokens = []
    label_ids = []
    for t, labels in zip(orig_text, orig_labels):
        # t lista di sentences
        tmp = []
        for s, l in zip(t, labels):
            if len(s) > 3:
                bert_tokens.append(simple_tokenize_inc(s, tokenizer, l, label_map, max_length)[0])
                label_ids.append(simple_tokenize_inc(s, tokenizer, l, label_map, max_length)[1])

    # print(bert_tokens)
    # print('AAAAAAAA',  label_ids)
    # print(len(bert_tokens))
    # print(len(label_ids))
    
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    
    # print(len(input_ids))
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    label_ids = pad_sequences(label_ids, maxlen=max_length, dtype="long", truncating="post", padding="post",
                              value=pad_token_label_id)
    # print(len(input_ids))
    # print(len(input_ids[0]))
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)
    return input_ids, attention_masks, label_ids

def tokenize_with_new_mask_news(orig_text, max_length, tokenizer):
    """
    tokenize a array of raw text and generate corresponding
    attention labels array and attention masks array
    """
    # tokenizza semplice
    # lunga quanto # di righe di input di dataset

    # print('A', len(orig_text))
  
    bert_tokens = []
    for t in orig_text:
        # t lista di sentences
        tmp = []
        for s in t:
            if len(s) > 1:
                tmp.append(simple_tokenize_news(s, tokenizer))
        bert_tokens.append(tmp)

    # print(len(bert_tokens))

 
    # bert_tokens = [simple_tokenize(t, tokenizer) for t in orig_text]
   
    
    # embedding
    # lunga quanto # di righe di input di dataset

    input_ids = []
    for t in bert_tokens:
        tmp = []
        for s in t:
            tmp.append(tokenizer.convert_tokens_to_ids(s))
        input_ids.append(tmp)
    
    # print('IDS',len(input_ids))
    # for i in range(len(input_ids)):
    #     print(input_ids[i])

    text_encoded = []
    for t in input_ids:
        tmp = []
        for s in t:
            mean = int(sum(s) / len(s))
            tmp.append(mean)
        text_encoded.append(tmp)
  
    
    # input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    # print('BBBB', len(input_ids))
    
    # padding delle sequenze in modo da averle tutte della stessa lunghezza
    # input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    
    input_ids = pad_sequences(text_encoded,maxlen=max_length,dtype="long", truncating="post", padding="post")
    # print(len(input_ids))
    # print(input_ids[0].shape)
    # print(input_ids)

    # creo attention masks, metto 1 dove embeddings in input_ids > 0
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)
    
    return input_ids, attention_masks