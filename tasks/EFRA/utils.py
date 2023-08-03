import six
import numpy as np

def simple_tokenize(orig_tokens, tokenizer):
    """
    tokenize a array of raw text
    """
    bert_tokens = [tokenizer.cls_token]
    for x in orig_tokens:
        bert_tokens.extend(tokenizer.tokenize(x))
    bert_tokens.append(tokenizer.sep_token)
    return bert_tokens

def tokenize_with_new_mask_efra(orig_text, max_length, tokenizer):
    """
    tokenize a array of raw text and generate corresponding
    attention labels array and attention masks array
    """
    # tokenizza semplice
    # lunga quanto # di righe di input di dataset

    print('A', len(orig_text))
  
    bert_tokens = []
    for t in orig_text:
        # t lista di sentences
        tmp = []
        for s in t:
            if len(s) > 1:
                tmp.append(simple_tokenize(s, tokenizer))
        bert_tokens.append(tmp)

    print(len(bert_tokens))

 
    # bert_tokens = [simple_tokenize(t, tokenizer) for t in orig_text]
   
    
    # embedding
    # lunga quanto # di righe di input di dataset

    input_ids = []
    for t in bert_tokens:
        tmp = []
        for s in t:
            tmp.append(tokenizer.convert_tokens_to_ids(s))
        input_ids.append(tmp)
    
    print('IDS',len(input_ids))
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
    print(len(input_ids))
    # print(input_ids[0].shape)
    # print(input_ids)

    # creo attention masks, metto 1 dove embeddings in input_ids > 0
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)
    
    return input_ids, attention_masks

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x