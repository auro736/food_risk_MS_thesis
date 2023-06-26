import argparse

# CUSTOM PARSER FOR TRC TASK

def my_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default='roberta-large', type=str)

    parser.add_argument("--model_type", default='bertweet-seq', type=str)

    parser.add_argument('--n_epochs', default=2, type=int)

    parser.add_argument('--max_length', default=128, type=int)

    # non lo usa se bert-model is not None
    parser.add_argument('--rnn_hidden_size', default=384, type=int)

    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--eval_batch_size', default=300, type=int)

    parser.add_argument('--test_batch_size', default=300, type=int)

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--learning_rate', default=1e-6, type=float)

    parser.add_argument('--data', default='/home/cc/rora_tesi_new/data/Tweet-Fid', type=str)

    parser.add_argument('--log_dir', default='log/log_TRC', type=str)

    parser.add_argument("--save_model", default=False, action='store_true')

    parser.add_argument("--early_stop", default=False, action='store_true')

    parser.add_argument("--assign_weight", default=True, action='store_true')

    parser.add_argument("--train_file", default='train.p', type=str)

    parser.add_argument("--val_file", default='dev.p', type=str)

    parser.add_argument("--test_file", default='test.p', type=str)

    parser.add_argument("--performance_file", default='performance/performance_TRC.txt', type=str)

    parser.add_argument("--embeddings_file", default='glove.840B.300d.txt', type=str)

    return parser.parse_args()