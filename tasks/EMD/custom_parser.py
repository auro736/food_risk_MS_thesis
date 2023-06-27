import argparse

def my_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default= 'xlm-roberta-large-finetuned-conll03-english', type=str)

    parser.add_argument("--model_type", default='bertweet-token-crf', type=str)

    parser.add_argument("--task_type", default='entity_detection', type=str)

    parser.add_argument('--n_epochs', default=24, type=int)

    parser.add_argument('--max_length', default=128, type=int)

    parser.add_argument('--rnn_hidden_size', default=384, type=int)

    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('--eval_batch_size', default=300, type=int)

    parser.add_argument('--test_batch_size', default=300, type=int)

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--learning_rate', default=1e-6, type=float)

    parser.add_argument('--data', default='/home/cc/rora_tesi_new/data/Tweet-Fid', type=str)

    parser.add_argument('--log_dir', default='log/log_EMD', type=str)

    parser.add_argument("--save_model", default=True, action='store_true')

    parser.add_argument("--early_stop", default=False, action='store_true')

    parser.add_argument("--assign_weight", default=True, action='store_true')

    parser.add_argument("--train_file", default='train.p', type=str)

    parser.add_argument("--val_file", default='dev.p', type=str)

    parser.add_argument("--test_file", default='test.p', type=str)

    parser.add_argument("--label_map", default='label_map.json', type=str)

    parser.add_argument("--performance_file", default='performance/performance_EMD.txt', type=str)

    parser.add_argument("--embeddings_file", default='glove.840B.300d.txt', type=str)

    return parser.parse_args()