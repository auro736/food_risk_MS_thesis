import argparse

def my_parser():

    parser = argparse.ArgumentParser()

    # mettere path fino all'ultimo / non mettere nome file .bin e .json 
    parser.add_argument("--saved_model_path", default=None, type=str)

    parser.add_argument("--from_finetuned", default = False, action = 'store_true')

    parser.add_argument("--bert_model", default=None, type=str)

    parser.add_argument("--model_type", default='bertweet-token-crf', type=str)
    
    parser.add_argument("--task_type", default='entity_detection', type=str)

    parser.add_argument('--n_epochs', default=None, type=int)

    parser.add_argument('--max_length', default=128, type=int)

    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--eval_batch_size', default=32, type=int)

    parser.add_argument('--test_batch_size', default=32, type=int)

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--learning_rate', default=1e-6, type=float)

    parser.add_argument('--data', default='/home/agensale/rora_tesi_new/data/SampleAgroknow', type=str)

    parser.add_argument('--log_dir', default='log/log_EFRA', type=str)

    parser.add_argument("--save_model", default=False, action='store_true')

    parser.add_argument("--early_stop", default=False, action='store_true')

    parser.add_argument("--assign_weight", default=False, action='store_true')

    parser.add_argument("--label_map", default='label_map_efra.json', type=str)

    return parser.parse_args()