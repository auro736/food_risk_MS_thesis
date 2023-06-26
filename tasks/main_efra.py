import random
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoConfig

from EMD.models import *


SEED = 42
ASSIGN_WEIGHT = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_local_model(model_path, config_path, device, model_name):

    config = config = AutoConfig.from_pretrained(config_path)
    if 'deberta' in model_name:
        print('deberta')
        model = ModelForTokenClassificationWithCRFDeberta(model_name=model_name,config=config)
    else:
        model = ModelForTokenClassificationWithCRF(model_name=model_name,config=config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    return model

def main():
    
    incidents = pd.read_pickle('/home/cc/rora_tesi_new/data/SampleAgroknow/incidents_annotated.pickle')
    # print(incidents.dtypes)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    model_name = 'microsoft/deberta-v3-large'

    model_path = '/home/cc/rora_tesi_new/log/log_EMD/deberta-v3-large/bertweet-token-crf/20_epoch/data/True_weight/42_seed/saved-model/pytorch_model.bin'
    config_path = '/home/cc/rora_tesi_new/log/log_EMD/deberta-v3-large/bertweet-token-crf/20_epoch/data/True_weight/42_seed/saved-model/config.json'

    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization = True)
    model = load_local_model(model_path, config_path, device, model_name)
    model = model.to(device)

if __name__ == '__main__':
    main()