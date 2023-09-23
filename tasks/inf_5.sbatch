#!/bin/sh
#SBATCH --job-name=inf_5
#SBATCH --output=/home/agensale/output_efra/inf_5.out
#SBATCH --nodes=1            
#SBATCH --ntasks=1     
#SBATCH --partition=cuda        
#SBATCH --cpus-per-task=1       
#SBATCH --gres=gpu:1   
#SBATCH --time=00:15:00          
#SBATCH --mail-type=ALL        
#SBATCH --mem=4GB

module load nvidia/cudasdk/11.6
#9.2


cd /home/agensale/rora_tesi_new

# source ~/.bashrc 

# activate the environment
source ~/miniconda3/bin/activate rora_venv


# check if the program is running on the right environment
which pip && which python


python tasks/inference_on_EFRA_incidents.py \
    --bert_model Jean-Baptiste/roberta-large-ner-english\
    --saved_model_path /home/agensale/rora_tesi/log_rora_tesi/log-token-classification/roberta-large-ner-english/bertweet-token-crf/entity_detection/24_epoch/data/True_weight/42_seed/saved-model/ \
    --from_finetuned \
    --data /home/agensale/rora_tesi_new/data/SampleAgroknow \
    --assign_weight \