#!/bin/sh
#SBATCH --job-name=emd_roberta_base
#SBATCH --output=/home/agensale/output_efra/emd_roberta_base.out
#SBATCH --nodes=1            
#SBATCH --ntasks=1     
#SBATCH --partition=cuda        
#SBATCH --cpus-per-task=1       
#SBATCH --gres=gpu:1   
#SBATCH --time=01:00:00          
#SBATCH --mail-type=ALL        
#SBATCH --mem=10GB

module load nvidia/cudasdk/11.6
#9.2


cd /home/agensale/rora_tesi_new

# source ~/.bashrc 

# activate the environment
source ~/miniconda3/bin/activate rora_venv


# check if the program is running on the right environment
which pip && which python


python tasks/main_EMD.py \
    --n_epochs 20 \
    --bert_model roberta-base \
    --model_type bertweet-token-crf \
    --data /home/agensale/rora_tesi_new/data/Tweet-Fid \
    --learning_rate 1e-5 \
    --max_length 128 \
    --save_model \