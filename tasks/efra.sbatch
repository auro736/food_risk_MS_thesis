#!/bin/sh
#SBATCH --job-name=bin_class_tweet_large
#SBATCH --output=/home/agensale/output_efra/xlm_for_ner.out
#SBATCH --nodes=1            
#SBATCH --ntasks=1     
#SBATCH --partition=cuda        
#SBATCH --cpus-per-task=1       
#SBATCH --gres=gpu:1   
#SBATCH --time=00:20:00          
#SBATCH --mail-type=ALL        
#SBATCH --mem=10GB

module load nvidia/cudasdk/11.6
#9.2


cd /home/agensale/rora_tesi

# source ~/.bashrc 

# activate the environment
source ~/miniconda3/bin/activate rora_venv


# check if the program is running on the right environment
which pip && which python



python tweet_classification.py \
    --bert_model cardiffnlp/twitter-roberta-large-2022-154m \
    --n_epochs 24