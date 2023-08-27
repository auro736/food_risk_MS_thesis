#!/bin/sh
#SBATCH --job-name=incidents_no_finetuned
#SBATCH --output=/home/agensale/output_efra/incidents_no_finetuned.out
#SBATCH --nodes=1            
#SBATCH --ntasks=1     
#SBATCH --partition=cuda        
#SBATCH --cpus-per-task=1       
#SBATCH --gres=gpu:1   
#SBATCH --time=08:00:00          
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


python tasks/main_efra_incidents.py \
    --n_epochs 6 \
    --bert_model roberta-large \
    --model_type bertweet-token-crf \
    --data /home/agensale/rora_tesi_new/data/SampleAgroknow \
    --learning_rate 1e-6 \
    --max_length 200 \