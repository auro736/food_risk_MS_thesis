#!/bin/sh
#SBATCH --job-name=news_no_fine
#SBATCH --output=/home/agensale/output_efra/news_no_fine.out
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


python tasks/main_efra_news.py \
    --bert_model cardiffnlp/twitter-roberta-large-2022-154m \
    --n_epochs 10 \
    --model_type bertweet-seq \
    --data /home/agensale/rora_tesi_new/data/SampleAgroknow \
