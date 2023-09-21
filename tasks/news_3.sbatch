#!/bin/sh
#SBATCH --job-name=news_3
#SBATCH --output=/home/agensale/output_efra/news_3.out
#SBATCH --nodes=1            
#SBATCH --ntasks=1     
#SBATCH --partition=cuda        
#SBATCH --cpus-per-task=1       
#SBATCH --gres=gpu:1   
#SBATCH --time=02:10:00          
#SBATCH --mail-type=ALL        
#SBATCH --mem=5GB

module load nvidia/cudasdk/11.6
#9.2


cd /home/agensale/rora_tesi_new

# source ~/.bashrc 

# activate the environment
source ~/miniconda3/bin/activate rora_venv


# check if the program is running on the right environment
which pip && which python


python tasks/main_efra_news.py \
    --bert_model xlm-roberta-large \
    --n_epochs 10 \
    --data /home/agensale/rora_tesi_new/data/SampleAgroknow \
    --assign_weight \
    --from_finetuned \
    --saved_model_path /home/agensale/rora_tesi/log_rora_tesi/log-tweet-classification/xlm-roberta-large/bertweet-seq/14_epoch/data/True_weight/42_seed/saved-model/