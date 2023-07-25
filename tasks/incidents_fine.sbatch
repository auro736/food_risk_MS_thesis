#!/bin/sh
#SBATCH --job-name=incidents_finetuned
#SBATCH --output=/home/agensale/output_efra/incidents_finetuned.out
#SBATCH --nodes=1            
#SBATCH --ntasks=1     
#SBATCH --partition=cuda        
#SBATCH --cpus-per-task=1       
#SBATCH --gres=gpu:1   
#SBATCH --time=00:30:00          
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
    --bert_model roberta-large \
    --n_epochs 6 \
    --from_finetuned \

