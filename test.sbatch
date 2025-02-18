#!/bin/bash
#SBATCH --job-name=FINETUNE_SCRIPT_TEST               # Job name  
#SBATCH --gres=gpu:H200:1                # GPU type (L40S. H200. etc) and number of GPUs 
#SBATCH -t1:00:00                        # Duration of the job (Ex: 1 hour) 
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences 
#SBATCH --mail-user=$EMAIL_SBATCH # E-mail address for notifications 

cd $SLURM_SUBMIT_DIR

module unload anaconda3
module load anaconda3
conda activate mbertft

python3 finetune.py
