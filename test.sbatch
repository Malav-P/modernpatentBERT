#!/bin/bash
#SBATCH --job-name=FINETUNE_SCRIPT_TEST
#SBATCH --gres=gpu:A100:1 --cpus-per-gpu=8
#SBATCH -t8:00:00
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=groy8@gatech.edu

# Initialize Conda (Mamba uses the same initialization)
source ~/.bashrc  # Load your shell config
eval "$(mamba shell hook --shell bash)"  # Initialize mamba
mamba activate mbertft 

echo "Using Python at: $(which python)"
echo "Python version: $(python --version)"
# Debug: Check PyTorch and environment
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Conda environment: $CONDA_PREFIX"

python3 finetune.py
