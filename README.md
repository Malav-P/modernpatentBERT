# modernpatentBERT

## Environment Creation
```bash
conda env create -f environment.yaml # this should install flash attention by default

conda activate mbertft
```

## Submitting A Test Job for Finetuning
```bash
# if necessary, export your huggingface authentication token
export HF_TOKEN=<your token here>

# optional, wandb login token
export WANDB_API_KEY=<your token here>

# ensure you are in the directory where the .sbatch file is located
srun sbatch test.sbatch
```