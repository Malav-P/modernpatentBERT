# modernpatentBERT

## Environment Creation
```bash
conda env create -f environment.yaml # this should install flash attention by default

conda activate mbertft
```
Then do 
```bash
cp .env.example .env
```
And complete the .env file

## Submitting A Test Job for Finetuning
```bash
# ensure you are in the directory where the .sbatch file is located
srun sbatch test.sbatch
```
