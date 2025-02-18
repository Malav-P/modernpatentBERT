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
sbatch --export=EMAIL_SBATCH=groy8@gatech.edu test.sbatch
```
if you have your email in env var
```bash
EMAIL=$(source .env; echo $EMAIL_SBATCH) && sed "s/EMAIL_PLACEHOLDER/$EMAIL/" test.sbatch | sbatch
```
