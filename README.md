# modernpatentBERT

## Environment Creation
```bash
conda env create -f environment.yml # this should install flash attention by default

conda activate mbertft
```
Then do 
```bash
cp .env.example .env
```
And complete the .env file

## Finetune
```python
python3 finetune.py # add -h to see flags you can pass
```

## Pretrain
```python
python pretrain.py # add -h to see flags you can pass
```
