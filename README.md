# modernpatentBERT
Experiments require GPU resources. If on slurm, request an interactive session with
```bash
salloc -c 4 -GL40S:1
```

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
To finetune ModernBERT on USPTO-3M, simply run the following:
```python
python3 finetune.py # add -h to see flags you can pass
```

## Pretrain
To pretrain ModernBERT on USPTO-3M, simply run the following:
```python
python pretrain.py # add -h to see flags you can pass
```

## Evaluate a Fine-Tuned Model
First ensure you have the test set downloaded to the `./uspto_3m_test_sets` directory. You can do this by running `python3 create_test_set.py`. Then, to evaluate a finetuned model, run the following
```python
python eval_patent_bert_test.py --model-path /path/to/model/checkpoint
```
