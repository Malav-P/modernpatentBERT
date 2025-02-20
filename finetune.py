# to start jupyter server
# salloc -G <GPU name here>:1

# conda activate mberft
# jupyter notebook --ip=0.0.0.0 --port=8888

import os
import torch
import numpy as np
import torch
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from collections import Counter

from dotenv import load_dotenv

load_dotenv()
YELLOW = '\033[33m'
RESET = '\033[0m'
RATIO_SIZE_OG_DATASET = float(os.getenv("RATIO_SIZE_OG_DATASET",0.5))
#use the cache or not
CACHE_TOKENIZED = True
#remove the cache
CLEAR_TOKEN_CACHE=True


# Define cache path
CACHE_DIR = Path.home() / "scratch/.cache/modernBert"

# Create the directory if it doesn’t exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache_file_path = os.path.join(CACHE_DIR, "eval_tokenized.arrow")
#clear the cache
if CLEAR_TOKEN_CACHE:
    if os.path.exists(cache_file_path):
        os.remove(cache_file_path)
        print(f"Cache cleared: {cache_file_path}")
    else:
        print(f"No cache file found at: {cache_file_path}")




os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print(f"{YELLOW}Warning: cuda capable device not found, using cpu...{RESET}", flush=True)

hf_token = os.getenv("HF_TOKEN")

patents = load_dataset("MalavP/USPTO-3M", split="train")
# Split the dataset: 90% for "dummy" (discarded), 10% for the subset
mini_patents = patents.train_test_split(
    test_size=RATIO_SIZE_OG_DATASET,  # 10% of the original data
    shuffle=True,   # Randomize selection
    seed=42         # For reproducibility
)["test"]           # Keep the 10% test split
# mini_patents = patents

mini_patents = mini_patents.rename_column("cpc_ids", "labels")
# mini_patents = mini_patents.map(lambda x: {'labels': x['labels'].split(',')})
def preprocess_labels(batch):
    return {
        # Split each string in the batch
        'labels': [labels_str.split(',') for labels_str in batch['labels']]
    }

mini_patents = mini_patents.map(
    preprocess_labels,
    batched=True,  # Required for multiprocessing
    batch_size=1000,  # Adjust based on memory
    num_proc=6  # For 8 CPUs, leave 2 cores free
)
# import json

# def save_class_statistics(dataset, output_file):
#     """
#     Calculates:
#       1. The total number of unique classes
#       2. The total number of occurrences of each class
#     and writes the results to a text file.
#     """
#     # Flatten the list of lists
#     all_labels = [label for sublist in dataset["labels"] for label in sublist]
#     counts = Counter(all_labels)
#     total_unique = len(counts)
    
#     stats = {"total_unique": total_unique, "counts": dict(counts)}
    
#     # Write to file
#     with open(output_file, 'w') as f:
#         json.dump(stats, f, indent=4)
    
#     print(f"Class statistics saved to {output_file}")

# save_class_statistics(mini_patents,"stats.txt")

unique_labels = {label for sublist in mini_patents["labels"] for label in sublist}
NUM_LABELS = len(unique_labels)


# Initialize and fit the LabelEncoder
label_encoder = LabelEncoder().fit(list(unique_labels))

def convert_labels(example):
    indices = label_encoder.transform(example['labels'])
    labels = np.zeros(NUM_LABELS, dtype=float)
    labels[indices] = 1.0 # BCELossWithLogits requires target to contain floating point labels
    example["labels"] = labels
    return example

# Apply the transformation to the dataset
mini_patents = mini_patents.map(convert_labels)



# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-base"
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
# # Tokenize helper function
def tokenize(batch):
    # return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")
    return tokenizer(batch['text'], truncation=True, padding=True, max_length=1024, return_tensors="pt")

# # Tokenize dataset

# train_dataset = train_dataset.map(tokenize, batched=True)
# eval_dataset = eval_dataset.map(tokenize, batched=True)

tokenize_kwargs = {
        "batched": True,
        "batch_size": 1000,
        "num_proc": 6
    }
mini_patents = mini_patents.map(
    tokenize,
    **tokenize_kwargs,
)

split_datasets = mini_patents.train_test_split(test_size=0.05, seed=42)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

hf_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

torch.cuda.empty_cache()
torch.cuda.init()  # Add this line

# 6. Load the model for sequence classification (adjust num_labels as needed).
# model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=NUM_LABELS,use_flash_attention_2=True).to(device)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=NUM_LABELS,
    device_map="auto",  # Automatic GPU placement
    torch_dtype=torch.bfloat16  # Matches your training args
)
model.config.problem_type= "multi_label_classification"

# 7. Define a compute_metrics function to compute accuracy.
def compute_metrics(eval_pred):
    predictions, labels = eval_pred # shape (N, num classes) and shape (N, num classes)
    pred_labels = np.argmax(predictions, axis=-1)  # shape (N,)

    # Assuming labels are lists of labels, check if pred_labels are in any of the true labels
    correct = 0
    total = len(labels)

    for pred, true_labels in zip(pred_labels, labels):
        correct += true_labels[pred]

    # Accuracy calculation
    accuracy = correct / total
    return {"accuracy": accuracy}


# Set your hyperparameters and task identifier
train_bsz, val_bsz = 32, 32
lr = 8e-5 # the authors use 2e-5
betas = (0.9, 0.98)
n_epochs = 2
eps = 1e-6
wd = 8e-6
task = "your_task_name"  # Set this to an appropriate identifier for your task

wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    import wandb
    wandb.login(key=wandb_api_key)
    report_to = "wandb"
    wandb.init(project="patent-bert")

else:
    print(f"{YELLOW}Warning: wandb api key not found, wandb is disabled for this run.{RESET}", flush=True)
    os.environ["WANDB_MODE"] = "disabled"
    report_to = None

# 8. Define training arguments using your specified hyperparameters.
training_args = TrainingArguments(
    output_dir=f"aai_ModernBERT_{task}_ft",
    learning_rate=lr,
    per_device_train_batch_size=train_bsz,
    per_device_eval_batch_size=val_bsz,
    num_train_epochs=n_epochs,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    adam_beta1=betas[0],
    adam_beta2=betas[1],
    adam_epsilon=eps,
    weight_decay=wd,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    bf16=True,
    bf16_full_eval=True,
    push_to_hub=False,
    disable_tqdm=False,
    report_to = report_to
)



# 9. Create the Trainer with the compute_metrics function.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=hf_data_collator,
    compute_metrics=compute_metrics
)



trainer.train()

if wandb_api_key:
    wandb.finish()