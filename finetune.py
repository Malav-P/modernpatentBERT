# to start jupyter server
# salloc -G <GPU name here>:1

# conda activate mberft
# jupyter notebook --ip=0.0.0.0 --port=8888

import os
import torch
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

YELLOW = '\033[33m'
RESET = '\033[0m'


wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    import wandb
    wandb.login(key=wandb_api_key)
    report_to = "wandb"

else:
    print(f"{YELLOW}Warning: wandb api key not found, wandb is disabled for this run.{RESET}", flush=True)
    os.environ["WANDB_MODE"] = "disabled"
    report_to = None

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print(f"{YELLOW}Warning: cuda capable device not found, using cpu...{RESET}", flush=True)

hf_token = os.getenv("HF_TOKEN")
patents = load_dataset("MalavP/USPTO-3M", split="train", use_auth_token = hf_token)
# Split the dataset: 90% for "dummy" (discarded), 10% for the subset
mini_patents = patents.train_test_split(
    test_size=0.01,  # 1% of the original data
    shuffle=True,   # Randomize selection
    seed=42         # For reproducibility
)["test"]           # Keep the 10% test split

mini_patents = mini_patents.rename_column("cpc_ids", "labels")
mini_patents = mini_patents.map(lambda x: {'labels': x['labels'].split(',')})
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
split_datasets = mini_patents.train_test_split(test_size=0.05, seed=42)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-base"
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
# Tokenize helper function
def tokenize(batch):
    # return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")
    return tokenizer(batch['text'], truncation=True, padding=True, max_length=1024, return_tensors="pt")

# Tokenize dataset
train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

hf_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. Load the model for sequence classification (adjust num_labels as needed).
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=NUM_LABELS).to(device)
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
n_epochs = 1
eps = 1e-6
wd = 8e-6
task = "your_task_name"  # Set this to an appropriate identifier for your task
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