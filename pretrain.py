import os
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

def get_modernbert():
    model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base", device_map="auto")
    return model

def dataset_for_mlm(test_size = 0.01, mlm_probability = 0.3):
    
    dataset = load_dataset("MalavP/USPTO-3M", split="train")
    model_name = "answerdotai/ModernBERT-base" 
    tokenizer = AutoTokenizer.from_pretrained(model_name) 

    # Tokenization function
    def transform(batch):
        batch["text"] = [text for text in batch["text"] if text is not None and text.strip() != ""]

        return tokenizer(batch["text"], truncation=True, padding=True, max_length=1024)

    dataset.set_transform(transform)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability  # 30% of tokens are masked
    )

    dataset = dataset.train_test_split(
        test_size=test_size,  # 10% of the original data
        shuffle=True,   # Randomize selection
        seed=42         # For reproducibility
    ) 

    return dataset, data_collator


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable-tqdm',
        action='store_true',
        help='Disable tqdm progress bars if specified.'
    )
    args = parser.parse_args()
    
    # load environment variables
    load_dotenv()
    if os.environ["WANDB_API_KEY"]:
        print("Found WANDB_API_KEY, logging to wandb...")
        report_to = "wandb"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"

    training_args = TrainingArguments(
        output_dir=f"ModernBERT_pretrain",
        overwrite_output_dir = True,
        learning_rate=8e-4,
        per_device_train_batch_size=96,
        per_device_eval_batch_size=96,
        num_train_epochs=2,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        weight_decay=1e-6,
        logging_strategy="steps",
        logging_steps=100,          # Log every 100 steps
        eval_strategy="steps",      
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,
        load_best_model_at_end=True,
        bf16=True,
        bf16_full_eval=True,
        push_to_hub=False,
        disable_tqdm=args.disable_tqdm,
        remove_unused_columns = False,
        report_to = report_to
    )

    model = get_modernbert()
    dataset, collator = dataset_for_mlm()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator)
    
    trainer.train()