import argparse
import os
import torch

from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    TrainingArguments,
    Trainer
)

from preprocessing import get_modernbert
from preprocessing import get_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to probabilities
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    # Apply threshold for binary predictions
    threshold = 0.5
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(labels, preds, average="macro")
    prec_mic, rec_mic, f1_mic, _ = precision_recall_fscore_support(labels, preds, average="micro")
    
    return {
        "accuracy": acc,
        "micro_f1": f1_mic,
        "macro_f1": f1_mac,
        "micro_recall": rec_mic,
        "macro_recall": rec_mac,
        "micro_precision": prec_mic,
        "macro_precision": prec_mac
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable-tqdm',
        action='store_true',
        help='Disable tqdm progress bars if specified.'
    )
    parser.add_argument(
        '--epochs',
        default=1,
        type=int,
        help="number of epochs to train"
    )
    parser.add_argument(
        '--batchsize',
        default=32,
        type=int,
        help="per device batch size"
    )
    parser.add_argument(
        '--lr',
        default = 8e-5,
        type=float,
        help="learning rate"
    )
    parser.add_argument(
        '--beta1',
        default=0.9,
        type=float,
        help="beta 1 hyperparameter for adam"
    )
    parser.add_argument(
        '--beta2',
        default=0.98,
        type=float,
        help="beta 2 hyperparameter for adam"
    )
    parser.add_argument(
        '--weight-decay',
        default=8e-6,
        type=float,
        help="weight decay (i.e. L2 regularization)"
    )
    parser.add_argument(
        '--eval-steps',
        default=500,
        type=int,
        help="number of training steps between each eval and equivalently between each checkpoint"
    )
    parser.add_argument(
        '--logging-steps',
        default=10,
        type=int,
        help="number of train steps between each log"
    )
    parser.add_argument(
        '--resume-from-checkpoint',
        action="store_true",
        help="whether to resume from latest checkpoint"
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
        output_dir=f"aai_ModernBERT_ft",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batchsize,
        per_device_eval_batch_size=args.batchsize,
        num_train_epochs=args.epochs,
        # max_steps=10,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        adam_epsilon=1e-6,
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        logging_steps=args.logging_steps,         
        eval_strategy="steps",      
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=5*args.eval_steps,
        save_total_limit=5,
        load_best_model_at_end=True,
        bf16=True,
        bf16_full_eval=True,
        push_to_hub=False,
        disable_tqdm=args.disable_tqdm,
        remove_unused_columns = False,
        report_to = report_to
    )

    dataset, collator, num_labels = get_dataset(task="cls") # num labels is 665
    model = get_modernbert(task="cls", num_labels=num_labels)
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        compute_metrics=compute_metrics)
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model("model_mbert/")