import os
import argparse

from dotenv import load_dotenv
from transformers import (
    TrainingArguments,
    Trainer
)

from preprocessing import get_modernbert
from preprocessing import get_dataset


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
        default=96,
        type=int,
        help="per device batch size"
    )
    parser.add_argument(
        '--lr',
        default = 8e-4,
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
        default=1e-6,
        type=float,
        help="weight decay (i.e. L2 regularization)"
    )
    parser.add_argument(
        '--eval-steps',
        default=10000,
        type=int,
        help="number of training steps between each eval"
    )
    parser.add_argument(
        '--logging-steps',
        default=10,
        type=int,
        help="number of train steps between each log"
    )
    parser.add_argument(
        '--resume-from-checkpoint',
        default=False,
        type=bool,
        help="whether to resume training from checkpoint"
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
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        adam_epsilon=1e-6,
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        logging_steps=args.logging_steps,          # Log every 100 steps
        eval_strategy="steps",      
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=5,
        load_best_model_at_end=True,
        bf16=True,
        bf16_full_eval=True,
        push_to_hub=False,
        disable_tqdm=args.disable_tqdm,
        remove_unused_columns = False,
        report_to = report_to
    )

    
    dataset, collator, _ = get_dataset(task="mlm")
    model = get_modernbert(task="mlm")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator)
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)