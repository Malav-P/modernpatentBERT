# -*- coding: utf-8 -*-
import argparse
from functools import partial
import json
import os
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
import shutil
from datasets import load_dataset, Dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.utils import HfHubHTTPError
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import traceback

YELLOW = '\033[33m'
RESET = '\033[0m'

def preprocess_labels_split(batch):
    processed_labels_batch = []
    if 'labels' not in batch:
        raise KeyError("Column 'labels' not found in batch during preprocess_labels_split. Check raw dataset.")
    for labels_obj in batch['labels']:
        labels_list = []
        if isinstance(labels_obj, str):
            labels_list = [label.strip() for label in labels_obj.split(',') if label.strip()]
        elif isinstance(labels_obj, list):
            labels_list = [str(label).strip() for label in labels_obj if str(label).strip()]
        processed_labels_batch.append(labels_list)
    return {'labels_list': processed_labels_batch}


def convert_labels_one_hot(example, label_encoder, num_labels):
    current_labels_list = example['labels_list']
    known_labels = [
        lbl for lbl in current_labels_list
        if isinstance(lbl, str) and lbl in label_encoder.classes_
    ]
    indices = []
    if known_labels:
        indices = label_encoder.transform(known_labels)
    labels_one_hot = np.zeros(num_labels, dtype=np.float32)
    if len(indices) > 0:
        labels_one_hot[indices] = 1.0
    example["labels"] = labels_one_hot
    return example


def tokenize(batch, tokenizer):
    if 'text' not in batch:
        raise KeyError("Column 'text' not found in batch during tokenization. Check dataset on Hub.")
    return tokenizer(batch['text'], truncation=True, padding=False, max_length=1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a multi-label text classification model using datasets from Hugging Face Hub.")
    parser.add_argument(
        '--model-path',
        default="./model_mbert/",
        type=str,
        help="Path to your trained model/checkpoint directory"
    )
    parser.add_argument(
        '--tokenizer-id',
        default="answerdotai/ModernBERT-base",
        type=str,
        help="Tokenizer identifier used during training (e.g., from Hugging Face Hub)"
    )
    # --- Modified HF Arguments with Defaults ---
    parser.add_argument(
        '--hf-repo-base',
        # required=False, # No longer required
        default="0zo/google_patentsview_claims", # Default base repo
        type=str,
        help="Base Hugging Face repository identifier (e.g., 'username/repo_prefix'). The script will append '_<test-set-name>' to this base. Defaults to '0zo/google_patentsview_claims'."
    )
    parser.add_argument(
        '--test-set-names',
        # required=False, # No longer required
        default=["2016"], # Default test set suffix
        nargs='+',
        help="List of test set suffixes (e.g., '2015-B', '2016'). Full repo ID will be <hf-repo-base>_<test-set-name>. Defaults to ['2016']."
    )
    # --- End Modified HF Arguments ---
    parser.add_argument(
        '--label-stats-file',
        default="class_stats.txt",
        type=str,
        help="Path to the class statistics JSON file containing known classes"
    )
    parser.add_argument(
        '--eval-batch-size',
        default=64,
        type=int,
        help="Batch size per device for evaluation"
    )
    parser.add_argument(
        '--num-proc',
        default=None,
        type=int,
        help="Number of processes for dataset mapping operations (default: auto-detect)"
    )
    parser.add_argument(
        '--output-eval-dir',
        default="./eval_temp_output",
        type=str,
        help="Temporary directory for Hugging Face Trainer intermediate files"
    )
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    TOKENIZER_ID = args.tokenizer_id
    HF_REPO_BASE = args.hf_repo_base # Will use default "0zo/google_patentsview_claims" if not provided
    TEST_SET_NAMES = args.test_set_names # Will use default ["2016"] if not provided
    LABEL_STATS_FILE = args.label_stats_file
    EVAL_BATCH_SIZE = args.eval_batch_size
    output_eval_dir = args.output_eval_dir

    if args.num_proc is None:
        NUM_PROC = max(os.cpu_count() // 2, 1)
    else:
        NUM_PROC = args.num_proc
    print(f"Using {NUM_PROC} processes for dataset operations.")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    print(f"Loading known classes from: {LABEL_STATS_FILE}...")
    try:
        with open(LABEL_STATS_FILE, 'r') as f:
            stats = json.load(f)
        known_classes = list(stats["counts"].keys())
        label_encoder = LabelEncoder().fit(known_classes)
        num_labels = len(label_encoder.classes_)
        print(f"Loaded {num_labels} known classes.")
    except FileNotFoundError:
        print(f"{YELLOW}Error: Label stats file not found at '{LABEL_STATS_FILE}'. Exiting.{RESET}")
        exit(1)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"{YELLOW}Error reading or parsing label stats file '{LABEL_STATS_FILE}': {e}. Exiting.{RESET}")
        exit(1)

    print(f"Loading tokenizer: {TOKENIZER_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    except Exception as e:
        print(f"{YELLOW}Error loading tokenizer '{TOKENIZER_ID}': {e}. Exiting.{RESET}")
        exit(1)

    print(f"Loading model from: {MODEL_PATH}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        ).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"{YELLOW}Error loading model from '{MODEL_PATH}': {e}. Exiting.{RESET}")
        exit(1)

    print("Initializing Trainer for prediction...")
    dummy_training_args = TrainingArguments(
        output_dir=output_eval_dir,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        dataloader_num_workers=min(NUM_PROC, 4),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        bf16_full_eval=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )
    hf_data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
    trainer = Trainer(
        model=model,
        args=dummy_training_args,
        data_collator=hf_data_collator,
        tokenizer=tokenizer
    )

    results = {}
    COLUMNS_TO_KEEP = ["input_ids", "attention_mask", "labels"]

    # Now TEST_SET_NAMES will default to ['2016'] if not specified
    print(f"Target Hugging Face datasets: {', '.join([f'{HF_REPO_BASE}_{name}' for name in TEST_SET_NAMES])}")

    for test_name_suffix in TEST_SET_NAMES:
        repo_id = f"{HF_REPO_BASE}_{test_name_suffix}"
        print(f"\n--- Evaluating on Test Set: {test_name_suffix} (from HF Repo: {repo_id}) ---")
        start_time = time.time()

        print(f"Loading raw test set from Hugging Face Hub: {repo_id}...")
        try:
            eval_dataset_raw = load_dataset(repo_id, split="train")
            print(f"Loaded {len(eval_dataset_raw)} raw examples from {repo_id}.")
        except DatasetNotFoundError:
            print(f"{YELLOW}Error: Dataset repository not found on Hugging Face Hub: '{repo_id}'. Check name and access. Skipping.{RESET}")
            results[test_name_suffix] = {"Error": f"Dataset not found at {repo_id}"}
            continue
        except HfHubHTTPError as e:
            print(f"{YELLOW}Error: Hugging Face Hub HTTP error loading '{repo_id}': {e}. Check network/auth. Skipping.{RESET}")
            results[test_name_suffix] = {"Error": f"HF Hub HTTP error: {e}"}
            continue
        except Exception as e:
             print(f"{YELLOW}Error loading dataset '{repo_id}' from Hugging Face Hub: {e}. Skipping.{RESET}")
             results[test_name_suffix] = {"Error": f"Failed to load dataset: {e}"}
             continue

        column_names = eval_dataset_raw.column_names
        if 'cpc_ids' in column_names:
             print("Found 'cpc_ids' column, renaming to 'labels'.")
             eval_dataset_raw = eval_dataset_raw.rename_column("cpc_ids", "labels")
        elif 'labels' not in column_names:
             print(f"{YELLOW}Error: Neither 'cpc_ids' nor 'labels' column found in '{repo_id}'. Cols: {column_names}. Skipping.{RESET}")
             results[test_name_suffix] = {"Error": "Missing label column ('cpc_ids' or 'labels')"}
             continue
        else:
            print("Found 'labels' column.")

        if 'text' not in eval_dataset_raw.column_names:
             print(f"{YELLOW}Error: 'text' column not found in '{repo_id}'. Cols: {column_names}. Skipping.{RESET}")
             results[test_name_suffix] = {"Error": "Missing 'text' column"}
             continue
        else:
            print("Found 'text' column.")

        print("Preprocessing dataset...")
        columns_before_processing = eval_dataset_raw.column_names

        try:
            eval_dataset_processed = eval_dataset_raw.map(
                preprocess_labels_split, batched=True, batch_size=1000, num_proc=NUM_PROC
            )
            print("Step 1/3: Label splitting complete.")

            convert_func = partial(convert_labels_one_hot, label_encoder=label_encoder, num_labels=num_labels)
            eval_dataset_processed = eval_dataset_processed.map(convert_func, num_proc=NUM_PROC)
            print("Step 2/3: One-hot encoding complete.")

            tokenize_func = partial(tokenize, tokenizer=tokenizer)
            eval_dataset_processed = eval_dataset_processed.map(
                tokenize_func, batched=True, batch_size=1000, num_proc=NUM_PROC
            )
            print("Step 3/3: Tokenization complete.")

            final_columns = set(eval_dataset_processed.column_names)
            columns_to_remove = list(final_columns - set(COLUMNS_TO_KEEP))

            for essential_col in COLUMNS_TO_KEEP:
                if essential_col in columns_to_remove:
                    print(f"{YELLOW}Warning: Attempting to remove essential column '{essential_col}'. Keeping it.{RESET}")
                    columns_to_remove.remove(essential_col)

            if 'labels_list' in eval_dataset_processed.column_names and 'labels_list' not in COLUMNS_TO_KEEP:
                 if 'labels_list' not in columns_to_remove: columns_to_remove.append('labels_list')

            print(f"Removing unused columns: {columns_to_remove}")
            eval_dataset_processed = eval_dataset_processed.remove_columns(columns_to_remove)
            print(f"Preprocessing complete. Final columns: {eval_dataset_processed.column_names}")

            print("Getting model predictions...")
            predictions_output = trainer.predict(eval_dataset_processed)
            logits = predictions_output.predictions
            true_labels_one_hot = predictions_output.label_ids

            if true_labels_one_hot is None:
                 print(f"{YELLOW}Error: Trainer did not return label_ids for {test_name_suffix}. Check 'labels' column processing. Skipping metrics.{RESET}")
                 results[test_name_suffix] = {"Error": "Trainer predict call failed to return labels"}
                 continue

            if not isinstance(logits, np.ndarray): logits = np.array(logits)
            if not isinstance(true_labels_one_hot, np.ndarray): true_labels_one_hot = np.array(true_labels_one_hot)
            print("Prediction complete.")

            print("Calculating metrics (P@1, R@1, F1@1)...")
            probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
            top_pred_indices = np.argmax(probs, axis=1)
            N = len(true_labels_one_hot)

            if N == 0:
                 print(f"{YELLOW}Warning: No samples found after processing for {test_name_suffix}. Skipping metrics.{RESET}")
                 results[test_name_suffix] = {"Error": "Zero samples after processing"}
                 continue

            true_labels_bool = true_labels_one_hot.astype(bool)
            sample_indices = np.arange(N)
            correct_at_1_mask = true_labels_bool[sample_indices, top_pred_indices]
            correct_at_1 = correct_at_1_mask.sum()

            precision_at_1 = correct_at_1 / N if N > 0 else 0.0
            total_true_positives = true_labels_bool.sum()
            recall_at_1 = correct_at_1 / total_true_positives if total_true_positives > 0 else 0.0

            if precision_at_1 + recall_at_1 == 0:
                f1_at_1 = 0.0
            else:
                f1_at_1 = 2 * (precision_at_1 * recall_at_1) / (precision_at_1 + recall_at_1)

            results[test_name_suffix] = {
                "Precision@1": precision_at_1,
                "Recall@1": recall_at_1,
                "F1@1": f1_at_1,
                "Num_Samples": N,
                "Total_True_Positives": int(total_true_positives),
                "Correct@1": int(correct_at_1)
            }
            print(f"Metrics for {test_name_suffix}: P@1={precision_at_1:.4f}, R@1={recall_at_1:.4f}, F1@1={f1_at_1:.4f}")

        except KeyError as e:
             print(f"{YELLOW}KeyError during processing/evaluation for {test_name_suffix}: {e}. Check column names. Skipping.{RESET}")
             results[test_name_suffix] = {"Error": f"Missing column/KeyError: {e}"}
             continue
        except Exception as e:
             print(f"{YELLOW}An unexpected error occurred during processing/evaluation for {test_name_suffix}: {e}. Skipping.{RESET}")
             traceback.print_exc()
             results[test_name_suffix] = {"Error": f"Processing/Evaluation failed: {e}"}
             continue

        eval_duration = time.time() - start_time
        print(f"Evaluation for {test_name_suffix} took {eval_duration:.2f} seconds.")

    if os.path.exists(dummy_training_args.output_dir):
        print(f"Cleaning up temporary directory: {dummy_training_args.output_dir}")
        try:
            shutil.rmtree(dummy_training_args.output_dir)
        except OSError as e:
             print(f"{YELLOW}Warning: Could not remove temporary directory '{dummy_training_args.output_dir}': {e}{RESET}")

    print("\n--- Final Evaluation Results ---")
    for test_name, metrics in results.items():
        print(f"\nTest Set Suffix: {test_name}")
        if "Error" in metrics:
            print(f"  {YELLOW}Evaluation failed: {metrics['Error']}{RESET}")
        else:
            print(f"  Number of Samples:       {metrics['Num_Samples']}")
            print(f"  Total True Positives:    {metrics['Total_True_Positives']}")
            print(f"  Correct @ 1:           {metrics['Correct@1']}")
            print(f"  Precision@1:           {metrics['Precision@1']:.4f}")
            print(f"  Recall@1:              {metrics['Recall@1']:.4f}")
            print(f"  F1 Score@1:            {metrics['F1@1']:.4f}")

    print("\n--- Evaluation Script Finished ---")