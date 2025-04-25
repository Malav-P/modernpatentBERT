import argparse # Added
from functools import partial
import json
import os
# Use LabelEncoder from sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
# import joblib # LabelEncoder can be saved/loaded directly or recreated from classes_
import shutil
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# --- Configuration (Defaults will be set by argparse) ---
YELLOW = '\033[33m'
RESET = '\033[0m'

# Function to preprocess labels (split comma-separated strings - MORE ROBUST)
def preprocess_labels_split(batch):
    processed_labels_batch = []
    # Make sure 'labels' column exists
    if 'labels' not in batch:
        raise KeyError("Column 'labels' not found in batch during preprocess_labels_split. Check raw dataset.")

    for labels_obj in batch['labels']: # Use a different name
        labels_list = []
        if isinstance(labels_obj, str):
            # Split, strip whitespace from each part, filter out empty strings
            labels_list = [label.strip() for label in labels_obj.split(',') if label.strip()]
        elif isinstance(labels_obj, list):
            # Assume list contains strings, strip and filter empties
            labels_list = [str(label).strip() for label in labels_obj if str(label).strip()]
        # Handle None or other unexpected types gracefully by leaving labels_list empty
        processed_labels_batch.append(labels_list)
    return {'labels_list': processed_labels_batch} # Store in a new column temporarily


# Function to convert text labels to one-hot encoding
def convert_labels_one_hot(example, label_encoder, num_labels):
    current_labels_list = example['labels_list'] # Direct access - will raise KeyError if missing
    known_labels = [
        lbl for lbl in current_labels_list
        if isinstance(lbl, str) and lbl in label_encoder.classes_
    ]
    indices = [] # Initialize indices to avoid potential UnboundLocalError if known_labels is empty
    if known_labels:
        indices = label_encoder.transform(known_labels)

    labels_one_hot = np.zeros(num_labels, dtype=np.float32)
    if len(indices) > 0:
        labels_one_hot[indices] = 1.0
    example["labels"] = labels_one_hot
    return example


# Tokenize function
def tokenize(batch, tokenizer):
    return tokenizer(batch['text'], truncation=True, padding=False, max_length=1024)


# --- Main Execution ---
if __name__ == "__main__": # Added if __name__ == "__main__": block

    parser = argparse.ArgumentParser(description="Evaluate a multi-label text classification model.") # Added
    parser.add_argument( # Added
        '--model-path',
        default="./model_mbert/",
        type=str,
        help="Path to your trained model/checkpoint directory"
    )
    parser.add_argument( # Added
        '--tokenizer-id',
        default="answerdotai/ModernBERT-base",
        type=str,
        help="Tokenizer identifier used during training (e.g., from Hugging Face Hub)"
    )
    parser.add_argument( # Added
        '--test-sets-dir',
        default="./uspto_3m_test_sets/",
        type=str,
        help="Directory where test set subdirectories (e.g., '2015-B') are saved"
    )
    parser.add_argument( # Added
        '--label-stats-file',
        default="class_stats.txt",
        type=str,
        help="Path to the class statistics JSON file containing known classes"
    )
    parser.add_argument( # Added
        '--eval-batch-size',
        default=64,
        type=int,
        help="Batch size per device for evaluation"
    )
    parser.add_argument( # Added
        '--num-proc',
        default=None, # Default to None to signal auto-detection
        type=int,
        help="Number of processes for dataset mapping operations (default: auto-detect, max(cpu_count // 2, 1))"
    )
    parser.add_argument( # Added
        '--test-set-names',
        default=["2016"],
        nargs='+', # Expect one or more arguments
        help="List of test set names (subdirectories in test_sets_dir) to evaluate"
    )
    parser.add_argument( # Added
        '--output-eval-dir',
        default="./eval_temp_output",
        type=str,
        help="Temporary directory for Hugging Face Trainer intermediate files during evaluation"
    )
    args = parser.parse_args() # Added

    # --- Assign Args to Variables --- # Added block
    MODEL_PATH = args.model_path
    TOKENIZER_ID = args.tokenizer_id
    TEST_SETS_DIR = args.test_sets_dir
    LABEL_STATS_FILE = args.label_stats_file
    EVAL_BATCH_SIZE = args.eval_batch_size
    TEST_SET_NAMES = args.test_set_names
    output_eval_dir = args.output_eval_dir # Used for TrainingArguments

    # Handle NUM_PROC: Use arg if provided, otherwise auto-detect
    if args.num_proc is None:
        NUM_PROC = max(os.cpu_count() // 2, 1)
    else:
        NUM_PROC = args.num_proc

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Keep dynamic detection
    # --- End Assign Args ---

    # --- Load Label Encoder --- (Moved inside main block)
    print(f"Loading known classes from: {LABEL_STATS_FILE}...")
    try: # Added basic error handling for file access
        with open(LABEL_STATS_FILE, 'r') as f:
            stats = json.load(f)
        known_classes = list(stats["counts"].keys())
        label_encoder = LabelEncoder().fit(known_classes)
        num_labels = len(label_encoder.classes_)
        print(f"Loaded {num_labels} known classes.")
    except FileNotFoundError:
        print(f"{YELLOW}Error: Label stats file not found at '{LABEL_STATS_FILE}'. Exiting.{RESET}")
        exit()
    except (json.JSONDecodeError, KeyError) as e:
        print(f"{YELLOW}Error reading or parsing label stats file '{LABEL_STATS_FILE}': {e}. Exiting.{RESET}")
        exit()

    # --- Load Tokenizer --- (Moved inside main block)
    print(f"Loading tokenizer: {TOKENIZER_ID}...")
    try: # Added basic error handling
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    except Exception as e:
        print(f"{YELLOW}Error loading tokenizer '{TOKENIZER_ID}': {e}. Exiting.{RESET}")
        exit()


    # 3. Load Model (Keep basic error handling here)
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
        exit()

    # 4. Instantiate Trainer
    print("Initializing Trainer for prediction...")
    # output_eval_dir defined above from args
    dummy_training_args = TrainingArguments(
        output_dir=output_eval_dir, # Use arg value
        per_device_eval_batch_size=EVAL_BATCH_SIZE, # Use arg value
        dataloader_num_workers=min(NUM_PROC, 4), # Use calculated NUM_PROC
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        bf16_full_eval=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        label_names=["labels"],
        remove_unused_columns=False, 
        dataloader_pin_memory=True, # Added for potential speed up
        # No training-specific args needed
    )
    hf_data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
    trainer = Trainer(
        model=model,
        args=dummy_training_args,
        data_collator=hf_data_collator,
        tokenizer=tokenizer
    )

    results = {}
    COLUMNS_TO_KEEP = ["input_ids", "attention_mask", "labels"] # Still useful to know expected output columns

    for test_name in TEST_SET_NAMES: # Use arg value
        print(f"\n--- Evaluating on Test Set: {test_name} ---")
        start_time = time.time()
        test_set_path = os.path.join(TEST_SETS_DIR, test_name) # Use arg value
        # processed_dataset_path = os.path.join(test_set_path, "processed_eval_cache") # Not used anymore

        # --- Load Raw Dataset ---
        print(f"Loading raw test set from: {test_set_path}...")
        # load_from_disk will raise error if path not found or corrupted
        try: # Added basic error handling for dataset loading
            eval_dataset_raw = load_from_disk(test_set_path)
        except FileNotFoundError:
            print(f"{YELLOW}Error: Test set directory not found at '{test_set_path}'. Skipping.{RESET}")
            results[test_name] = {"Error": f"Dataset not found at {test_set_path}"}
            continue # Skip to next test set
        except Exception as e:
             print(f"{YELLOW}Error loading dataset from '{test_set_path}': {e}. Skipping.{RESET}")
             results[test_name] = {"Error": f"Failed to load dataset: {e}"}
             continue # Skip to next test set


        # Check if 'cpc_ids' exists before renaming
        if 'cpc_ids' in eval_dataset_raw.column_names:
             eval_dataset_raw = eval_dataset_raw.rename_column("cpc_ids", "labels")
        elif 'labels' not in eval_dataset_raw.column_names:
             print(f"{YELLOW}Error: Neither 'cpc_ids' nor 'labels' column found in raw dataset for {test_name}. Skipping.{RESET}")
             results[test_name] = {"Error": "Missing label column ('cpc_ids' or 'labels')"}
             continue

        print(f"Loaded {len(eval_dataset_raw)} raw examples.")

        # --- Preprocess dataset ---
        print("Preprocessing dataset (splitting labels, one-hot encoding, tokenizing)...")
        columns_before_processing = eval_dataset_raw.column_names

        try: # Wrap processing steps in try-except
            eval_dataset_processed = eval_dataset_raw.map(
                preprocess_labels_split,
                batched=True,
                batch_size=1000,
                num_proc=NUM_PROC # Use calculated NUM_PROC
            )
            print("Step 1/3: Label splitting complete.")

            # b) Convert to one-hot labels
            # convert_labels_one_hot
            convert_func = partial(convert_labels_one_hot, label_encoder=label_encoder, num_labels=num_labels)
            eval_dataset_processed = eval_dataset_processed.map(
                convert_func,
                num_proc=NUM_PROC, # Or set num_proc=1 explicitly if needed for debugging map errors
            )
            print("Step 2/3: One-hot encoding complete.")

            # c) Tokenize text
            tokenize_func = partial(tokenize, tokenizer=tokenizer)
            eval_dataset_processed = eval_dataset_processed.map(
                tokenize_func,
                batched=True,
                batch_size=1000,
                num_proc=NUM_PROC, # Use calculated NUM_PROC
            )
            print("Step 3/3: Tokenization complete.")

            # d) Remove columns NOT needed for the model
            final_columns = set(eval_dataset_processed.column_names)
            original_columns = set(columns_before_processing)
            columns_to_remove = list((final_columns - set(COLUMNS_TO_KEEP)) | (original_columns - set(COLUMNS_TO_KEEP)))
            columns_to_remove = [col for col in columns_to_remove if col in eval_dataset_processed.column_names]
            if 'labels_list' in eval_dataset_processed.column_names and 'labels_list' not in COLUMNS_TO_KEEP:
                 if 'labels_list' not in columns_to_remove: columns_to_remove.append('labels_list')

            # Make sure essential columns are not accidentally removed
            for essential_col in COLUMNS_TO_KEEP:
                if essential_col in columns_to_remove:
                    print(f"{YELLOW}Warning: Attempting to remove essential column '{essential_col}'. Keeping it.{RESET}")
                    columns_to_remove.remove(essential_col)

            print(f"Removing unused columns: {columns_to_remove}")
            eval_dataset_processed = eval_dataset_processed.remove_columns(columns_to_remove)
            print("Preprocessing complete.")

            # --- Get predictions using Trainer ---
            print("Getting model predictions...")
            # trainer.predict will raise errors if model fails, data format is wrong, OOM, etc.
            predictions_output = trainer.predict(eval_dataset_processed)
            logits = predictions_output.predictions
            true_labels_one_hot = predictions_output.label_ids

            # Check if labels were returned
            if true_labels_one_hot is None:
                 print(f"{YELLOW}Error: Trainer did not return label_ids for {test_name}. Cannot calculate metrics. Skipping.{RESET}")
                 results[test_name] = {"Error": "Trainer predict call failed to return labels"}
                 continue # Skip metrics calculation

            # Ensure predictions and labels are numpy arrays
            if not isinstance(logits, np.ndarray):
                 logits = np.array(logits)
            if not isinstance(true_labels_one_hot, np.ndarray):
                 true_labels_one_hot = np.array(true_labels_one_hot)


            print("Prediction complete.")


            # --- Calculate P@1, R@1, F1@1 ---
            print("Calculating metrics (P@1, R@1, F1@1)...")

            # These operations will raise errors on type mismatches, shape mismatches, division by zero etc.
            probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy() # Ensure tensor is float32
            top_pred_indices = np.argmax(probs, axis=1)
            N = len(true_labels_one_hot) 

            if N == 0:
                 print(f"{YELLOW}Warning: No samples found after processing for {test_name}. Skipping metrics.{RESET}")
                 results[test_name] = {"Error": "Zero samples after processing"}
                 continue

            true_labels_bool = true_labels_one_hot.astype(bool)

            sample_indices = np.arange(N)
            correct_at_1_mask = true_labels_bool[sample_indices, top_pred_indices]
            correct_at_1 = correct_at_1_mask.sum()

            # Avoid division by zero
            precision_at_1 = correct_at_1 / N if N > 0 else 0.0

            total_true_positives = true_labels_bool.sum()
            true_labels_indices = [set(np.where(row == 1)[0]) for row in true_labels_one_hot]
            recall_values = [
                1 / len(true) if pred in true else 0
                for pred, true in zip(top_pred_indices, true_labels_indices)
            ]
            recall_at_1 = np.mean(recall_values)

            # THIS IS WRONG, SO COMMENTED OUT
            # recall_at_1 = correct_at_1 / total_true_positives if total_true_positives > 0 else 0.0 

            if precision_at_1 + recall_at_1 == 0:
                f1_at_1 = 0.0
            else:
                f1_at_1 = 2 * (precision_at_1 * recall_at_1) / (precision_at_1 + recall_at_1)

            # Store results only if calculation succeeds
            results[test_name] = {
                "Precision@1": precision_at_1,
                "Recall@1": recall_at_1,
                "F1@1": f1_at_1,
                "Num_Samples": N,
                "Total_True_Positives": int(total_true_positives),
                "Correct@1": int(correct_at_1)
            }
            print(f"Metrics for {test_name}: P@1={precision_at_1:.4f}, R@1={recall_at_1:.4f}, F1@1={f1_at_1:.4f}")

        except Exception as e: # Catch errors during processing/prediction/metrics
             print(f"{YELLOW}An error occurred during processing or evaluation for {test_name}: {e}. Skipping.{RESET}")
             # You might want more specific error handling here depending on common issues
             import traceback
             traceback.print_exc() # Print traceback for debugging
             results[test_name] = {"Error": f"Processing/Evaluation failed: {e}"}
             continue # Skip to next test set


        eval_duration = time.time() - start_time
        print(f"Evaluation for {test_name} took {eval_duration:.2f} seconds.")


    # Clean up temporary directory
    if os.path.exists(dummy_training_args.output_dir):
        print(f"Cleaning up temporary directory: {dummy_training_args.output_dir}")
        try: # Add try-except for cleanup
            shutil.rmtree(dummy_training_args.output_dir)
        except OSError as e:
             print(f"{YELLOW}Warning: Could not remove temporary directory '{dummy_training_args.output_dir}': {e}{RESET}")

    # --- Print Final Results ---
    print("\n--- Final Evaluation Results ---")
    for test_name, metrics in results.items():
        print(f"\nTest Set: {test_name}")
        if "Error" in metrics: # Check if an error occurred for this set
            print(f"  {YELLOW}Evaluation failed: {metrics['Error']}{RESET}")
        else:
            # No need to check for "Error" key anymore
            print(f"  Number of Samples:       {metrics['Num_Samples']}")
            print(f"  Total True Positives:    {metrics['Total_True_Positives']}")
            print(f"  Correct @ 1:           {metrics['Correct@1']}")
            print(f"  Precision@1:           {metrics['Precision@1']:.4f}")
            print(f"  Recall@1:              {metrics['Recall@1']:.4f}")
            print(f"  F1 Score@1:            {metrics['F1@1']:.4f}")

    print("\n--- Evaluation Script Finished ---")