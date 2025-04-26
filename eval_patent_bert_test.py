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
import threading
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    print("pynvml not found, GPU monitoring will be disabled.")

"""
python eval_patent_bert_test_2.py --model-paths model_best_hparams_2epochs model_bs16_lr3e-5 model_bs16_lr3e-5_hier0.5 model_gen_hier_lr4e-5_wd5e-5 model_gen_hier_lr5e-5_wd1e-5 model_gen_hier_lr7e-5_wd1e-5_lambda0.5 model_gen_hier_lr8e-5_wd1e-5 model_gen_hier_lr8e-5_wd1e-5_lambda0.5_2epochs model_gen_hier_lr8e-5_wd1e-5_lambda0.75 model_gen_hier_lr8e-5_wd8e-6_lambda0.5 model_gen_hier_lr9e-5_wd1e-5_lambda0.5 model_gen_lr4e-5_wd5e-5 model_gen_lr5e-5_wd1e-5 model_gen_lr8e-5_wd1e-5 model_grad_accum2_lambda2_lr5e-5 model_hier_lambda1.5_lr5e-5_weightsReduceGap model_hier_lambda1_lr5e-5 model_hier_lambda2_lr5e-5 model_hier_lambda2_lr5e-5_beta999 model_hier_lambda2_lr8e-5 model_hier_lambda3_lr4e-5 model_lambda2_lr5e-5_bs32 model_mbert_hierarchical_8e-5lr model_more_weight_decay_1e5 modern-bert-just-fine-tuned more_weight_decay_1e5
python eval_patent_bert_test_3.py --model-paths model_more_weight_decay_1e5 model_hier_lambda1.5_lr5e-5_weightsReduceGap model_lambda2_lr5e-5_bs32 model_bs16_lr3e-5_hier0.5 model_best_hparams_2epochs model_gen_hier_lr8e-5_wd1e-5_lambda0.5_2epochs model_gen_hier_lr7e-5_wd1e-5_lambda0.5 model_gen_hier_lr9e-5_wd1e-5_lambda0.5 model_gen_hier_lr8e-5_wd8e-6_lambda0.5 model_gen_hier_lr8e-5_wd1e-5_lambda0.75
"""

YELLOW = '\033[33m'
RESET = '\033[0m'

# --- GPU Monitoring Setup ---
gpu_monitoring_active = False
gpu_util_readings = []
gpu_mem_readings = []
monitor_stop_event = threading.Event()

def monitor_gpu(stop_event, interval=0.5):
    """Polls GPU utilization and memory usage in a separate thread."""
    global gpu_util_readings, gpu_mem_readings
    try:
        pynvml.nvmlInit()
        # Try to get GPU 0, handle potential errors if no GPU or NVML issue
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assumes GPU 0
            print(f"Started GPU monitoring (GPU 0)")
        except pynvml.NVMLError as e:
            print(f"{YELLOW}NVML Error getting GPU handle: {e}. GPU monitoring cannot proceed.{RESET}")
            return # Exit thread if handle cannot be obtained

        while not stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util_readings.append(util.gpu) # % utilization
                gpu_mem_readings.append(mem.used / (1024**2)) # MB used
            except pynvml.NVMLError as e:
                print(f"{YELLOW}NVML Error during monitoring poll: {e}. Attempting to continue...{RESET}")
                # Decide if you want to break or continue. Continuing might spam errors.
                # Breaking might be safer if errors persist. Let's sleep longer and retry.
                time.sleep(interval * 5) # Wait longer if error
            time.sleep(interval)

    except pynvml.NVMLError as e:
        # This might catch the init error if HAS_PYNVML was true but init failed
        print(f"{YELLOW}Could not initialize NVML for GPU monitoring: {e}. Monitoring disabled.{RESET}")
    except Exception as e:
        print(f"{YELLOW}Unexpected error in GPU monitor thread: {e}{RESET}")
    finally:
        # Ensure shutdown happens even if errors occurred
        if HAS_PYNVML:
            try:
                pynvml.nvmlShutdown()
                print("Stopped GPU monitoring.")
            except pynvml.NVMLError as e:
                 # If shutdown fails, it might be because init failed or already shut down
                 print(f"{YELLOW}NVML Error during shutdown: {e}{RESET}")


def start_gpu_monitor_thread():
    global gpu_monitoring_active, monitor_stop_event, gpu_util_readings, gpu_mem_readings
    # Only attempt if pynvml imported successfully AND we are using CUDA
    if HAS_PYNVML and DEVICE == "cuda":
        monitor_stop_event.clear()
        gpu_util_readings = []
        gpu_mem_readings = []
        # We attempt pynvml.nvmlInit() inside the thread now for better error handling
        monitor_thread = threading.Thread(target=monitor_gpu, args=(monitor_stop_event,), daemon=True)
        monitor_thread.start()
        # We assume it started successfully if thread created, actual check happens in thread
        gpu_monitoring_active = True
        return monitor_thread
    else:
        if DEVICE == "cuda" and not HAS_PYNVML:
            print(f"{YELLOW}GPU monitoring requested but pynvml is not installed. Skipping.{RESET}")
        gpu_monitoring_active = False
        return None

def stop_gpu_monitor_thread(thread):
    global gpu_monitoring_active
    if thread and gpu_monitoring_active:
        monitor_stop_event.set()
        thread.join(timeout=5) # Wait max 5s for thread to stop gracefully
        if thread.is_alive():
            print(f"{YELLOW}GPU monitoring thread did not stop gracefully.{RESET}")
        gpu_monitoring_active = False
        # NVML shutdown is handled within the thread's finally block

def calculate_gpu_stats():
    if not gpu_util_readings or not gpu_mem_readings:
        return {"avg_gpu_util": 0, "max_gpu_util": 0, "avg_gpu_mem_mb": 0, "max_gpu_mem_mb": 0}

    try:
        avg_util = np.mean(gpu_util_readings) if gpu_util_readings else 0
        max_util = np.max(gpu_util_readings) if gpu_util_readings else 0
        avg_mem = np.mean(gpu_mem_readings) if gpu_mem_readings else 0
        max_mem = np.max(gpu_mem_readings) if gpu_mem_readings else 0
        return {"avg_gpu_util": avg_util, "max_gpu_util": max_util, "avg_gpu_mem_mb": avg_mem, "max_gpu_mem_mb": max_mem}
    except Exception as e:
        print(f"{YELLOW}Error calculating GPU stats: {e}{RESET}")
        return {"avg_gpu_util": -1, "max_gpu_util": -1, "avg_gpu_mem_mb": -1, "max_gpu_mem_mb": -1} # Indicate error

# --- End GPU Monitoring Setup ---


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

def tokenize(batch, tokenizer, max_length=1024):
    if 'text' not in batch:
        raise KeyError("Column 'text' not found in batch during tokenization. Check dataset on Hub.")
    return tokenizer(batch['text'], truncation=True, padding=False, max_length=max_length)


if __name__ == "__main__":
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description="Evaluate multiple multi-label text classification models using datasets from Hugging Face Hub.")
    parser.add_argument('--model-paths', nargs='+', required=True, type=str, help="List of paths to your trained model/checkpoint directories")
    parser.add_argument('--hf-repo-base', default="0zo/google_patentsview_claims", type=str, help="Base Hugging Face repository identifier. Defaults to '0zo/google_patentsview_claims'.")
    parser.add_argument('--test-set-names', default=["2016"], nargs='+', help="List of test set suffixes. Defaults to ['2016'].")
    parser.add_argument('--label-stats-file', default="class_stats.txt", type=str, help="Path to the class statistics JSON file containing known classes")
    parser.add_argument('--eval-batch-size', default=64, type=int, help="Batch size per device for evaluation")
    parser.add_argument('--num-proc', default=None, type=int, help="Number of processes for dataset mapping operations (default: auto-detect)")
    parser.add_argument('--output-eval-dir', default="./eval_temp_output", type=str, help="Temporary directory for Hugging Face Trainer intermediate files (will be cleaned up)")
    parser.add_argument('--results-file', default="results.txt", type=str, help="File path to save the evaluation results")
    parser.add_argument('--use-bert', action='store_true', help="Use BERT instead of ModernBERT for evaluation", default=False)
    args = parser.parse_args()

    MODEL_PATHS = args.model_paths
    HF_REPO_BASE = args.hf_repo_base
    TEST_SET_NAMES = args.test_set_names
    LABEL_STATS_FILE = args.label_stats_file
    EVAL_BATCH_SIZE = args.eval_batch_size
    output_eval_dir_base = args.output_eval_dir
    RESULTS_FILE = args.results_file

    if args.use_bert:
        TOKENIZER_ID = "bert-base-uncased"
    else:
        TOKENIZER_ID = "answerdotai/ModernBERT-base"

    if args.num_proc is None:
        NUM_PROC = max(os.cpu_count() -5, 1)
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

    # --- Results File Setup ---
    with open(RESULTS_FILE, 'a') as f_results:
        f_results.write(f"--- Evaluation Run Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f_results.write(f"Tokenizer: {TOKENIZER_ID}\n")
        f_results.write(f"Label File: {LABEL_STATS_FILE}\n")
        f_results.write(f"Test Sets Base: {HF_REPO_BASE}\n")
        f_results.write(f"Test Set Suffixes: {', '.join(TEST_SET_NAMES)}\n")
        f_results.write(f"Device: {DEVICE}\n")
        f_results.write(f"Eval Batch Size: {EVAL_BATCH_SIZE}\n")
        f_results.write("--------------------------------------------------\n\n")

        total_models_time = 0

        # --- Loop through each model path ---
        for model_idx, current_model_path in enumerate(MODEL_PATHS):
            model_start_time = time.time() # Time the whole process for this model
            print(f"\n\n===== Evaluating Model {model_idx+1}/{len(MODEL_PATHS)}: {current_model_path} =====")
            f_results.write(f"Model Path: {current_model_path}\n")

            current_output_eval_dir = f"{output_eval_dir_base}_{model_idx}"

            print(f"Loading model from: {current_model_path}...")
            model_load_start = time.time()
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    current_model_path,
                    num_labels=num_labels,
                    problem_type="multi_label_classification"
                ).to(DEVICE)
                model.eval()
                model_load_duration = time.time() - model_load_start
                print(f"Model loaded in {model_load_duration:.2f} seconds.")
                # Log model load time separately if desired, but not focus of efficiency here
                # f_results.write(f"  Model Load Time (s): {model_load_duration:.2f}\n")
            except Exception as e:
                model_load_duration = time.time() - model_load_start
                error_msg = f"{YELLOW}Error loading model from '{current_model_path}' (after {model_load_duration:.2f}s): {e}. Skipping this model.{RESET}"
                print(error_msg)
                f_results.write(f"  Error: Failed to load model - {e}\n\n")
                continue

            print("Initializing Trainer for prediction...")
            dummy_training_args = TrainingArguments(
                output_dir=current_output_eval_dir,
                per_device_eval_batch_size=EVAL_BATCH_SIZE,
                dataloader_num_workers=min(NUM_PROC, 4), # Limit workers slightly for eval
                bf16=DEVICE == "cuda" and torch.cuda.is_bf16_supported(),
                bf16_full_eval=DEVICE == "cuda" and torch.cuda.is_bf16_supported(),
                report_to="none",
                label_names=["labels"],
                remove_unused_columns=False, # Keep necessary cols like 'labels'
                dataloader_pin_memory=True,
            )
            hf_data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
            trainer = Trainer(
                model=model,
                args=dummy_training_args,
                data_collator=hf_data_collator,
                tokenizer=tokenizer
            )

            model_results = {} # Store results per test set for the current model
            total_prediction_time_model = 0
            total_samples_model = 0
            COLUMNS_TO_KEEP = ["input_ids", "attention_mask", "labels"] # Essential for predict

            print(f"Target Hugging Face datasets: {', '.join([f'{HF_REPO_BASE}_{name}' for name in TEST_SET_NAMES])}")

            # --- Loop through each test set for the current model ---
            for test_name_suffix in TEST_SET_NAMES:
                repo_id = f"{HF_REPO_BASE}_{test_name_suffix}"
                print(f"\n--- Evaluating on Test Set: {test_name_suffix} (from HF Repo: {repo_id}) ---")
                # No timing needed for loading/preprocessing specifically for results

                print(f"Loading and preprocessing test set: {repo_id}...")
                monitor_thread = None # Ensure thread var exists
                try:
                    # --- Data Loading ---
                    eval_dataset_raw = load_dataset(repo_id, split="train")
                    print(f"Loaded {len(eval_dataset_raw)} raw examples.")

                    # --- Column Checks & Renaming ---
                    column_names = eval_dataset_raw.column_names
                    if 'cpc_ids' in column_names:
                         eval_dataset_raw = eval_dataset_raw.rename_column("cpc_ids", "labels")
                    elif 'labels' not in column_names:
                         raise ValueError(f"Neither 'cpc_ids' nor 'labels' column found. Cols: {column_names}")
                    if 'text' not in column_names:
                         raise ValueError(f"'text' column not found. Cols: {column_names}")

                    # --- Preprocessing ---
                    eval_dataset_processed = eval_dataset_raw.map(
                        preprocess_labels_split, batched=True, batch_size=1000, num_proc=NUM_PROC
                    )
                    convert_func = partial(convert_labels_one_hot, label_encoder=label_encoder, num_labels=num_labels)
                    eval_dataset_processed = eval_dataset_processed.map(convert_func, num_proc=NUM_PROC)

                    max_len = 512 if args.use_bert else 1024 # Use tokenizer default potentially
                    tokenize_func = partial(tokenize, tokenizer=tokenizer, max_length=max_len)
                    eval_dataset_processed = eval_dataset_processed.map(
                        tokenize_func, batched=True, batch_size=1000, num_proc=NUM_PROC
                    )

                    # --- Column Cleanup ---
                    final_columns = set(eval_dataset_processed.column_names)
                    columns_to_remove = list(final_columns - set(COLUMNS_TO_KEEP))
                    eval_dataset_processed = eval_dataset_processed.remove_columns(columns_to_remove)
                    print(f"Preprocessing complete. Final columns: {eval_dataset_processed.column_names}")


                    # --- Prediction (The part we measure) ---
                    print("Getting model predictions...")
                    monitor_thread = start_gpu_monitor_thread() # Start GPU monitor just before predict
                    predict_start_time = time.time()

                    predictions_output = trainer.predict(eval_dataset_processed)

                    predict_duration = time.time() - predict_start_time
                    stop_gpu_monitor_thread(monitor_thread) # Stop GPU monitor immediately after
                    gpu_stats = calculate_gpu_stats() # Calculate stats from the readings

                    logits = predictions_output.predictions
                    true_labels_one_hot = predictions_output.label_ids # Should be present now

                    if true_labels_one_hot is None:
                         raise ValueError("Trainer did not return label_ids. Check 'labels' column processing.")
                    if not isinstance(logits, np.ndarray): logits = np.array(logits)
                    if not isinstance(true_labels_one_hot, np.ndarray): true_labels_one_hot = np.array(true_labels_one_hot)

                    N = len(true_labels_one_hot)
                    total_samples_model += N
                    total_prediction_time_model += predict_duration
                    throughput = N / predict_duration if predict_duration > 0 else 0
                    print(f"Prediction complete in {predict_duration:.2f} seconds ({throughput:.2f} samples/sec).")
                    if DEVICE == "cuda" and HAS_PYNVML:
                        print(f"  GPU Stats (Prediction): Avg Util={gpu_stats['avg_gpu_util']:.1f}%, Max Util={gpu_stats['max_gpu_util']:.1f}%, Avg Mem={gpu_stats['avg_gpu_mem_mb']:.0f}MB, Max Mem={gpu_stats['max_gpu_mem_mb']:.0f}MB")


                    # --- Metric Calculation ---
                    print("Calculating metrics (P@1, R@1, F1@1)...")
                    probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
                    # print(f"Predictions shape: {probs.shape}")
                    top_pred_indices = np.argmax(probs, axis=1)

                    if N == 0:
                        print(f"{YELLOW}Warning: No samples found for metric calculation in {test_name_suffix}. Skipping metrics.{RESET}")
                        model_results[test_name_suffix] = {
                            "Error": "Zero samples after processing",
                            # Still record prediction efficiency even if metrics can't be calculated
                            "PredictTime_s": predict_duration,
                            "Throughput_samples_s": throughput,
                            "AvgGPUMemory_MB": gpu_stats["avg_gpu_mem_mb"],
                            "MaxGPUMemory_MB": gpu_stats["max_gpu_mem_mb"],
                            "AvgGPUUtil_percent": gpu_stats["avg_gpu_util"],
                            "MaxGPUUtil_percent": gpu_stats["max_gpu_util"],
                        }
                        continue # Skip metrics calculation

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

                    model_results[test_name_suffix] = {
                        # Accuracy Metrics
                        "Precision@1": precision_at_1,
                        "Recall@1": recall_at_1,
                        "F1@1": f1_at_1,
                        "Num_Samples": N,
                        "Total_True_Positives": int(total_true_positives),
                        "Correct@1": int(correct_at_1),
                        # --- Prediction Efficiency Metrics ---
                        "PredictTime_s": predict_duration,
                        "Throughput_samples_s": throughput,
                        "AvgGPUMemory_MB": gpu_stats["avg_gpu_mem_mb"],
                        "MaxGPUMemory_MB": gpu_stats["max_gpu_mem_mb"],
                        "AvgGPUUtil_percent": gpu_stats["avg_gpu_util"],
                        "MaxGPUUtil_percent": gpu_stats["max_gpu_util"],
                        # ------------------------------------
                    }
                    print(f"Metrics for {test_name_suffix}: P@1={precision_at_1:.4f}, R@1={recall_at_1:.4f}, F1@1={f1_at_1:.4f}")

                except (DatasetNotFoundError, HfHubHTTPError, ValueError, KeyError) as e:
                    error_msg = f"{YELLOW}Error during loading/preprocessing/prediction for {test_name_suffix}: {e}. Skipping this test set.{RESET}"
                    print(error_msg)
                    # Ensure monitor thread is stopped if it was started before the error
                    stop_gpu_monitor_thread(monitor_thread)
                    model_results[test_name_suffix] = {"Error": f"Load/Preproc/Predict failed: {e}"}
                    continue
                except Exception as e:
                    error_msg = f"{YELLOW}An unexpected error occurred for {test_name_suffix}: {e}. Skipping.{RESET}"
                    print(error_msg)
                    traceback.print_exc()
                    # Ensure monitor thread is stopped if it was started before the error
                    stop_gpu_monitor_thread(monitor_thread)
                    model_results[test_name_suffix] = {"Error": f"Unexpected failure: {e}"}
                    continue

            # --- End test set loop ---

            # Calculate overall model efficiency
            model_duration = time.time() - model_start_time
            total_models_time += model_duration
            overall_model_throughput = total_samples_model / total_prediction_time_model if total_prediction_time_model > 0 else 0

            # --- Write results for the current model to file ---
            f_results.write(f"  Total Model Evaluation Time (s): {model_duration:.2f}\n") # Overall wall time
            f_results.write(f"  Total Prediction Time (s): {total_prediction_time_model:.2f}\n") # Sum of predict durations
            f_results.write(f"  Total Samples Predicted: {total_samples_model}\n")
            f_results.write(f"  Overall Prediction Throughput (samples/s): {overall_model_throughput:.2f}\n")
            f_results.write("  Results per Test Set:\n")
            for test_name, metrics in model_results.items():
                f_results.write(f"    Test Set Suffix: {test_name}\n")
                if "Error" in metrics:
                    f_results.write(f"      Error: {metrics['Error']}\n")
                    # Include partial efficiency if prediction happened before error
                    if "PredictTime_s" in metrics:
                         f_results.write(f"      PredictTime_s:         {metrics['PredictTime_s']:.2f} (before error)\n")
                         f_results.write(f"      Throughput_samples_s:  {metrics['Throughput_samples_s']:.2f} (before error)\n")
                         # GPU stats might be relevant even if metrics failed
                         f_results.write(f"      AvgGPUMemory_MB:       {metrics.get('AvgGPUMemory_MB', 0):.0f}\n")
                         f_results.write(f"      MaxGPUMemory_MB:       {metrics.get('MaxGPUMemory_MB', 0):.0f}\n")
                         f_results.write(f"      AvgGPUUtil_percent:    {metrics.get('AvgGPUUtil_percent', 0):.1f}\n")
                         f_results.write(f"      MaxGPUUtil_percent:    {metrics.get('MaxGPUUtil_percent', 0):.1f}\n")
                else:
                    # Accuracy metrics
                    f_results.write(f"      Num_Samples:           {metrics['Num_Samples']}\n")
                    f_results.write(f"      Correct@1:             {metrics['Correct@1']}\n")
                    f_results.write(f"      Precision@1:           {metrics['Precision@1']:.4f}\n")
                    f_results.write(f"      Recall@1:              {metrics['Recall@1']:.4f}\n")
                    f_results.write(f"      F1@1:                  {metrics['F1@1']:.4f}\n")
                    # Prediction Efficiency metrics
                    f_results.write(f"      PredictTime_s:         {metrics['PredictTime_s']:.2f}\n")
                    f_results.write(f"      Throughput_samples_s:  {metrics['Throughput_samples_s']:.2f}\n")
                    f_results.write(f"      AvgGPUMemory_MB:       {metrics['AvgGPUMemory_MB']:.0f}\n")
                    f_results.write(f"      MaxGPUMemory_MB:       {metrics['MaxGPUMemory_MB']:.0f}\n")
                    f_results.write(f"      AvgGPUUtil_percent:    {metrics['AvgGPUUtil_percent']:.1f}\n")
                    f_results.write(f"      MaxGPUUtil_percent:    {metrics['MaxGPUUtil_percent']:.1f}\n")
            f_results.write("\n")

            # --- Print results for the current model to console ---
            print(f"\n--- Prediction Efficiency Summary for Model: {current_model_path} ---")
            print(f"  Total Samples Predicted: {total_samples_model}")
            print(f"  Total Prediction Time: {total_prediction_time_model:.2f} s")
            print(f"  Overall Prediction Throughput: {overall_model_throughput:.2f} samples/s")
            print(f"  (Total Model Wall Time: {model_duration:.2f} s)") # Contextual info
            # Optionally print per-test-set details again or just the summary

            # --- Cleanup ---
            if os.path.exists(current_output_eval_dir):
                print(f"Cleaning up temporary directory: {current_output_eval_dir}")
                try:
                    shutil.rmtree(current_output_eval_dir)
                except OSError as e:
                     print(f"{YELLOW}Warning: Could not remove temporary directory '{current_output_eval_dir}': {e}{RESET}")

            if DEVICE == "cuda":
                del model # Explicitly delete model
                torch.cuda.empty_cache() # Clear cache
                print("Released model and cleared GPU cache.")

        # --- End model path loop ---

        script_end_time = time.time()
        script_duration = script_end_time - script_start_time

        f_results.write("--------------------------------------------------\n")
        f_results.write(f"Total script execution time: {script_duration:.2f} seconds\n")
        f_results.write(f"--- Evaluation Run Finished: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")

    print("\n--- Evaluation Script Finished ---")
    print(f"Total script execution time: {script_duration:.2f} seconds")
    print(f"Results focused on prediction efficiency saved to: {RESULTS_FILE}")
