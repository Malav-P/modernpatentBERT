import numpy as np
import json

from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset

from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding
)

def get_dataset(task: str, test_size: float = 0.005, mlm_probability: float = 0.3):
    """
    Gets the dataset, the collator, and the number of classes for the task (if required)

    Args:
        task : Either "mlm" for pretraining or "cls" for finetuning. If "cls" is passes, `mlm_probability` is ignored
        test_size : fraction of dataset reserved for validation
        mlm_probability: fraction of tokens that will be masked for the MLM objective

    Returns:
        dataset : an HF dataset object
        collator : a HF collator object
        num_labels : the number of labels as an integer. Is `None` if task is "mlm"
    """

    dataset = load_dataset("MalavP/USPTO-3M", split="train")
    model_name = "answerdotai/ModernBERT-base" 
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    
    if task == "mlm":
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
            test_size=test_size,  # 0.5% of the original data
            shuffle=True,   # Randomize selection
            seed=42         # For reproducibility
        )
        num_labels = None 

    elif task == "cls":
        transform, num_labels = get_cls_transform("class_stats.txt", tokenizer=tokenizer)
        dataset.set_transform(transform)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        dataset = dataset.train_test_split(
            test_size=test_size,  # 1% of the original data
            shuffle=True,   # Randomize selection
            seed=42         # For reproducibility
        ) 

    else:
        raise ValueError(f"task not valid, given: {task}")


    return dataset, data_collator, num_labels

def get_modernbert(task, num_labels = 2):
    """
    Load the pretrained ModernBERT base for the task

    Args:
        task : Either "mlm" for pretraining or "cls" for finetuning.
        num_labels: the number of classes (needed for the "cls" task, ignored for the "mlm" task)

    """
    if task == "mlm":
        model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base", device_map="auto")
    elif task == "cls":
        model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", device_map="auto", num_labels=num_labels) 
        model.config.problem_type= "multi_label_classification"
    else:
        raise ValueError(f"invalid argument for task passed: {task}")

    return model

def preprocess_labels(batch):
    """
    Helper function to split a string of concatenated labels into a list of labels
    """
    return {
        # Split each string in the batch
        'labels': [labels_str.split(',') for labels_str in batch['labels']]
    }


def save_class_statistics(output_file: str):
    """
    Calculates:
      1. The total number of unique classes
      2. The total number of occurrences of each class
    and writes the results to a text file.

    Args:
        output_file : path to output file
    """
    dataset = load_dataset("MalavP/USPTO-3M", split="train").rename_column("cpc_ids", "labels")

    dataset = dataset.map(
        preprocess_labels,
        batched=True,  # Required for multiprocessing
        batch_size=1000,  # Adjust based on memory
        num_proc=6  # For 8 CPUs, leave 2 cores free
    )
    all_labels = [label for sublist in dataset["labels"] for label in sublist]
    counts = Counter(all_labels)
    total_unique = len(counts)
    
    stats = {"total_unique": total_unique, "counts": dict(counts)}
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Class statistics saved to {output_file}")

def get_cls_transform(class_stats_file: str, tokenizer):
    """
    Get the transform required during finetuning
    
    Args:
        class_stats_file: path to the statistics of the dataset
        tokenizer : a HF tokenizer object

    Returns:
        A tuple (callable, int) of the transform function and the number of classes in the finetuning task 
    """

    with open(class_stats_file, 'r') as f:
        stats = json.load(f)
    label_encoder = LabelEncoder().fit(list(stats["counts"].keys()))
    num_labels = len(label_encoder.classes_)

    def transform(example):
        bsize = len(example["cpc_ids"])
        # tokenize text
        example["text"] = [text for text in example["text"] if text is not None and text.strip() != ""]
        tokenized = tokenizer(example['text'], truncation=True, padding=True, max_length=1024)

        # process labels
        unprocessed_labels = [labels_str.split(',') for labels_str in example['cpc_ids']]
        indices_list = [label_encoder.transform(x) for x in unprocessed_labels]
        labels = np.zeros(shape=(bsize,num_labels), dtype=float)
        for i, indices in enumerate(indices_list):
            labels[i, indices] = 1.0


        return {**tokenized, "labels" : labels}
    
    return transform, num_labels

def get_sorted_class_names(class_stats_file: str) -> list[str]:
    """
    Loads class statistics from a JSON file and returns a list of
    class names sorted alphabetically.

    Args:
        class_stats_file: Path to the JSON file containing class statistics
                          (expected to have a "counts" key with class names as keys).

    Returns:
        A list of unique class names, sorted alphabetically.
    """
    print(f"Loading class statistics from: {class_stats_file}")
    with open(class_stats_file, 'r') as f:
        stats = json.load(f)
    class_names = list(stats["counts"].keys())
    sorted_class_names = sorted(class_names)
    print(f"Loaded and sorted {len(sorted_class_names)} class names.")
    return sorted_class_names


if __name__ == "__main__":
    pass