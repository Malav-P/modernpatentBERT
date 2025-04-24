import os
import glob
from datasets import load_dataset
from huggingface_hub import HfApi  # Assuming logged in via CLI

CSV_DIR = "./bq_test_sets"
HF_USERNAME = "0zo"
REPO_NAME_PREFIX = "google_patentsview_claims"  # <--- CHANGE THIS or set to ""
PRIVATE_DATASET = False

print("--- Finding CSV Files ---")
csv_files = glob.glob(os.path.join(CSV_DIR, "*_test_set.csv"))
print(f"Found: {csv_files}")

if not csv_files:
    exit("No CSV files found.")

for csv_path in csv_files:
    base_name = os.path.basename(csv_path).replace("_test_set.csv", "")
    print(f"\n--- Processing: {base_name} ---")

    repo_suffix = f"_{base_name}" if base_name else ""
    repo_id = f"{HF_USERNAME}/{REPO_NAME_PREFIX}{repo_suffix}"
    print(f"Target Repo ID: {repo_id}")

    raw_dataset = load_dataset("csv", data_files=csv_path, split="train")
    print(f"Loaded {len(raw_dataset)} rows.")

    print(f"Pushing dataset...")
    raw_dataset.push_to_hub(repo_id=repo_id, private=PRIVATE_DATASET)
    print(f"Pushed to: https://huggingface.co/datasets/{repo_id}")

print("\n--- Script Finished ---")
