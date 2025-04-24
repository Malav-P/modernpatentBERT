import os
import pandas as pd
from google.cloud import bigquery
from tqdm import tqdm

# --- Configuration ---
PROJECT_ID = "test-project-embeddings"
SAVE_DIR = "./bq_test_sets"
SEED = 42
NUM_SAMPLES = 150000

test_set_definitions = {
    "2016": ("2016-01-01", "2016-12-31"),
    "2017-B": ("2017-01-01", "2017-08-31"),
}

# --- Ensure save directory exists ---
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Base SQL Query Template ---
BASE_SQL_QUERY = """
SELECT STRING_AGG(distinct t2.group_id ORDER BY t2.group_id) AS cpc_ids,
       t1.id, t1.date, t3.text
FROM `patents-public-data.patentsview.patent` t1,
     `patents-public-data.patentsview.cpc_current` t2,
     `patents-public-data.patentsview.claim` t3
WHERE t1.id = t2.patent_id
  AND t1.id = t3.patent_id
  AND t1.date >= '{start_date}'
  AND t1.date <= '{end_date}'
  AND t3.sequence = '1'
  AND t1.type = 'utility'
GROUP BY t1.id, t1.date, t3.text
"""

# --- Initialize BigQuery Client ---
# Ensure 'google-cloud-bigquery[storage]' is installed for to_dataframe_iterable
# pip install "google-cloud-bigquery[storage]"
client = bigquery.Client(project=PROJECT_ID)
print(f"BigQuery client initialized for project: {PROJECT_ID}")


print("\n--- Querying BigQuery and Creating Test Sets ---")

for name, (start_str, end_str) in test_set_definitions.items():
    print(f"\nProcessing test set: {name} ({start_str} to {end_str})")

    sql = BASE_SQL_QUERY.format(start_date=start_str, end_date=end_str)
    temp_output_filename = os.path.join(SAVE_DIR, f"{name}_temp_full.csv")

    print(
        f"Running query for {name} and writing chunks to temp file: {temp_output_filename}"
    )
    query_job = client.query(sql)

    results_iterator = query_job.result()  # Wait for job completion
    total_rows_expected = results_iterator.total_rows
    print(f"Expected total rows: {total_rows_expected}")

    dataframe_iterator = results_iterator.to_dataframe_iterable()

    first_write = True
    rows_written = 0

    with tqdm(total=total_rows_expected, desc=f"Writing Chunks {name}") as pbar:
        for df_chunk in dataframe_iterator:
            df_chunk.to_csv(
                temp_output_filename, mode="a", index=False, header=first_write
            )
            first_write = False
            rows_written += len(df_chunk)
            pbar.update(len(df_chunk))

    print(
        f"Finished writing {rows_written} rows to temporary file: {temp_output_filename}"
    )
    num_fetched = rows_written

    if num_fetched == 0:
        print(f"Skipping {name}: No patents found or written.")
        if os.path.exists(temp_output_filename):
            os.remove(temp_output_filename)
        continue

    # --- SAMPLING FROM THE SAVED CSV ---
    final_output_filename = os.path.join(SAVE_DIR, f"{name}_test_set.csv")
    print(f"Sampling or shuffling from temp file: {temp_output_filename}")

    # WARNING: Reading the whole CSV can still cause OOM for very large files.
    full_df = pd.read_csv(temp_output_filename)
    actual_fetched = len(full_df)  # Use len of DataFrame after reading

    if actual_fetched > NUM_SAMPLES:
        print(f"Sampling {NUM_SAMPLES} patents using seed {SEED}...")
        final_df = full_df.sample(n=NUM_SAMPLES, random_state=SEED)
        final_df.to_csv(final_output_filename, index=False)
        print(
            f"Saved final sampled dataset ({len(final_df)} samples) to: {final_output_filename}"
        )

    elif (
        actual_fetched <= NUM_SAMPLES
    ):  # Includes case where exactly NUM_SAMPLES fetched
        if actual_fetched < NUM_SAMPLES:
            print(
                f"Warning: Only {actual_fetched} patents found. Using all available for {name}."
            )
        else:
            print(f"Using all {actual_fetched} fetched patents.")

        final_df = full_df.sample(frac=1, random_state=SEED)  # Shuffle
        final_df.to_csv(final_output_filename, index=False)
        print(
            f"Saved final dataset ({len(final_df)} samples) to: {final_output_filename}"
        )

    # Clean up the temporary full file
    if os.path.exists(temp_output_filename):
        os.remove(temp_output_filename)


print("\n--- Script Finished ---")
print(f"Test subsets saved as CSV files in directory: {SAVE_DIR}")
