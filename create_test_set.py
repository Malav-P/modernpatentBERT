import datasets
from datetime import datetime
from functools import partial
import os

SEED = 42  # Seed for reproducible shuffling/sampling
NUM_SAMPLES = 150000 # Target number of samples for each test set
DATASET_NAME = "0zo/google_patentsview_claims_2016"
SAVE_DIR = "./uspto_3m_test_sets" # Local directory to save the subsets

os.makedirs(SAVE_DIR, exist_ok=True)

patents_dataset = datasets.load_dataset(DATASET_NAME, split='train')

test_set_definitions = {
    "2015-B": ("2015-01-01", "2015-12-31"),
    "2016": ("2016-01-01", "2016-12-31"),
    "2017": ("2017-01-01", "2017-08-31"), # Jan to Aug for 2017
}

def filter_by_date_range(example, start_date_str, end_date_str):
    try:
        # Extract the date part (assuming space separates date and time)
        patent_date_str = example['date'].split(" ")[0]
        patent_date = datetime.strptime(patent_date_str, '%Y-%m-%d').date()
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        return start_date <= patent_date <= end_date
    except (ValueError, TypeError, AttributeError, KeyError):
        return False # Exclude problematic records

print("\n--- Creating and Saving Test Sets ---")

for name, (start_str, end_str) in test_set_definitions.items():
    print(f"\nProcessing test set: {name} ({start_str} to {end_str})")

    print("Filtering patents by date...")
    filter_func = partial(filter_by_date_range, start_date_str=start_str, end_date_str=end_str)
    filtered_dataset = patents_dataset.filter(filter_func, num_proc=os.cpu_count() // 2 or 1)
    num_found = len(filtered_dataset)

    # 2. Check if any patents were found
    if num_found == 0:
        print(f"Skipping {name}: No patents found in this range.")
        continue

    # 3. Determine actual number of samples to select
    actual_samples_to_select = min(NUM_SAMPLES, num_found)
    if num_found < NUM_SAMPLES:
         print(f"Warning: Only {num_found} patents found. Using all available for {name}.")
    else:
        print(f"Selecting {NUM_SAMPLES} samples for {name}.")


    # 4. Shuffle reproducibly & Select the desired number
    print(f"Shuffling with seed {SEED} and selecting {actual_samples_to_select} samples...")
    shuffled_filtered_dataset = filtered_dataset.shuffle(seed=SEED)
    final_test_set = shuffled_filtered_dataset.select(range(actual_samples_to_select))

    # 5. Save the subset locally
    subset_path = os.path.join(SAVE_DIR, name)
    print(f"Saving {name} dataset ({len(final_test_set)} samples) to: {subset_path}")
    try:
        final_test_set.save_to_disk(subset_path)
        print(f"Successfully saved {name}.")
    except Exception as e:
        print(f"Error saving dataset {name} to {subset_path}: {e}")


print("\n--- Script Finished ---")
print(f"Test subsets saved in directory: {SAVE_DIR}")

################################### If we want to create 2016 and 2017 test sets, need to use big query like this ###################################
# import os
# import pandas as pd
# from google.cloud import bigquery
# from datetime import datetime

# # --- Configuration ---
# PROJECT_ID = "your-gcp-project-id"  # <--- CHANGE THIS to your Google Cloud Project ID
# SAVE_DIR = "./bq_test_sets"         # Directory to save the output CSV files
# SEED = 42                          # Seed for reproducible sampling
# NUM_SAMPLES = 150000                 # Target number of samples per test set

# # --- Test Set Definitions (Date Ranges) ---
# test_set_definitions = {
#     "2015-B": ("2015-01-01", "2015-12-31"),
#     "2016": ("2016-01-01", "2016-12-31"),
#     "2017": ("2017-01-01", "2017-08-31"), # Jan to Aug for 2017
# }

# # --- Ensure save directory exists ---
# os.makedirs(SAVE_DIR, exist_ok=True)

# # --- Base SQL Query Template ---
# # NOTE: Table names (e.g., `patents-public-data.patents.publications`)
# # might need verification if the public dataset schema has changed.
# # `kind_code = 'A'` is a common indicator for US utility patents grants,
# # but you might need to adjust based on specific requirements.
# BASE_SQL_QUERY = """
# SELECT
#     t1.publication_number,
#     t1.filing_date,
#     t3.text AS first_claim_text,
#     -- Aggregate distinct 4-character CPC subclasses
#     STRING_AGG(DISTINCT SUBSTR(t2.code, 1, 4), ',') AS cpc_subclasses
# FROM
#     `patents-public-data.patents.publications` AS t1
# JOIN
#     `patents-public-data.patents.cpc` AS t2 ON t1.publication_number = t2.publication_number
# JOIN
#     `patents-public-data.patents.claims` AS t3 ON t1.publication_number = t3.publication_number
# WHERE
#     t1.country_code = 'US'
#     AND t1.kind_code = 'A' -- Filter for common US utility patent grant indicator
#     AND t3.sequence = 1    -- Filter for the first claim only
#     AND DATE(t1.filing_date) BETWEEN DATE('{start_date}') AND DATE('{end_date}')
# GROUP BY
#     t1.publication_number, t1.filing_date, t3.text
# ORDER BY
#     -- Order randomly within BigQuery before fetching.
#     -- Note: Can be computationally intensive on very large result sets.
#     -- An alternative is fetching all results and sampling in pandas (see commented code below)
#     RAND()
# LIMIT {limit} -- Fetch slightly more than needed to ensure enough for sampling if RAND() isn't perfectly uniform
# """

# # Alternative Strategy Limit (fetch more, sample in pandas): Fetch all if preferred
# # FETCH_LIMIT = NUM_SAMPLES * 2 # Fetch more than needed for better pandas sampling
# FETCH_LIMIT = NUM_SAMPLES + 5000 # Fetch slightly more in case RAND() is uneven

# # --- Initialize BigQuery Client ---
# try:
#     client = bigquery.Client(project=PROJECT_ID)
#     print(f"BigQuery client initialized for project: {PROJECT_ID}")
# except Exception as e:
#     print(f"Error initializing BigQuery client: {e}")
#     print("Please check your project ID and authentication.")
#     exit()

# # --- Create and Save Test Sets ---
# print("\n--- Querying BigQuery and Creating Test Sets ---")
# print("WARNING: BigQuery queries incur costs.")

# for name, (start_str, end_str) in test_set_definitions.items():
#     print(f"\nProcessing test set: {name} ({start_str} to {end_str})")

#     # 1. Format the SQL Query
#     sql = BASE_SQL_QUERY.format(start_date=start_str, end_date=end_str, limit=FETCH_LIMIT)
#     # print("Executing SQL:\n", sql[:500] + "...") # Uncomment to view query

#     # 2. Execute the Query and Fetch Results to Pandas DataFrame
#     try:
#         print(f"Running query for {name} (this may take time)...")
#         query_job = client.query(sql)
#         # Use .to_dataframe() which handles pagination
#         # Use db_dtypes=True for newer pandas versions to use appropriate types
#         results_df = query_job.to_dataframe(create_bqstorage_client=True, progress_bar_type='tqdm')
#         num_fetched = len(results_df)
#         print(f"Fetched {num_fetched} candidate patents for {name}.")

#     except Exception as e:
#         print(f"Error running BigQuery query for {name}: {e}")
#         print("Check your SQL query, permissions, and project billing status.")
#         continue # Skip to the next test set

#     # 3. Check if any patents were found
#     if num_fetched == 0:
#         print(f"Skipping {name}: No patents found in this range matching criteria.")
#         continue

#     # 4. Sample the DataFrame (if needed)
#     if num_fetched > NUM_SAMPLES:
#         print(f"Sampling {NUM_SAMPLES} patents using seed {SEED}...")
#         # Use random_state for reproducibility
#         final_df = results_df.sample(n=NUM_SAMPLES, random_state=SEED)
#     elif num_fetched < NUM_SAMPLES:
#          print(f"Warning: Only {num_fetched} patents found. Using all available for {name}.")
#          # Optionally shuffle even if using all, for consistency
#          final_df = results_df.sample(frac=1, random_state=SEED)
#     else:
#         # Exactly NUM_SAMPLES fetched
#         final_df = results_df # No sampling needed, already limited by BQ


#     # 5. Save the subset locally as CSV
#     output_filename = os.path.join(SAVE_DIR, f"{name}_test_set.csv")
#     print(f"Saving {name} dataset ({len(final_df)} samples) to: {output_filename}")
#     try:
#         final_df.to_csv(output_filename, index=False)
#         print(f"Successfully saved {name}.")
#     except Exception as e:
#         print(f"Error saving DataFrame to CSV for {name}: {e}")

# print("\n--- Script Finished ---")
# print(f"Test subsets saved as CSV files in directory: {SAVE_DIR}")