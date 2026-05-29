import os
import sys
import json
import zipfile
import requests
import pandas as pd
from tqdm import tqdm

def download_file(url, dest_path):
    """
    Downloads a file from the given URL to the specified destination path.
    Shows a progress bar using tqdm.

    Args:
        url (str): URL to download from.
        dest_path (str): Destination file path.
    """
    if os.path.exists(dest_path):
        print(f"File {dest_path} already exists. Skipping download.")
        return

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except requests.exceptions.RequestException as e:
        status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
        raise RuntimeError(f"Failed to download {url}. Status code: {status_code}. Error: {e}")

def extract_zip(zip_path, extract_to):
    """
    Extracts a zip file to the specified directory.

    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory to extract to.
    """
    if os.path.exists(extract_to) and os.path.isdir(extract_to) and len(os.listdir(extract_to)) > 0:
        print(f"Directory {extract_to} already exists and is not empty. Skipping extraction.")
        return

    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(extract_to))
    print("Extraction complete.")

def process_data(raw_data_dir, processed_data_dir):
    """
    Processes the raw MovieLens dataset, creates ID mappings, and splits into train/test sets.

    Args:
        raw_data_dir (str): Directory containing the raw extracted dataset.
        processed_data_dir (str): Directory to save processed files.
    """
    ratings_file = os.path.join(raw_data_dir, 'ratings.dat')
    movies_file = os.path.join(raw_data_dir, 'movies.dat')

    if not os.path.exists(ratings_file) or not os.path.exists(movies_file):
        raise FileNotFoundError("Raw data files not found. Ensure extraction was successful.")

    print("Loading raw datasets...")
    ratings_df = pd.read_csv(
        ratings_file,
        sep='::',
        engine='python',
        header=None,
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    movies_df = pd.read_csv(
        movies_file,
        sep='::',
        engine='python',
        header=None,
        names=['movie_id', 'title', 'genres'],
        encoding='latin-1'
    )

    # Validate columns
    expected_rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    expected_movie_cols = ['movie_id', 'title', 'genres']
    
    if list(ratings_df.columns) != expected_rating_cols:
        raise ValueError(f"Expected rating columns {expected_rating_cols}, found {list(ratings_df.columns)}")
    if list(movies_df.columns) != expected_movie_cols:
        raise ValueError(f"Expected movie columns {expected_movie_cols}, found {list(movies_df.columns)}")

    print("Creating ID mappings...")
    unique_users = ratings_df['user_id'].unique()
    unique_items = ratings_df['movie_id'].unique()

    user_map = {str(uid): int(idx) for idx, uid in enumerate(unique_users)}
    item_map = {str(iid): int(idx) for idx, iid in enumerate(unique_items)}
    id_to_item = {str(idx): int(iid) for idx, iid in enumerate(unique_items)}
    
    # Create item_names mapping
    movie_id_to_title = dict(zip(movies_df['movie_id'], movies_df['title']))
    item_names = {str(idx): str(movie_id_to_title.get(iid, f"Unknown Item {iid}")) 
                  for idx, iid in enumerate(unique_items)}

    # Apply mappings to ratings dataframe
    print("Applying ID mappings...")
    ratings_df['user_idx'] = ratings_df['user_id'].astype(str).map(user_map)
    ratings_df['item_idx'] = ratings_df['movie_id'].astype(str).map(item_map)

    # Save JSON mappings
    print("Saving ID mappings to JSON...")
    os.makedirs(processed_data_dir, exist_ok=True)
    with open(os.path.join(processed_data_dir, 'user_map.json'), 'w') as f:
        json.dump(user_map, f)
    with open(os.path.join(processed_data_dir, 'item_map.json'), 'w') as f:
        json.dump(item_map, f)
    with open(os.path.join(processed_data_dir, 'id_to_item.json'), 'w') as f:
        json.dump(id_to_item, f)
    with open(os.path.join(processed_data_dir, 'item_names.json'), 'w') as f:
        json.dump(item_names, f)

    print("Performing user-stratified train/test split...")
    # Sort by timestamp
    ratings_df = ratings_df.sort_values(by=['user_idx', 'timestamp'])
    
    # Group by user and split 80/20
    def split_user(group):
        n = len(group)
        test_size = max(1, int(n * 0.2))
        train = group.iloc[:-test_size]
        test = group.iloc[-test_size:]
        return pd.Series({'train': train, 'test': test})

    splits = ratings_df.groupby('user_idx', group_keys=False).apply(split_user, include_groups=False)
    train_df = pd.concat(splits['train'].tolist())
    test_df = pd.concat(splits['test'].tolist())

    print("Saving train and test sets...")
    cols_to_save = ['user_idx', 'item_idx', 'rating']
    train_df[cols_to_save].to_csv(os.path.join(processed_data_dir, 'train.csv'), index=False)
    test_df[cols_to_save].to_csv(os.path.join(processed_data_dir, 'test.csv'), index=False)

    # Print summary
    num_users = len(unique_users)
    num_items = len(unique_items)
    num_train = len(train_df)
    num_test = len(test_df)
    sparsity = 1.0 - ((num_train + num_test) / (num_users * num_items))

    print("\n--- Summary ---")
    print(f"Total Users: {num_users}")
    print(f"Total Items: {num_items}")
    print(f"Train Ratings: {num_train}")
    print(f"Test Ratings: {num_test}")
    print(f"Matrix Sparsity: {sparsity:.4%}")
    print("-----------------\n")

def main():
    """
    Main execution flow for data preparation.
    """
    raw_dir = os.getenv('RAW_DATA_PATH', './data/raw')
    processed_dir = os.getenv('PROCESSED_DATA_PATH', './data/processed')
    url = os.getenv('MOVIELENS_URL', 'https://files.grouplens.org/datasets/movielens/ml-1m.zip')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    zip_path = os.path.join(raw_dir, 'ml-1m.zip')
    extracted_dir = os.path.join(raw_dir, 'ml-1m')
    
    try:
        download_file(url, zip_path)
        extract_zip(zip_path, extracted_dir)
        process_data(extracted_dir, processed_dir)
    except Exception as e:
        print(f"Error during data preparation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
