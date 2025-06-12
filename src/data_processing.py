"""
Functions for data loading, feature engineering, and processing.
"""
import os
import re
import zipfile
from io import BytesIO
from urllib.parse import urlparse
import glob
import pickle

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import sys
import src.config as cfg

# --- GloVe Embedding Loading ---

def load_glove_embeddings(glove_file=cfg.GLOVE_FILE):
    """
    Loads GloVe embeddings from file, using a cached version if available.
    Downloads the raw file if it doesn't exist.
    """
    glove_dir = os.path.dirname(glove_file)
    cached_word_to_idx_path = os.path.join(glove_dir, "word_to_idx.pkl")
    cached_embeddings_path = os.path.join(glove_dir, "embeddings.npy")

    # Try to load from cache first
    if os.path.exists(cached_word_to_idx_path) and os.path.exists(cached_embeddings_path):
        print("Loading cached GloVe embeddings...")
        with open(cached_word_to_idx_path, 'rb') as f:
            word_to_idx = pickle.load(f)
        embeddings = np.load(cached_embeddings_path)
        print(f"✅ Cached GloVe embeddings loaded: {len(word_to_idx):,} words, {embeddings.shape[1]} dim.")
        return word_to_idx, embeddings

    # If cache not found, load from raw file
    if not os.path.exists(glove_file):
        print("Downloading GloVe 200d embeddings...")
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        response = requests.get(url, stream=True)
        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            # Note: The file inside the zip is 'glove.6B.200d.txt'
            with zip_file.open("glove.6B.200d.txt") as source:
                os.makedirs(glove_dir, exist_ok=True)
                with open(glove_file, "wb") as target:
                    target.write(source.read())
        print("✅ Download complete!")

    print("Loading GloVe word vectors from raw file...")
    word_to_idx = {}
    embeddings_list = []
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            word_to_idx[word] = len(embeddings_list)
            embeddings_list.append(vector)
    
    embeddings = np.array(embeddings_list)
    print(f"✅ GloVe embeddings loaded: {len(word_to_idx):,} words, {embeddings.shape[1]} dim.")
    
    # Save to cache for future use
    print("Caching GloVe embeddings for faster loading...")
    with open(cached_word_to_idx_path, 'wb') as f:
        pickle.dump(word_to_idx, f)
    np.save(cached_embeddings_path, embeddings)
    print("✅ GloVe embeddings cached.")

    return word_to_idx, embeddings

# --- Feature Engineering Functions ---

def clean_text(text):
    """Cleans text for embedding lookup."""
    text = text.lower()
    text = re.sub(r'-', ' ', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    return text.split()

def title_to_embedding(title, word_to_index, embeddings, method='average'):
    """Converts title to an embedding using GloVe."""
    words = clean_text(title)
    word_embeddings = [embeddings[word_to_index[word]] for word in words if word in word_to_index]
    
    if not word_embeddings:
        return np.zeros(embeddings.shape[1])
    
    if method == 'average':
        return np.mean(word_embeddings, axis=0)
    elif method == 'sum':
        return np.sum(word_embeddings, axis=0)
    elif method == 'max':
        return np.max(word_embeddings, axis=0)
    return np.mean(word_embeddings, axis=0)

def extract_domains(url):
    """Extracts and cleans domain from a URL."""
    if pd.isna(url) or url == '':
        return 'self_post'
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        domain = urlparse(url).netloc.lower()
        prefixes = ['www.', 'm.', 'mobile.', 'old.']
        for prefix in prefixes:
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
                break
        return domain.split(':')[0].rstrip('.') or 'parse_error'
    except:
        return 'parse_error'

# --- Data Augmentation ---

def simple_text_augment(title):
    """Simple text variations using basic substitutions."""
    simple_synonyms = {
        'new': ['latest', 'fresh'], 'best': ['top', 'great'], 'simple': ['easy', 'basic'],
        'fast': ['quick', 'rapid'], 'build': ['create', 'make'], 'use': ['utilize', 'leverage'],
        'good': ['solid', 'effective'], 'free': ['open', 'gratis'], 'guide': ['tutorial', 'howto'],
        'tool': ['utility', 'app'], 'way': ['method', 'approach'], 'learn': ['master', 'understand'],
    }
    words = title.split()
    if len(words) > 2 and np.random.random() < 0.5:
        for i, word in enumerate(words):
            if word.lower() in simple_synonyms and np.random.random() < 0.3:
                words[i] = np.random.choice(simple_synonyms[word.lower()])
                break
    return ' '.join(words)

# --- Main Data Preparation Pipeline ---

def prepare_features(cfg):
    """
    Loads data, performs feature engineering, augmentation, and prepares
    data for the model.
    """
    print("--- Starting Data Preparation ---")
    
    # Load raw data
    df = pd.read_parquet(cfg.DATA_PATH)
    df_sample = df.sample(n=cfg.NUMBER_OF_SAMPLES, random_state=42).copy()
    
    # Filter data
    df_filtered = df_sample[
        (df_sample['score'] >= cfg.MINIMUM_SCORE) &
        (df_sample['score'] <= cfg.MAXIMUM_SCORE) &
        (df_sample['title'].notna()) &
        (df_sample['by'].notna()) &
        (df_sample['time'].notna())
    ].copy()
    
    print(f"Filtered to {len(df_filtered):,} samples.")

    # Feature Engineering
    df_filtered['score_log'] = np.log1p(df_filtered['score'])
    df_filtered['timestamp'] = pd.to_datetime(df_filtered['time'], unit='s')
    df_filtered['hour_of_day'] = df_filtered['timestamp'].dt.hour
    df_filtered['day_of_week'] = df_filtered['timestamp'].dt.dayofweek
    df_filtered['time_of_day_sin'] = np.sin(2 * np.pi * df_filtered['hour_of_day'] / 24)
    df_filtered['time_of_day_cos'] = np.cos(2 * np.pi * df_filtered['hour_of_day'] / 24)
    df_filtered['day_of_week_sin'] = np.sin(2 * np.pi * df_filtered['day_of_week'] / 7)
    df_filtered['day_of_week_cos'] = np.cos(2 * np.pi * df_filtered['day_of_week'] / 7)
    df_filtered['domain'] = df_filtered['url'].apply(extract_domains)
    df_filtered['word_count'] = df_filtered['title'].apply(lambda x: len(x.strip().split()))
    
    print("✅ Base features engineered.")

    # Data Augmentation
    print("🚀 Performing data augmentation...")
    log_scores = df_filtered['score_log'].values
    log_score_bins = np.linspace(log_scores.min(), log_scores.max(), 20)
    df_filtered['log_bin'] = pd.cut(df_filtered['score_log'], bins=log_score_bins, include_lowest=True)
    bin_counts = df_filtered['log_bin'].value_counts().sort_index()

    augmented_data = [df_filtered]
    total_added = 0
    np.random.seed(42)

    for bin_label, current_count in bin_counts.items():
        if current_count >= cfg.MIN_TRESHOLD or total_added >= cfg.TOTAL_BUDGET:
            continue
        
        needed = min(cfg.MIN_TRESHOLD - current_count, cfg.MAX_AUGMENT_PER_BIN, cfg.TOTAL_BUDGET - total_added)
        if needed <= 0: continue

        bin_data = df_filtered[df_filtered['log_bin'] == bin_label]
        augmented_rows = []
        for _ in range(needed):
            base_row = bin_data.sample(1).iloc[0].copy()
            base_row['title'] = simple_text_augment(base_row['title'])
            # Add small noise to time features
            hour_noise = np.random.normal(0, 0.5)
            new_hour = (base_row['hour_of_day'] + hour_noise) % 24
            base_row['time_of_day_sin'] = np.sin(2 * np.pi * new_hour / 24)
            base_row['time_of_day_cos'] = np.cos(2 * np.pi * new_hour / 24)
            base_row['word_count'] = len(base_row['title'].strip().split())
            augmented_rows.append(base_row)
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            augmented_data.append(augmented_df)
            total_added += len(augmented_rows)

    df_augmented = pd.concat(augmented_data, ignore_index=True).drop('log_bin', axis=1)
    print(f"✅ Augmentation complete. Total samples: {len(df_augmented):,}")

    # Load GloVe and create title embeddings
    word_to_idx, embeddings = load_glove_embeddings(cfg.GLOVE_FILE)
    
    # Process title embeddings in batches and save to disk
    print("Creating title embeddings in batches...")
    batch_size = 10000
    temp_dir = "temp_embeddings"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create memory-mapped array for final embeddings
    total_samples = len(df_augmented)
    embedding_dim = embeddings.shape[1]  # Should be 200 for GloVe
    X_title_embeddings = np.memmap(
        f"{temp_dir}/embeddings.mmap",
        dtype='float32',
        mode='w+',
        shape=(total_samples, embedding_dim)
    )
    
    # Process and save embeddings in batches
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        batch_titles = df_augmented['title'].iloc[i:end_idx]
        batch_embeddings = [title_to_embedding(title, word_to_idx, embeddings) for title in batch_titles]
        # Write directly to memory-mapped array
        X_title_embeddings[i:end_idx] = batch_embeddings
        # Clear memory
        del batch_embeddings
        if i % 100000 == 0:
            print(f"Processed {i:,} titles...")
    
    # Flush changes to disk
    X_title_embeddings.flush()
    
    # Update word counts after augmentation
    df_augmented['word_count'] = [len(title.strip().split()) for title in df_augmented['title']]
    
    # Clean up of temp files is now handled in the main training script
    
    # Prepare other features
    print("Preparing numerical features...")
    numerical_features = df_augmented[['word_count', 'time_of_day_sin', 'time_of_day_cos', 
                                     'day_of_week_sin', 'day_of_week_cos']].values
    # SCALER IS NOW APPLIED IN THE TRAINING SCRIPT AFTER SPLITTING
    # scaler = StandardScaler()
    # X_numerical_scaled = scaler.fit_transform(numerical_features)

    # Prepare domain and user encodings
    print("Preparing domain and user encodings...")
    domain_counts = df_augmented['domain'].value_counts()
    top_domains = domain_counts.head(cfg.NUM_DOMAINS).index
    df_augmented['domain_mapped'] = df_augmented['domain'].apply(lambda x: x if x in top_domains else 'OTHER')
    domain_encoder = LabelEncoder().fit(df_augmented['domain_mapped'])
    X_domain_ids = domain_encoder.transform(df_augmented['domain_mapped'])

    user_counts = df_augmented['by'].value_counts()
    top_users = user_counts.head(cfg.NUM_USERS).index
    df_augmented['user_mapped'] = df_augmented['by'].apply(lambda x: x if x in top_users else 'OTHER')
    user_encoder = LabelEncoder().fit(df_augmented['user_mapped'])
    X_user_ids = user_encoder.transform(df_augmented['user_mapped'])
    
    y = df_augmented['score_log'].values
    
    # Clear memory
    del df_filtered
    del df_augmented
    
    print("--- Data Preparation Finished ---")

    return {
        "X_title_embeddings": X_title_embeddings,
        "X_numerical": numerical_features,
        "X_domain_ids": X_domain_ids,
        "X_user_ids": X_user_ids,
        "y": y,
        "domain_encoder": domain_encoder,
        "user_encoder": user_encoder,
        # "scaler": scaler,
        "n_domains": len(domain_encoder.classes_),
        "n_users": len(user_encoder.classes_),
    }

def create_data_loader(data_dict, indices, batch_size, shuffle=True):
    """Creates a PyTorch DataLoader."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # X_title_embeddings is a memory-mapped numpy array, which can be used directly
    dataset = TensorDataset(
        torch.from_numpy(data_dict["X_title_embeddings"][indices]).float().to(device),
        torch.FloatTensor(data_dict["X_numerical_scaled"][indices]).to(device),
        torch.LongTensor(data_dict["X_domain_ids"][indices]).to(device),
        torch.LongTensor(data_dict["X_user_ids"][indices]).to(device),
        torch.FloatTensor(data_dict["y"][indices]).to(device)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 