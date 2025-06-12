"""
Configuration file for the HackerNews Score Prediction project.
"""

# Data Constants
NUMBER_OF_SAMPLES = 2000000
MINIMUM_SCORE = 5
MAXIMUM_SCORE = 1000
MIN_TRESHOLD = 10000
MAX_AUGMENT_PER_BIN = 15000
TOTAL_BUDGET = 100000
NUM_DOMAINS = 200
NUM_USERS = 1000

# Model Architecture Constants
DOMAIN_EMB_DIM = 60        # Domain embedding dimension
USER_EMB_DIM = 64          # User embedding dimension
TITLE_EMB_DIM = 200        # Title embedding dimension (from GloVe)
NUMERICAL_DIM = 5          # Number of numerical features (word_count, time_sin, time_cos, day_sin, day_cos)
HIDDEN_DIM = 128           # Hidden layer dimension
DROPOUT_RATE = 0.1         # Dropout rate

# Training Constants
BATCH_SIZE = 256           # Batch size for training
LEARNING_RATE = 1e-4       # Learning rate
WEIGHT_DECAY = 1e-4        # Weight decay for AdamW
NUM_EPOCHS = 100            # Maximum training epochs
PATIENCE = 100             # Early stopping patience
FACTOR = 0.5               # Learning rate reduction factor
VAL_SIZE = 0.25            # Validation set size
TEST_SIZE = 0.2            # Test set size

# File Paths
DATA_PATH = "data/hackernews_full_data.parquet"
GLOVE_FILE = "data/glove.6B.200d.txt"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = f"{ARTIFACTS_DIR}/best_combined_model.pth"
MODEL_CONFIG_PATH = f"{ARTIFACTS_DIR}/model_config.pkl"
DOMAIN_ENCODER_PATH = f"{ARTIFACTS_DIR}/domain_encoder.pkl"
USER_ENCODER_PATH = f"{ARTIFACTS_DIR}/user_encoder.pkl"
SCALER_PATH = f"{ARTIFACTS_DIR}/scaler.pkl" 