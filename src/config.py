import os

class Config:
    # Paths
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    PAIRS_CSV = "data/pairs.csv"
    MODEL_SAVE_DIR = "models/saved_models"
    
    # Data preparation
    TRAIN_SPLIT = 0.8
    RANDOM_SEED = 42
    
    # Image processing
    IMG_SIZE = (224, 224)
    
    # Model architecture
    EMBEDDING_DIM = 512
    MLP_HIDDEN_DIM = 256
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 0.00001
    NUM_EPOCHS = 100
    DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    
    # Evaluation
    SIMILARITY_THRESHOLD = 0.5
    
    # Create directories
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)