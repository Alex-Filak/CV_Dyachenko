import torch

# Dataset settings
DATASET_NAME = "ModelNet10"
NUM_POINTS = 1024
DATA_PATH = "data/modelnet10_1024.h5"
MODELNET_RAW_PATH = "ModelNet10"

# Model settings
NUM_CLASSES = 10
FEATURE_TRANSFORM = True

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 50
TRAIN_FRAC = 0.8
NUM_WORKERS = 0
LEARNING_RATE = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Output settings
OUTPUT_DIR = "data"
MODEL_PATH = f"{OUTPUT_DIR}/pointnet_{DATASET_NAME.lower()}.pth"

# Visualization
CLASS_NAMES = [
    'bathtub', 'bed', 'chair', 'desk', 'dresser',
    'monitor', 'night_stand', 'sofa', 'table', 'toilet'
]
