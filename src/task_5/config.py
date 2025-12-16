import torch
import os

# =============== Google Drive Setup ===============
# Base path to the project in Google Drive
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/PointNet++"

os.makedirs(os.path.join(DRIVE_PROJECT_ROOT, "data"), exist_ok=True)
os.makedirs(os.path.join(DRIVE_PROJECT_ROOT, "models"), exist_ok=True)
os.makedirs(os.path.join(DRIVE_PROJECT_ROOT, "outputs"), exist_ok=True)
# ================================================

# Dataset parameters
NUM_POINTS = 4096
NUM_CLASSES = 13
INPUT_CHANNELS = 6

# Training parameters
BATCH_SIZE = 8   # 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = os.path.join(DRIVE_PROJECT_ROOT, "data", "s3dis_segmentation.h5")
MODEL_PATH = os.path.join(DRIVE_PROJECT_ROOT, "models", "best_pointnet2_seg.pth")
OUTPUT_DIR = os.path.join(DRIVE_PROJECT_ROOT, "outputs")

# Model architecture parameters
SA_PARAMS = [
    {'npoint': 1024, 'radius': 0.1, 'nsample': 32, 'mlp': [32, 32, 64]},
    {'npoint': 256, 'radius': 0.2, 'nsample': 32, 'mlp': [64, 64, 128]},
    {'npoint': 64, 'radius': 0.4, 'nsample': 32, 'mlp': [128, 128, 256]},
    {'npoint': 16, 'radius': 0.8, 'nsample': 32, 'mlp': [256, 256, 512]}
]
FP_PARAMS = [
    {'in_channel': 512 + 256, 'mlp': [256, 256]},
    {'in_channel': 256 + 128, 'mlp': [256, 256]},
    {'in_channel': 256 + 64, 'mlp': [256, 128]},
    {'in_channel': 128 + INPUT_CHANNELS, 'mlp': [128, 128, 128]}
]
HEAD_MLP = [128, 128, 128]

# Data loading
NUM_WORKERS = 2

# Augmentation parameters
AUGMENT_ROTATION = True
AUGMENT_SCALE = (0.8, 1.2)
AUGMENT_JITTER = 0.02
AUGMENT_DROPOUT = (0.9, 1.0)

# S3DIS classes
S3DIS_CLASSES = [
    'ceiling', 'floor', 'wall', 'beam', 'column',
    'window', 'door', 'table', 'chair', 'sofa',
    'bookcase', 'board', 'clutter'
]
