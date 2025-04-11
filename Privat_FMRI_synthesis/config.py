import torch
import os

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reproducibility
SEED = 42

# General
NUM_CLASSES = 5
LATENT_SHAPE = (8, 8, 8)  # Shape from the autoencoder's latent space

# Training
BATCH_SIZE = 8
EPOCHS_AUTOENCODER = 50
EPOCHS_DIFFUSION = 100
EPOCHS_SKIP = 50
EPOCHS_FINETUNE = 20
LEARNING_RATE = 5e-4
PATIENCE = 10
NUM_WORKERS = 16
PREFETCH_FACTOR = 4

# Fine-tuning
W_VAR = 1.0 #
TARGET_VARIANCE = 0.01

# Loss Weights
W_MSE = 0.8
W_L1 = 0.1
W_SSIM = 0.1

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_New")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


DIFFUSION_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "latent_diffusion.pth")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_autoencoder.pth") # Best autoencoder model
SKIP_PREDICTOR_BEST = os.path.join(CHECKPOINT_DIR, "skip_predictor_best.pth")
LOG_FILE = os.path.join(CHECKPOINT_DIR, "diffusion_training_log.txt")
LOSS_PLOT_PATH = os.path.join(CHECKPOINT_DIR, "loss_plot.png")

# Dataset paths
TRAIN_CSV = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/train.csv'
VAL_CSV = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/val.csv'
TEST_CSV = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/test.csv'
DATA_DIR = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri'

# Logging
LOG_FILE = "autoencoder_training_log.csv"
OUTPUT_DIR = r'/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/test_analysis'
BASE_LOG_DIR = r'/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/logs'

# Visualazation
PERMUTE_ORDER = (2, 0, 1)