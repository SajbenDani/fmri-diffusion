import torch

# Switch between CPU and GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Change manually if needed

# Reproducibility
SEED = 42

# Training settings
BATCH_SIZE = 8
NUM_CLASSES = 5 # Number of unique labels in the dataset
EPOCHS_AUTOENCODER = 50
EPOCHS_DIFFUSION = 100
LEARNING_RATE = 5e-4
PATIENCE = 10
NUM_WORKERS = 16
NUM_TIMESTEPS = 1000 # Number of diffusion steps


# Paths
CHECKPOINT_DIR = "checkpoints_New/"
AUTOENCODER_CHECKPOINT = CHECKPOINT_DIR + "autoencoder.pth"
DIFFUSION_CHECKPOINT = CHECKPOINT_DIR + "diffusion_model.pth"
DIFFUSION_LAST_CHECKPOINT = CHECKPOINT_DIR + "diffusion_model_last.pth"

TRAIN_CSV = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/train.csv'
VAL_CSV = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/val.csv'
TEST_CSV = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/test.csv'
DATA_DIR = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri'

# Logging
LOG_DIR = "C:/Users/sajbe/Documents/onLab/Privat_FMRI_synthesis/logs"

# Model-specific constants
SPATIAL_DIM = int(256 ** 0.5)  # Assumes latent space is 256 (diffusion model)
LATENT_DIM = 256


# Plotting
PERMUTE_ORDER = (2, 0, 1)  # For visualization: [D, H, W] -> [W, D, H]