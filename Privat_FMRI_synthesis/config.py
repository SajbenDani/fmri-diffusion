import torch

# Switch between CPU and GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Change manually if needed

# Training settings
BATCH_SIZE = 12
EPOCHS_AUTOENCODER = 50
EPOCHS_DIFFUSION = 100
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "checkpoints_New/"

# Paths
AUTOENCODER_CHECKPOINT = CHECKPOINT_DIR + "autoencoder_2D_latent.pth"
DIFFUSION_CHECKPOINT = CHECKPOINT_DIR + "diffusion_model_2D.pth"
TRAIN_DATA=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/train.csv'
VAL_DATA=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/val.csv'
TEST_DATA=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/test.csv'
DATA_DIR=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri'
LOGS_DIR = r'/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/logs'
