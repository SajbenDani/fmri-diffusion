import torch

# Switch between CPU and GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Change manually if needed

# Training settings
BATCH_SIZE = 8
EPOCHS_AUTOENCODER = 50
EPOCHS_DIFFUSION = 100
LEARNING_RATE = 5e-4
CHECKPOINT_DIR = "checkpoints_New/"

# Paths
AUTOENCODER_CHECKPOINT = CHECKPOINT_DIR + "autoencoder.pth"
DIFFUSION_CHECKPOINT = CHECKPOINT_DIR + "diffusion_model.pth"
