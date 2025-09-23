"""
Configuration module for fMRI Diffusion Model Project.

This module contains all the global configuration parameters for the fMRI-based
Latent Diffusion Model (LDM) super-resolution framework. The project implements
a two-stage approach: first training an autoencoder to create a latent space
representation of fMRI data, then training a diffusion model in that latent space
for high-quality fMRI image generation.

Key Components:
    - Device configuration for CPU/GPU training
    - Training hyperparameters for both autoencoder and diffusion stages
    - File paths for datasets, checkpoints, and logs
    - Model architecture parameters
    - Evaluation and visualization settings

Usage:
    Import this module to access configuration parameters:
    ```python
    from config import DEVICE, BATCH_SIZE, LEARNING_RATE
    ```

Architecture Overview:
    The system uses a 3D autoencoder to compress fMRI volumes (91x109x91) into
    a lower-dimensional latent space, then applies diffusion models for generation
    and super-resolution tasks. This approach enables efficient training and
    high-quality synthesis of brain activation patterns.
"""

import torch

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

# Automatically detect and configure compute device
# Users can manually override by changing this value if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Batch size for training - balance between memory usage and gradient stability
# Smaller batches may be needed for 3D data due to memory constraints
BATCH_SIZE = 12

# Training epochs for the two-stage approach
EPOCHS_AUTOENCODER = 50    # Stage 1: Learn latent representation of fMRI data
EPOCHS_DIFFUSION = 100     # Stage 2: Train diffusion model in latent space

# Learning rate for Adam optimizer
# Conservative value chosen for stable training of both autoencoder and diffusion model
LEARNING_RATE = 1e-4

# Directory for saving model checkpoints during training
CHECKPOINT_DIR = "checkpoints_New/"

# =============================================================================
# MODEL CHECKPOINT PATHS
# =============================================================================

# Trained autoencoder model checkpoint (Stage 1 output)
# This model learns to encode/decode fMRI volumes to/from latent space
AUTOENCODER_CHECKPOINT = CHECKPOINT_DIR + "autoencoder_2D_latent.pth"

# Trained diffusion model checkpoint (Stage 2 output) 
# This model performs diffusion-based generation in the latent space
DIFFUSION_CHECKPOINT = CHECKPOINT_DIR + "diffusion_model_2D.pth"

# =============================================================================
# DATASET PATHS
# =============================================================================

# CSV files containing dataset splits with file paths and labels
# Expected format: columns for file paths, labels (lh, rh, lf, rf, t), and metadata
TRAIN_DATA = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/train.csv'
VAL_DATA = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/val.csv'
TEST_DATA = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/test.csv'

# Root directory containing the actual fMRI data files (.nii.gz format)
# Structure expected: DATA_DIR/subjectID/sessionID/fmri_file.nii.gz
DATA_DIR = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri'

# Directory for storing training logs, tensorboard files, and other outputs
LOGS_DIR = r'/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/logs'
