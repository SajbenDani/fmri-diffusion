import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
import random
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Get the parent directory of the training folder
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

# Import configuration settings
from config import *
from models.autoencoder import fMRIAutoencoder  # Assuming autoencoder.py contains your model
from utils.dataset import FMRIDataModule

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Ensure directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "evaluation_log.txt")

# Initialize model
autoencoder = fMRIAutoencoder(latent_dim=1024, num_classes=5).to(DEVICE)
if os.path.exists(AUTOENCODER_CHECKPOINT):
    autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT))
    print(f"✅ Loaded pre-trained autoencoder from {AUTOENCODER_CHECKPOINT}")
else:
    raise FileNotFoundError(f"Checkpoint not found at {AUTOENCODER_CHECKPOINT}. Please ensure the trained model exists.")

# Define loss metric
criterion = torch.nn.MSELoss()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# Initialize Data Module and load test data
data_module = FMRIDataModule(
    train_csv=TRAIN_DATA,
    val_csv=VAL_DATA,
    test_csv=TEST_DATA,
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=16
)
data_module.setup()
test_loader = data_module.test_dataloader()

# Evaluation
autoencoder.eval()
full_mse, full_l1, full_ssim = 0, 0, 0
loss_conv_total, loss_lin_total = 0, 0

print("Evaluating autoencoder on test data...")
progress_bar = tqdm(test_loader, desc="Evaluating Test Set")

with torch.no_grad():
    for fmri_tensor, labels in progress_bar:
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
        
        # Full reconstruction
        recon = autoencoder(fmri_tensor, labels)
        mse_loss = criterion(recon, fmri_tensor)
        l1_loss = torch.nn.functional.l1_loss(recon, fmri_tensor)
        ssim_loss = 1 - ssim(recon, fmri_tensor)
        
        full_mse += mse_loss.item()
        full_l1 += l1_loss.item()
        full_ssim += ssim_loss.item()
        
        # Convolutional Reconstruction Loss (bypassing linear layers)
        conv_part = nn.Sequential(*autoencoder.encoder[:10])  # Up to third ReLU
        conv_features = conv_part(fmri_tensor)  # Shape: (batch, 128, 12, 14, 12)
        recon_conv = autoencoder.decoder[5:](conv_features)  # From ConvTranspose3d onward
        loss_conv = torch.nn.functional.mse_loss(recon_conv, fmri_tensor)
        loss_conv_total += loss_conv.item()
        
        # Linear Compression Loss
        flattened = autoencoder.encoder[10](conv_features)  # Flatten
        latent = autoencoder.encoder[11:](flattened)  # Linear layers to latent
        reconstructed_flattened = autoencoder.decoder[:4](latent)  # Linear layers back
        loss_lin = torch.nn.functional.mse_loss(flattened, reconstructed_flattened)
        loss_lin_total += loss_lin.item()
        
        progress_bar.set_postfix(mse=f"{mse_loss.item():.4f}")

# Average losses
num_batches = len(test_loader)
full_mse /= num_batches
full_l1 /= num_batches
full_ssim /= num_batches
loss_conv_avg = loss_conv_total / num_batches
loss_lin_avg = loss_lin_total / num_batches

# Print results
print("\nEvaluation Results on Test Set:")
print(f"Full Reconstruction MSE: {full_mse:.4f}, L1: {full_l1:.4f}, SSIM Loss: {full_ssim:.4f}")
print(f"Convolutional Reconstruction MSE: {loss_conv_avg:.4f}")
print(f"Linear Compression MSE: {loss_lin_avg:.4f}")

# Log results
with open(LOG_FILE, "w") as log_file:
    log_file.write("Metric,Value\n")
    log_file.write(f"Full MSE,{full_mse:.4f}\n")
    log_file.write(f"Full L1,{full_l1:.4f}\n")
    log_file.write(f"Full SSIM Loss,{full_ssim:.4f}\n")
    log_file.write(f"Convolutional MSE,{loss_conv_avg:.4f}\n")
    log_file.write(f"Linear MSE,{loss_lin_avg:.4f}\n")
    print(f"✅ Results logged to {LOG_FILE}")

print("✅ Evaluation complete.")