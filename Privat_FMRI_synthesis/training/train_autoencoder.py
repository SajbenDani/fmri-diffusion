import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Add parent directory to sys.path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

# Imports
from models.autoencoder import Improved3DAutoencoder
from utils.dataset import FMRIDataModule
from config import *

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize model
autoencoder = Improved3DAutoencoder(latent_dims=(8, 8, 8), num_classes=NUM_CLASSES).to(DEVICE)

# Load last checkpoint if available
if os.path.exists(BEST_MODEL_PATH):
    autoencoder.load_state_dict(torch.load(BEST_MODEL_PATH))
    print("✅ Loaded last saved model for crash recovery")

# Losses and optimizer
mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
optimizer = optim.AdamW(autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

# Dataset
data_module = FMRIDataModule(
    train_csv=TRAIN_CSV,
    val_csv=VAL_CSV,
    test_csv=TEST_CSV,
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Utility
def one_hot_encode(labels, num_classes=NUM_CLASSES):
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

# Training loop
best_loss = float('inf')
counter = 0

print(f"Starting training with batch size: {BATCH_SIZE}")
print(f"Using device: {DEVICE}")
print(f"Latent space: {autoencoder.latent_dims} (total: {autoencoder.latent_size})")

for epoch in range(EPOCHS_AUTOENCODER):
    autoencoder.train()
    train_loss = 0.0

    for fmri_tensor, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_AUTOENCODER} [Train]"):
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
        labels_one_hot = one_hot_encode(labels)
        recon, _ = autoencoder(fmri_tensor, labels_one_hot)

        mse_loss = mse_criterion(recon, fmri_tensor)
        l1_loss = l1_criterion(recon, fmri_tensor)
        ssim_loss = 1 - ssim(recon, fmri_tensor)
        loss = 0.7 * mse_loss + 0.2 * l1_loss + 0.1 * ssim_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {train_loss:.6f}")

    # Validation
    autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for fmri_tensor, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS_AUTOENCODER} [Val]"):
            fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
            labels_one_hot = one_hot_encode(labels)
            recon, _ = autoencoder(fmri_tensor, labels_one_hot)

            mse_loss = mse_criterion(recon, fmri_tensor)
            l1_loss = l1_criterion(recon, fmri_tensor)
            ssim_loss = 1 - ssim(recon, fmri_tensor)
            composite_loss = 0.7 * mse_loss + 0.2 * l1_loss + 0.1 * ssim_loss
            val_loss += composite_loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1} Validation Loss: {val_loss:.6f}")

    # Logging, early stopping, saving
    with open(LOG_FILE, 'a') as f:
        f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f}\n")

    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(autoencoder.state_dict(), BEST_MODEL_PATH)
        print(f"✅ Checkpoint saved at epoch {epoch+1}")
    else:
        counter += 1
        if counter >= PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

