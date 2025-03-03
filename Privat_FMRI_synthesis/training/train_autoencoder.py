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

# Get the parent directory of the current script (training/)
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add parent directory to sys.path
sys.path.append(PARENT_DIR)
from models.autoencoder import Improved3DAutoencoder  # Import your autoencoder
from utils.dataset import FMRIDataModule
from config import *

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Define constants
NUM_CLASSES = 5  # Matches the autoencoder's default
EPOCHS = 100     # Adjustable based on your needs

# Initialize the autoencoder
autoencoder = Improved3DAutoencoder(latent_dims=(8, 8, 8), num_classes=NUM_CLASSES).to(DEVICE)

# Load last saved checkpoint if available (for crash recovery)
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "improved_autoencoder_last.pth")
if os.path.exists(LAST_MODEL_PATH):
    autoencoder.load_state_dict(torch.load(LAST_MODEL_PATH))
    print("✅ Loaded last saved model for crash recovery")

# Define losses and optimizer
mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()
optimizer = optim.AdamW(autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# Initialize FMRIDataModule
data_module = FMRIDataModule(
    train_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/train.csv',
    val_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/val.csv',
    test_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/test.csv',
    data_dir=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri',
    batch_size=BATCH_SIZE,
    num_workers=16
)

# Setup dataloaders
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Function to one-hot encode labels
def one_hot_encode(labels, num_classes=NUM_CLASSES):
    """
    Convert integer labels to one-hot encoded tensors.
    Args:
        labels: Tensor of shape [batch_size] with integer labels (0-4)
        num_classes: Number of classes (default: 5)
    Returns:
        one_hot: Tensor of shape [batch_size, num_classes]
    """
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

# Initialize tracking variables
best_loss = float('inf')
patience = 15  # Early stopping patience
counter = 0

print(f"Starting training with batch size: {BATCH_SIZE}")
print(f"Using device: {DEVICE}")
print(f"Model latent space dimensions: {autoencoder.latent_dims} (total: {autoencoder.latent_size})")

# Training loop
for epoch in range(EPOCHS):
    autoencoder.train()
    train_loss = 0.0
    
    for fmri_tensor, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
        labels_one_hot = one_hot_encode(labels, num_classes=NUM_CLASSES)
        
        # Forward pass through autoencoder
        recon, _ = autoencoder(fmri_tensor, labels_one_hot)
        
        # Compute composite loss
        mse_loss = mse_criterion(recon, fmri_tensor)
        l1_loss = l1_criterion(recon, fmri_tensor)
        ssim_loss = 1 - ssim(recon, fmri_tensor)
        loss = 0.7 * mse_loss + 0.2 * l1_loss + 0.1 * ssim_loss
        
        # Backward pass and optimization
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
        for fmri_tensor, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
            labels_one_hot = one_hot_encode(labels, num_classes=NUM_CLASSES)
            
            # Forward pass through autoencoder
            recon, _ = autoencoder(fmri_tensor, labels_one_hot)
            
            # Compute composite loss
            mse_loss = mse_criterion(recon, fmri_tensor)
            l1_loss = l1_criterion(recon, fmri_tensor)
            ssim_loss = 1 - ssim(recon, fmri_tensor)
            composite_loss = 0.7 * mse_loss + 0.2 * l1_loss + 0.1 * ssim_loss
            val_loss += composite_loss.item()
    
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1} Validation Loss: {val_loss:.6f}")
    
    # Logging, checkpointing, and early stopping
    with open(log_file, 'a') as f:
        f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f}\n")
    
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(autoencoder.state_dict(), os.path.join(CHECKPOINT_DIR, "best_autoencoder.pth"))
        print(f"Checkpoint saved at epoch {epoch+1}")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break
    
    # Save last model state for crash recovery
    torch.save(autoencoder.state_dict(), LAST_MODEL_PATH)