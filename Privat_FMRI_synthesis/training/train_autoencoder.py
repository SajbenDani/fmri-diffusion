import torch
import torch.optim as optim
import os
import numpy as np
import random
import sys
# Get the parent directory of the current script (training/)
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add parent directory to sys.path
sys.path.append(PARENT_DIR)
from models.autoencoder import fMRIAutoencoder
from utils.dataset import FMRIDataModule  # Import the new FMRIDataModule
from config import *
from torchmetrics import StructuralSimilarityIndexMeasure

# Set the random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Ensure reproducibility for CUDA
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize model
autoencoder = fMRIAutoencoder().to(DEVICE)

# Load last saved checkpoint if available (for crash recovery)
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "autoencoder_last.pth")
if os.path.exists(LAST_MODEL_PATH):
    autoencoder.load_state_dict(torch.load(LAST_MODEL_PATH))
    print("✅ Loaded last saved model for crash recovery")

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# Initialize FMRIDataModule
data_module = FMRIDataModule(
    train_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/train.csv',
    val_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/val.csv',
    test_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/test.csv',
    data_dir=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri',
    batch_size=BATCH_SIZE,
    num_workers=16  # Specify number of workers here
)

# ⚠️ Call setup() before using dataloaders
data_module.setup()

# Define dataloaders only once
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Initialize tracking variables before the loop
best_loss = float('inf')  # Initialize best_loss to infinity
counter = 0               # Initialize patience counter
patience = 10             # Define patience value if not defined in config

# Training loop with validation
for epoch in range(EPOCHS_AUTOENCODER):
    # Training Phase
    autoencoder.train()
    train_loss = 0
    print(f"Epoch {epoch + 1}/{EPOCHS_AUTOENCODER}")

    for i, (fmri_tensor, labels) in enumerate(train_loader):  # ✅ Use stored train_loader
        fmri_tensor = fmri_tensor.to(DEVICE)
        labels = labels.to(DEVICE)
        recon = autoencoder(fmri_tensor, labels)
        mse_loss = criterion(recon, fmri_tensor)
        ssim_loss = 1 - ssim(recon, fmri_tensor)  # SSIM (higher is better, so we subtract from 1)
        loss = mse_loss + 0.1 * ssim_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(f"Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1} completed - Avg Train Loss: {train_loss:.4f}")

    # Validation Phase
    autoencoder.eval()
    val_loss = 0

    with torch.no_grad():
        for i, (fmri_tensor, labels) in enumerate(val_loader):  # ✅ Use stored val_loader
            fmri_tensor = fmri_tensor.to(DEVICE)
            labels = labels.to(DEVICE)
            recon = autoencoder(fmri_tensor, labels)

            mse_loss = criterion(recon, fmri_tensor)
            ssim_loss = 1 - ssim(recon, fmri_tensor)
            loss = mse_loss + 0.1 * ssim_loss

            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch + 1} - Avg Validation Loss: {val_loss:.4f}")

    # ✅ Save the model after every epoch (for crash recovery)
    torch.save(autoencoder.state_dict(), LAST_MODEL_PATH)
    print(f"💾 Model checkpoint saved after epoch {epoch + 1}")

    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(autoencoder.state_dict(), AUTOENCODER_CHECKPOINT)
        print("✅ Best model updated!")

    else:
        counter += 1
        print(f"Patience counter: {counter}/{patience}")
        if counter >= patience:
            print("🚨 Early stopping triggered!")
            break

print(f"✅ Training complete. Final model saved at: {LAST_MODEL_PATH}")