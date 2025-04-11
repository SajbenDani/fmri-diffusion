import torch
import torch.optim as optim
import os
import numpy as np
import random
import sys

# Get the parent directory of the current script (training/)
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from models.autoencoder import fMRIAutoencoder
from utils.dataset import FMRIDataModule
from config import *
from torchmetrics import StructuralSimilarityIndexMeasure

# Set the random seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize model
autoencoder = fMRIAutoencoder().to(DEVICE)

# Load last saved checkpoint if available
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "autoencoder_last.pth")
if os.path.exists(LAST_MODEL_PATH):
    autoencoder.load_state_dict(torch.load(LAST_MODEL_PATH))
    print("✅ Loaded last saved model for crash recovery")

# Define loss, optimizer, metrics
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# Initialize dataset
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

best_loss = float('inf')
counter = 0

for epoch in range(EPOCHS_AUTOENCODER):
    # Training
    autoencoder.train()
    train_loss = 0
    print(f"Epoch {epoch + 1}/{EPOCHS_AUTOENCODER}")

    for i, (fmri_tensor, labels) in enumerate(train_loader):
        fmri_tensor = fmri_tensor.to(DEVICE)
        labels = labels.to(DEVICE)
        recon = autoencoder(fmri_tensor, labels)

        mse_loss = criterion(recon, fmri_tensor)
        ssim_loss = 1 - ssim(recon, fmri_tensor)
        loss = mse_loss + 0.1 * ssim_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(f"Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1} completed - Avg Train Loss: {train_loss:.4f}")

    # Validation
    autoencoder.eval()
    val_loss = 0
    with torch.no_grad():
        for fmri_tensor, labels in val_loader:
            fmri_tensor = fmri_tensor.to(DEVICE)
            labels = labels.to(DEVICE)
            recon = autoencoder(fmri_tensor, labels)

            mse_loss = criterion(recon, fmri_tensor)
            ssim_loss = 1 - ssim(recon, fmri_tensor)
            loss = mse_loss + 0.1 * ssim_loss
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch + 1} - Avg Validation Loss: {val_loss:.4f}")

    # Save model
    torch.save(autoencoder.state_dict(), LAST_MODEL_PATH)
    print(f"Model checkpoint saved after epoch {epoch + 1}")

    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(autoencoder.state_dict(), AUTOENCODER_CHECKPOINT)
        print("Best model updated!")
    else:
        counter += 1
        print(f"Patience counter: {counter}/{PATIENCE}")
        if counter >= PATIENCE:
            print("Early stopping triggered!")
            break

print(f"Training complete. Final model saved at: {LAST_MODEL_PATH}")
