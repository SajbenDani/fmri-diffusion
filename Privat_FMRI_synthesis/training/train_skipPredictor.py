import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import sys

# Get the parent directory of the current script (training/)
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from models.autoencoder import Improved3DAutoencoder
from models.skipPredictor import SkipPredictor
from utils.dataset import FMRIDataModule
from config import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load and freeze the pre-trained autoencoder
autoencoder = Improved3DAutoencoder(latent_dims=(8, 8, 8), num_classes=5).to(DEVICE)
PRETRAINED_AE_PATH = os.path.join(CHECKPOINT_DIR, "finetuned_autoencoder_best.pth")
if os.path.exists(PRETRAINED_AE_PATH):
    autoencoder.load_state_dict(torch.load(PRETRAINED_AE_PATH, map_location=DEVICE))
    print("✅ Loaded pretrained autoencoder.")
else:
    raise FileNotFoundError("Pretrained autoencoder not found.")
for param in autoencoder.parameters():
    param.requires_grad = False
autoencoder.eval()

# Initialize SkipPredictor
skip_predictor = SkipPredictor(latent_dims=(8, 8, 8)).to(DEVICE)
optimizer = optim.AdamW(skip_predictor.parameters(), lr=1e-3)
mse_criterion = nn.MSELoss()

# Paths for saving models
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "skip_predictor_best.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "skip_predictor_last.pth")

# Load last checkpoint if it exists (for resuming after crash)
if os.path.exists(LAST_MODEL_PATH):
    skip_predictor.load_state_dict(torch.load(LAST_MODEL_PATH, map_location=DEVICE))
    print("✅ Loaded last skip predictor checkpoint for resuming training.")

# Data loading
data_module = FMRIDataModule(
    train_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/train.csv',
    val_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/val.csv',
    test_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/test.csv',
    data_dir=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri',
    batch_size=BATCH_SIZE,
    num_workers=16
)
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# One-hot encoding function
def one_hot_encode(labels, num_classes=5):
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

# Training loop with checkpointing and early stopping
EPOCHS = 50
PATIENCE = 10  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(EPOCHS):
    skip_predictor.train()
    train_loss = 0.0
    for fmri_tensor, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
        labels_one_hot = one_hot_encode(labels, num_classes=5)

        with torch.no_grad():
            z, latent_3d, actual_e1, actual_e2 = autoencoder.encode(fmri_tensor)
            latent_3d = latent_3d.unsqueeze(1)
            label_features = autoencoder.label_embedding(labels_one_hot)
            modulated_z = z * torch.sigmoid(label_features)

        target_shape_e1 = actual_e1.shape[2:]
        target_shape_e2 = actual_e2.shape[2:]
        pred_e1, pred_e2 = skip_predictor(latent_3d, target_shape_e1, target_shape_e2)

        loss_e1 = mse_criterion(pred_e1, actual_e1)
        loss_e2 = mse_criterion(pred_e2, actual_e2)
        loss = loss_e1 + loss_e2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.6f}")

    # Validation with reconstruction test
    skip_predictor.eval()
    val_loss = 0.0
    recon_loss = 0.0
    with torch.no_grad():
        for fmri_tensor, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
            labels_one_hot = one_hot_encode(labels, num_classes=5)

            recon_actual, _ = autoencoder(fmri_tensor, labels_one_hot)

            z, latent_3d, actual_e1, actual_e2 = autoencoder.encode(fmri_tensor)
            latent_3d = latent_3d.unsqueeze(1)
            target_shape_e1 = actual_e1.shape[2:]
            target_shape_e2 = actual_e2.shape[2:]
            pred_e1, pred_e2 = skip_predictor(latent_3d, target_shape_e1, target_shape_e2)

            label_features = autoencoder.label_embedding(labels_one_hot)
            modulated_z = z * torch.sigmoid(label_features)

            recon_pred = autoencoder.decode(modulated_z, pred_e1, pred_e2)

            loss_e1 = mse_criterion(pred_e1, actual_e1)
            loss_e2 = mse_criterion(pred_e2, actual_e2)
            val_loss += (loss_e1 + loss_e2).item()
            recon_loss += mse_criterion(recon_pred, recon_actual).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_recon_loss = recon_loss / len(val_loader)
    print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.6f}, Reconstruction Loss: {avg_recon_loss:.6f}")

    # Checkpointing: Save if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(skip_predictor.state_dict(), BEST_MODEL_PATH)
        print(f"✅ Saved best model at epoch {epoch+1} with validation loss: {avg_val_loss:.6f}")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"No improvement in validation loss. Early stopping counter: {early_stop_counter}/{PATIENCE}")

    # Always save the last model state for crash recovery
    torch.save(skip_predictor.state_dict(), LAST_MODEL_PATH)

    # Early stopping
    if early_stop_counter >= PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

print("✅ Training completed.")