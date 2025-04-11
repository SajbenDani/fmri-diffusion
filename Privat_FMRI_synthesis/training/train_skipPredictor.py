import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from models.autoencoder import Improved3DAutoencoder
from models.skipPredictor import SkipPredictor
from utils.dataset import FMRIDataModule
from config import *

# Setup
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = torch.device(DEVICE)

# Load and freeze pretrained autoencoder
autoencoder = Improved3DAutoencoder(latent_dims=LATENT_SHAPE, num_classes=NUM_CLASSES).to(DEVICE)
if os.path.exists(BEST_MODEL_PATH):
    autoencoder.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    print(" Loaded pretrained autoencoder.")
else:
    raise FileNotFoundError(" Pretrained autoencoder not found.")
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False

# Initialize SkipPredictor and optimizer
skip_predictor = SkipPredictor(latent_dims=LATENT_SHAPE).to(DEVICE)
optimizer = optim.AdamW(skip_predictor.parameters(), lr=LEARNING_RATE)
mse_criterion = nn.MSELoss()

# Resume from last checkpoint
if os.path.exists(SKIP_PREDICTOR_BEST):
    skip_predictor.load_state_dict(torch.load(SKIP_PREDICTOR_BEST, map_location=DEVICE))
    print(" Loaded last skip predictor checkpoint for resuming training.")

# Data loading
data_module = FMRIDataModule(
    train_csv=TRAIN_CSV,
    val_csv=VAL_CSV,
    test_csv=TEST_CSV,
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=16
)
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# One-hot encoding utility
def one_hot_encode(labels, num_classes=NUM_CLASSES):
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1)

# Training loop
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(EPOCHS_SKIP):
    skip_predictor.train()
    train_loss = 0.0

    for fmri_tensor, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_SKIP} [Train]"):
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
        labels_one_hot = one_hot_encode(labels)

        with torch.no_grad():
            z, latent_3d, actual_e1, actual_e2 = autoencoder.encode(fmri_tensor)
            latent_3d = latent_3d.unsqueeze(1)
            label_features = autoencoder.label_embedding(labels_one_hot)
            modulated_z = z * torch.sigmoid(label_features)

        pred_e1, pred_e2 = skip_predictor(latent_3d, actual_e1.shape[2:], actual_e2.shape[2:])
        loss = mse_criterion(pred_e1, actual_e1) + mse_criterion(pred_e2, actual_e2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f" Epoch {epoch+1} Training Loss: {avg_train_loss:.6f}")

    # Validation
    skip_predictor.eval()
    val_loss = 0.0
    recon_loss = 0.0

    with torch.no_grad():
        for fmri_tensor, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS_SKIP} [Val]"):
            fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
            labels_one_hot = one_hot_encode(labels)

            recon_actual, _ = autoencoder(fmri_tensor, labels_one_hot)
            z, latent_3d, actual_e1, actual_e2 = autoencoder.encode(fmri_tensor)
            latent_3d = latent_3d.unsqueeze(1)
            label_features = autoencoder.label_embedding(labels_one_hot)
            modulated_z = z * torch.sigmoid(label_features)

            pred_e1, pred_e2 = skip_predictor(latent_3d, actual_e1.shape[2:], actual_e2.shape[2:])
            recon_pred = autoencoder.decode(modulated_z, pred_e1, pred_e2)

            val_loss += mse_criterion(pred_e1, actual_e1).item() + mse_criterion(pred_e2, actual_e2).item()
            recon_loss += mse_criterion(recon_pred, recon_actual).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_recon_loss = recon_loss / len(val_loader)
    print(f" Epoch {epoch+1} Validation Loss: {avg_val_loss:.6f}, Reconstruction Loss: {avg_recon_loss:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(skip_predictor.state_dict(), SKIP_PREDICTOR_BEST)
        print(f" Saved best model at epoch {epoch+1}.")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f" No improvement. Early stopping counter: {early_stop_counter}/{PATIENCE}")


    # Early stopping
    if early_stop_counter >= PATIENCE:
        print(f" Early stopping triggered at epoch {epoch+1}.")
        break

print(" Training completed.")
