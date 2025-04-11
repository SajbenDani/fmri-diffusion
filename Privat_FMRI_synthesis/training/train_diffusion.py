import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from utils.dataset import FMRIDataModule
from models.autoencoder import Improved3DAutoencoder
from models.diffusion import LatentDiffusion
from config import *

from torchmetrics.image import StructuralSimilarityIndexMeasure

# Utility: one-hot encode class labels
def one_hot_encode(labels, num_classes=NUM_CLASSES):
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot

# Load pre-trained autoencoder
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f'Autoencoder checkpoint not found: {BEST_MODEL_PATH}')
autoencoder = Improved3DAutoencoder(latent_dims=LATENT_SHAPE, num_classes=NUM_CLASSES).to(DEVICE)
autoencoder.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
autoencoder.eval()
print("Loaded pre-trained autoencoder.")

# Initialize data module
data_module = FMRIDataModule(
    train_csv=TRAIN_CSV,
    val_csv=VAL_CSV,
    test_csv=TEST_CSV,
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR
)
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Initialize diffusion model
diffusion_model = LatentDiffusion(latent_shape=LATENT_SHAPE, num_classes=NUM_CLASSES, device=DEVICE)

if os.path.exists(DIFFUSION_CHECKPOINT):
    state = torch.load(DIFFUSION_CHECKPOINT, map_location=DEVICE)
    diffusion_model.model.load_state_dict(state)
    print(f"Loaded diffusion model checkpoint from {DIFFUSION_CHECKPOINT}")

diffusion_model.model.train()

# Losses
mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

best_val_loss = float('inf')
stopping_counter = 0

# Logging
with open(LOG_FILE, 'w') as f:
    f.write("Epoch,TrainLoss,ValLoss\n")

train_losses, val_losses = [], []

for epoch in range(EPOCHS_DIFFUSION):
    train_loss = 0.0
    diffusion_model.model.train()

    for fmri_tensor, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_DIFFUSION} [Train]"):
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
        labels_one_hot = one_hot_encode(labels)

        with torch.no_grad():
            _, latent_3d, _, _ = autoencoder.encode(fmri_tensor)
        latent_3d = latent_3d.unsqueeze(1)

        noisy_latent, t, true_noise = diffusion_model.forward_diffusion(latent_3d)
        pred_noise = diffusion_model.model(noisy_latent, t, labels_one_hot)

        mse_loss = mse_criterion(pred_noise, true_noise)
        l1_loss = l1_criterion(pred_noise, true_noise)
        ssim_loss = 1 - ssim(pred_noise, true_noise)

        loss = W_MSE * mse_loss + W_L1 * l1_loss + W_SSIM * ssim_loss

        diffusion_model.optim.zero_grad()
        loss.backward()
        diffusion_model.optim.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1} Training Loss: {train_loss:.6f}")

    # Validation
    diffusion_model.model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for fmri_tensor, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS_DIFFUSION} [Val]"):
            fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
            labels_one_hot = one_hot_encode(labels)

            _, latent_3d, _, _ = autoencoder.encode(fmri_tensor)
            latent_3d = latent_3d.unsqueeze(1)

            noisy_latent, t, true_noise = diffusion_model.forward_diffusion(latent_3d)
            pred_noise = diffusion_model.model(noisy_latent, t, labels_one_hot)

            mse_loss = mse_criterion(pred_noise, true_noise)
            l1_loss = l1_criterion(pred_noise, true_noise)
            ssim_loss = 1 - ssim(pred_noise, true_noise)
            loss = W_MSE * mse_loss + W_L1 * l1_loss + W_SSIM * ssim_loss

            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1} Validation Loss: {val_loss:.6f}")

    with open(LOG_FILE, 'a') as f:
        f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f}\n")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        stopping_counter = 0
        torch.save(diffusion_model.model.state_dict(), DIFFUSION_CHECKPOINT)
        print(f"Checkpoint saved at epoch {epoch+1}")
    else:
        stopping_counter += 1
        if stopping_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Latent-space Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(LOSS_PLOT_PATH)
plt.show()
