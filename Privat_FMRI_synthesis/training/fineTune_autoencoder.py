import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Szülő könyvtár hozzáadása
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)
from models.autoencoder import Improved3DAutoencoder
from utils.dataset import FMRIDataModule
from config import *

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Modell betöltése
autoencoder = Improved3DAutoencoder(latent_dims=LATENT_SHAPE, num_classes=NUM_CLASSES).to(DEVICE)

CHECKPOINT_PATH = BEST_MODEL_PATH  # from config
if os.path.exists(CHECKPOINT_PATH):
    autoencoder.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print(f" Loaded checkpoint from {CHECKPOINT_PATH}")
else:
    raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

# Loss functions
mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# Fine-tuning optimizer setup (can move these into config if preferred)
params = [
    {'params': [p for n, p in autoencoder.named_parameters() if 'enc_fc' in n or 'dec_fc' in n], 'lr': 1e-6},
    {'params': [p for n, p in autoencoder.named_parameters() if 'dec_conv' in n or 'dec_norm' in n or 'label_embedding' in n], 'lr': 1e-5},
    {'params': [p for n, p in autoencoder.named_parameters() if 'enc_conv' in n or 'enc_norm' in n], 'lr': 5e-6}
]
optimizer = optim.AdamW(params, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

# Data
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

# One-hot encoding
def one_hot_encode(labels, num_classes=NUM_CLASSES):
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

# Tracking
best_loss = float('inf')
train_losses, val_losses = [], []
counter = 0

print(f"Starting fine-tuning with batch size: {BATCH_SIZE}")
print(f"Using device: {DEVICE}")

EPOCHS = EPOCHS_FINETUNE if 'EPOCHS_FINETUNE' in globals() else 20
W_VAR = W_VAR if 'W_VAR' in globals() else 1.0
TARGET_VARIANCE = TARGET_VARIANCE if 'TARGET_VARIANCE' in globals() else 0.01


# Training loop
for epoch in range(EPOCHS):
    autoencoder.train()
    train_loss = 0.0
    
    for fmri_tensor, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
        labels_one_hot = one_hot_encode(labels)
        
        recon, _ = autoencoder(fmri_tensor, labels_one_hot)
        
        mse_loss = mse_criterion(recon, fmri_tensor)
        l1_loss = l1_criterion(recon, fmri_tensor)
        ssim_loss = 1 - ssim(recon, fmri_tensor)
        pixel_diff_var = (recon - fmri_tensor).var()
        variance_loss = W_VAR * torch.abs(TARGET_VARIANCE - pixel_diff_var)
        
        loss = W_MSE * mse_loss + W_L1 * l1_loss + W_SSIM * ssim_loss + variance_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1} Training Loss: {train_loss:.6f}")

    # Validation
    autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for fmri_tensor, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
            labels_one_hot = one_hot_encode(labels)
            
            recon, _ = autoencoder(fmri_tensor, labels_one_hot)
            
            mse_loss = mse_criterion(recon, fmri_tensor)
            l1_loss = l1_criterion(recon, fmri_tensor)
            ssim_loss = 1 - ssim(recon, fmri_tensor)
            pixel_diff_var = (recon - fmri_tensor).var()
            variance_loss = W_VAR * torch.abs(TARGET_VARIANCE - pixel_diff_var)
            composite_loss = W_MSE * mse_loss + W_L1 * l1_loss + W_SSIM * ssim_loss + variance_loss
            
            val_loss += composite_loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1} Validation Loss: {val_loss:.6f}")

    scheduler.step(val_loss)

    # Save best and last
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(autoencoder.state_dict(), CHECKPOINT_PATH)
        print(f" Saved best checkpoint at epoch {epoch+1}")
    else:
        counter += 1
        if counter >= PATIENCE:
            print(" Early stopping triggered.")
            break
    
    

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Fine-tuning Loss over Epochs')
plt.legend()
plt.grid(True)
plot_path = os.path.splitext(CHECKPOINT_PATH)[0] + "_loss_plot.png"
plt.savefig(plot_path)
plt.show()

print(" Fine-tuning completed.")
