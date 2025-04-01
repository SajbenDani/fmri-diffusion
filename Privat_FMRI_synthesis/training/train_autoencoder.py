import torch
import torch.optim as optim
import os
import sys
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Get the parent directory of the training folder
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

# Import configuration settings
from config import *
from models.autoencoder import fMRIAutoencoder
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
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "training_log.txt")

# Initialize model
autoencoder = fMRIAutoencoder().to(DEVICE)
if os.path.exists(AUTOENCODER_CHECKPOINT):
    autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT))
    print("✅ Loaded last saved model for crash recovery")

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# Learning Rate Scheduler (Cosine Annealing)
#scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_AUTOENCODER, eta_min=5e-6)

# Initialize Data Module
data_module = FMRIDataModule(
    train_csv=TRAIN_DATA,
    val_csv=VAL_DATA,
    test_csv=TEST_DATA,
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=16
)
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Training setup
best_loss = float('inf')
counter = 0
patience = 10

with open(LOG_FILE, "w") as log_file:
    log_file.write("Epoch,Train Loss,Validation Loss,Train MSE,Train L1,Train SSIM,Val MSE,Val L1,Val SSIM\n")

    for epoch in range(EPOCHS_AUTOENCODER):
        autoencoder.train()
        train_loss, train_mse, train_l1, train_ssim = 0, 0, 0, 0
        print(f"Epoch {epoch + 1}/{EPOCHS_AUTOENCODER}")
        progress_bar = tqdm(train_loader, desc=f"Training {epoch+1}/{EPOCHS_AUTOENCODER}")
        
        for fmri_tensor, labels in progress_bar:
            fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
            recon = autoencoder(fmri_tensor, labels)
            
            mse_loss = criterion(recon, fmri_tensor)
            l1_loss = torch.nn.functional.l1_loss(recon, fmri_tensor)
            ssim_loss = 1 - ssim(recon, fmri_tensor)
            loss = 0.5 * mse_loss + 0.2 * l1_loss + 0.3 * ssim_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += mse_loss.item()
            train_l1 += l1_loss.item()
            train_ssim += ssim_loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        train_l1 /= len(train_loader)
        train_ssim /= len(train_loader)
        print(f"Epoch {epoch + 1} completed - Avg Train Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, L1: {train_l1:.4f}, SSIM: {train_ssim:.4f}")
        # Update Learning Rate Scheduler
        #scheduler.step()
        #print(f"🔄 LR Updated: {scheduler.get_last_lr()[0]:.6f}")
      
        # Validation Phase
        autoencoder.eval()
        val_loss, val_mse, val_l1, val_ssim = 0, 0, 0, 0
        val_bar = tqdm(val_loader, desc=f"Validating {epoch+1}/{EPOCHS_AUTOENCODER}")
        with torch.no_grad():
            for fmri_tensor, labels in val_bar:
                fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
                recon = autoencoder(fmri_tensor, labels)
                
                mse_loss = criterion(recon, fmri_tensor)
                l1_loss = torch.nn.functional.l1_loss(recon, fmri_tensor)
                ssim_loss = 1 - ssim(recon, fmri_tensor)
                loss = 0.5 * mse_loss + 0.2 * l1_loss + 0.3 * ssim_loss
                
                val_loss += loss.item()
                val_mse += mse_loss.item()
                val_l1 += l1_loss.item()
                val_ssim += ssim_loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_l1 /= len(val_loader)
        val_ssim /= len(val_loader)
        print(f"Epoch {epoch + 1} - Avg Validation Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, L1: {val_l1:.4f}, SSIM: {val_ssim:.4f}")

        # Log losses
        log_file.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{train_mse:.4f},{train_l1:.4f},{train_ssim:.4f},{val_mse:.4f},{val_l1:.4f},{val_ssim:.4f}\n")
        log_file.flush()

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

print(f"✅ Training complete. Best model saved at: {AUTOENCODER_CHECKPOINT}")
