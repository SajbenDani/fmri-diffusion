import torch
import torch.optim as optim
import os
import sys
import numpy as np
import random
from tqdm import tqdm

# Get the parent directory of the training folder
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from models.autoencoder import fMRIAutoencoder
from models.diffusion import diffusion_model  # Import the simplified diffusion model
from diffusers import DDPMScheduler
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

# Load trained autoencoder
autoencoder = fMRIAutoencoder().to(DEVICE)
if os.path.exists(AUTOENCODER_CHECKPOINT):
    autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT, map_location=DEVICE))
    print("✅ Loaded trained autoencoder model")
autoencoder.eval()  # Autoencoder should remain frozen

# Initialize diffusion model
diffusion_model = diffusion_model.to(DEVICE)

# Load checkpoint for the diffusion model if available
LAST_DIFFUSION_PATH = os.path.join(CHECKPOINT_DIR, "diffusion_model_last.pth")
if os.path.exists(DIFFUSION_CHECKPOINT):
    diffusion_model.load_state_dict(torch.load(DIFFUSION_CHECKPOINT, map_location=DEVICE))
    print("✅ Loaded best diffusion model checkpoint")
elif os.path.exists(LAST_DIFFUSION_PATH):
    diffusion_model.load_state_dict(torch.load(LAST_DIFFUSION_PATH, map_location=DEVICE))
    print("✅ Loaded last saved diffusion model checkpoint")

# Define loss, optimizer, and scheduler
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(diffusion_model.parameters(), lr=LEARNING_RATE)
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Initialize FMRIDataModule
data_module = FMRIDataModule(
    train_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/train.csv',
    val_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/val.csv',
    test_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/test.csv',
    data_dir=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri',
    batch_size=BATCH_SIZE,
    num_workers=16
)

# Call setup before using dataloaders
data_module.setup()

# Define dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

best_loss = float('inf')
patience = 10
counter = 0

spatial_dim = int(256 ** 0.5)  # 16

print(f"\n🔥 Starting Training for {EPOCHS_DIFFUSION} Epochs on {DEVICE}... 🔥\n")

for epoch in range(1, EPOCHS_DIFFUSION + 1):
    # Training Phase
    diffusion_model.train()
    train_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS_DIFFUSION}", leave=True)
    
    for batch_fmri, batch_labels in progress_bar:
        batch_fmri = batch_fmri.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        with torch.no_grad():
            latent = autoencoder.encoder(batch_fmri)  # Extract latent space
        latent = latent.view(latent.shape[0], 1, spatial_dim, spatial_dim)

        noise = torch.randn_like(latent).to(DEVICE)
        timesteps = torch.randint(0, 1000, (latent.shape[0],), device=DEVICE).long()
        noisy_latent = scheduler.add_noise(latent, noise, timesteps)

        # Pass labels to the model for conditioning
        predicted_noise = diffusion_model(noisy_latent, timesteps, batch_labels)

        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)
    print(f"\n📊 Epoch {epoch} - Avg Train Loss: {avg_train_loss:.6f}")

    # Validation Phase
    diffusion_model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_fmri, batch_labels in val_loader:
            batch_fmri = batch_fmri.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            latent = autoencoder.encoder(batch_fmri)
            latent = latent.view(latent.shape[0], 1, spatial_dim, spatial_dim)

            noise = torch.randn_like(latent).to(DEVICE)
            timesteps = torch.randint(0, 1000, (latent.shape[0],), device=DEVICE).long()
            noisy_latent = scheduler.add_noise(latent, noise, timesteps)

            # Pass labels to the model for conditioning
            predicted_noise = diffusion_model(noisy_latent, timesteps, batch_labels)

            loss = criterion(predicted_noise, noise)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"📊 Epoch {epoch} - Avg Validation Loss: {avg_val_loss:.6f}")

    # Save last checkpoint
    torch.save(diffusion_model.state_dict(), LAST_DIFFUSION_PATH)
    print("💾 Last model checkpoint saved.")

    # Save best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        counter = 0
        torch.save(diffusion_model.state_dict(), DIFFUSION_CHECKPOINT)
        print("✅ Best model updated!")
    else:
        counter += 1
        print(f"⏳ No Improvement - Patience {counter}/{patience}")

        if counter >= patience:
            print("🚀 Early stopping triggered - Training Complete! ✅")
            break

print(f"✅ Training complete. Best model saved at: {DIFFUSION_CHECKPOINT}")