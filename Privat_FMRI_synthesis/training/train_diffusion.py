import torch
import torch.optim as optim
import os
import sys
import numpy as np
import random
from tqdm import tqdm

# Add parent directory to sys.path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from models.autoencoder import fMRIAutoencoder
from models.diffusion import diffusion_model
from diffusers import DDPMScheduler
from utils.dataset import FMRIDataModule
from config import *

# Set random seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create checkpoint dir if not exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load autoencoder
autoencoder = fMRIAutoencoder().to(DEVICE)
if os.path.exists(AUTOENCODER_CHECKPOINT):
    autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT, map_location=DEVICE))
    print("Loaded trained autoencoder model")
autoencoder.eval()

# Initialize diffusion model
diffusion_model = diffusion_model.to(DEVICE)

# Load diffusion checkpoints
if os.path.exists(DIFFUSION_CHECKPOINT):
    diffusion_model.load_state_dict(torch.load(DIFFUSION_CHECKPOINT, map_location=DEVICE))
    print("Loaded best diffusion model checkpoint")
elif os.path.exists(DIFFUSION_LAST_CHECKPOINT):
    diffusion_model.load_state_dict(torch.load(DIFFUSION_LAST_CHECKPOINT, map_location=DEVICE))
    print("Loaded last saved diffusion model checkpoint")

# Loss, optimizer, scheduler
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(diffusion_model.parameters(), lr=LEARNING_RATE)
scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)

# Load dataset
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

print(f"\n Starting Diffusion Training for {EPOCHS_DIFFUSION} Epochs on {DEVICE}... \n")

for epoch in range(1, EPOCHS_DIFFUSION + 1):
    # Training
    diffusion_model.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS_DIFFUSION}", leave=True)

    for batch_fmri, batch_labels in progress_bar:
        batch_fmri, batch_labels = batch_fmri.to(DEVICE), batch_labels.to(DEVICE)

        with torch.no_grad():
            latent = autoencoder.encoder(batch_fmri)
        latent = latent.view(latent.shape[0], 1, SPATIAL_DIM, SPATIAL_DIM)

        noise = torch.randn_like(latent).to(DEVICE)
        timesteps = torch.randint(0, NUM_TIMESTEPS, (latent.shape[0],), device=DEVICE).long()
        noisy_latent = scheduler.add_noise(latent, noise, timesteps)

        predicted_noise = diffusion_model(noisy_latent, timesteps, batch_labels)
        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)
    print(f"\n Epoch {epoch} - Avg Train Loss: {avg_train_loss:.6f}")

    # Validation
    diffusion_model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_fmri, batch_labels in val_loader:
            batch_fmri, batch_labels = batch_fmri.to(DEVICE), batch_labels.to(DEVICE)

            latent = autoencoder.encoder(batch_fmri)
            latent = latent.view(latent.shape[0], 1, SPATIAL_DIM, SPATIAL_DIM)

            noise = torch.randn_like(latent).to(DEVICE)
            timesteps = torch.randint(0, NUM_TIMESTEPS, (latent.shape[0],), device=DEVICE).long()
            noisy_latent = scheduler.add_noise(latent, noise, timesteps)

            predicted_noise = diffusion_model(noisy_latent, timesteps, batch_labels)
            loss = criterion(predicted_noise, noise)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f" Epoch {epoch} - Avg Validation Loss: {avg_val_loss:.6f}")

    # Save checkpoints
    torch.save(diffusion_model.state_dict(), DIFFUSION_LAST_CHECKPOINT)
    print(" Last model checkpoint saved.")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        counter = 0
        torch.save(diffusion_model.state_dict(), DIFFUSION_CHECKPOINT)
        print(" Best model updated!")
    else:
        counter += 1
        print(f"⏳ No Improvement - Patience {counter}/{PATIENCE}")
        if counter >= PATIENCE:
            print("🚀 Early stopping triggered - Training Complete! ")
            break

print(f" Training complete. Best model saved at: {DIFFUSION_CHECKPOINT}")
