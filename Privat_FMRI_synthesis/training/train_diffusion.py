import torch
import torch.optim as optim
import os
import sys
import numpy as np
import random
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import StructuralSimilarityIndexMeasure

# Get the parent directory of the training folder
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from models.autoencoder import fMRIAutoencoder
from models.diffusion import diffusion_model
from diffusers import DDPMScheduler
from utils.dataset import FMRIDataModule
from config import *

# Set random seed
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
autoencoder.eval()

# Initialize diffusion model
diffusion_model = diffusion_model.to(DEVICE)
LAST_DIFFUSION_PATH = os.path.join(CHECKPOINT_DIR, "diffusion_model_last.pth")
if os.path.exists(DIFFUSION_CHECKPOINT):
    diffusion_model.load_state_dict(torch.load(DIFFUSION_CHECKPOINT, map_location=DEVICE))
    print("✅ Loaded best diffusion model checkpoint")
elif os.path.exists(LAST_DIFFUSION_PATH):
    diffusion_model.load_state_dict(torch.load(LAST_DIFFUSION_PATH, map_location=DEVICE))
    print("✅ Loaded last saved diffusion model checkpoint")

diffusion_model.train()

# Define loss functions and optimizer
mse_loss = torch.nn.MSELoss()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
fid = FrechetInceptionDistance(normalize=True).to(DEVICE)
optimizer = optim.Adam(diffusion_model.parameters(), lr=LEARNING_RATE)
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Initialize FMRIDataModule
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

# Loss weight coefficients
lambda_diffusion = 0.7
lambda_latent = 0.15
lambda_recon = 0.15

best_loss = float('inf')
patience = 10
counter = 0
spatial_dim = int(256 ** 0.5)

print(f"\n🔥 Starting Training for {EPOCHS_DIFFUSION} Epochs on {DEVICE}... 🔥\n")

for epoch in range(1, EPOCHS_DIFFUSION + 1):
    diffusion_model.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS_DIFFUSION}")
    
    for batch_fmri, batch_labels in progress_bar:
        batch_fmri = batch_fmri.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)
        
        with torch.no_grad():
            latent = autoencoder.encoder(batch_fmri)
        latent = latent.view(latent.shape[0], 1, spatial_dim, spatial_dim)

        noise = torch.randn_like(latent).to(DEVICE)
        timesteps = torch.randint(0, 1000, (latent.shape[0],), device=DEVICE).long()
        noisy_latent = scheduler.add_noise(latent, noise, timesteps)
        
        predicted_noise = diffusion_model(noisy_latent, timesteps, batch_labels)
        
        L_diffusion = 0.8 * mse_loss(predicted_noise, noise) + 0.2 * (1 - ssim(predicted_noise, noise))
        L_latent = mse_loss(autoencoder.encoder(autoencoder.decoder(latent)), latent)
        L_recon = mse_loss(autoencoder.decoder(latent), batch_fmri)
        
        L_total = lambda_diffusion * L_diffusion + lambda_latent * L_latent + lambda_recon * L_recon

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        train_loss += L_total.item()
        progress_bar.set_postfix(loss=f"{L_total.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)
    print(f"\n📊 Epoch {epoch} - Avg Train Loss: {avg_train_loss:.6f}")

    # Validation
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

            predicted_noise = diffusion_model(noisy_latent, timesteps, batch_labels)

            L_diffusion = 0.8 * mse_loss(predicted_noise, noise) + 0.2 * (1 - ssim(predicted_noise, noise))
            L_latent = mse_loss(autoencoder.encoder(autoencoder.decoder(latent)), latent)
            L_recon = mse_loss(autoencoder.decoder(latent), batch_fmri)
            
            L_total = lambda_diffusion * L_diffusion + lambda_latent * L_latent + lambda_recon * L_recon
            val_loss += L_total.item()
            fid.update(predicted_noise, real=False)
            fid.update(noise, real=True)
    
    avg_val_loss = val_loss / len(val_loader)
    fid_score = fid.compute()
    fid.reset()
    print(f"📊 Epoch {epoch} - Avg Validation Loss: {avg_val_loss:.6f} - FID: {fid_score:.4f}")
    
    torch.save(diffusion_model.state_dict(), LAST_DIFFUSION_PATH)
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        counter = 0
        torch.save(diffusion_model.state_dict(), DIFFUSION_CHECKPOINT)
    else:
        counter += 1
        if counter >= patience:
            print("🚀 Early stopping triggered!")
            break
