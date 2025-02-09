import torch
import torch.optim as optim
import torch.nn as nn
import os
import math
from tqdm import tqdm  # Progress bar
from models.autoencoder import fMRIAutoencoder
from models.diffusion import diffusion_model
from diffusers import DDPMScheduler
from utils.dataset import get_dataloader
from config import *

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained autoencoder
autoencoder = fMRIAutoencoder().to(DEVICE)
autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT, map_location=DEVICE))
autoencoder.eval()

# Load checkpoint for the diffusion model if available
if os.path.exists(DIFFUSION_CHECKPOINT):
    diffusion_model.load_state_dict(torch.load(DIFFUSION_CHECKPOINT, map_location=DEVICE))
    print("Loaded diffusion checkpoint")

# Move diffusion model to the correct device
diffusion_model.to(DEVICE)

# Loss, optimizer, and noise scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(diffusion_model.parameters(), lr=LEARNING_RATE)
scheduler = DDPMScheduler(num_train_timesteps=1000)
dataloader = get_dataloader("data/new_format_config/train.csv", "data", BATCH_SIZE)

best_loss = float('inf')
patience = 5
counter = 0

# We expect the autoencoder latent to be of size 256, so we reshape it to (batch, 1, 16, 16)
spatial_dim = int(256 ** 0.5)  # 16

print(f"\n🔥 Starting Training for {EPOCHS_DIFFUSION} Epochs on {DEVICE}... 🔥\n")

for epoch in range(1, EPOCHS_DIFFUSION + 1):
    epoch_loss = 0
    batch_losses = []  # Track batch losses

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS_DIFFUSION}", leave=True)

    for batch_fmri, batch_labels in progress_bar:
        batch_fmri = batch_fmri.to(DEVICE)  # Move fMRI tensor to GPU/CPU
        batch_labels = batch_labels.to(DEVICE)  # Move labels to GPU/CPU if needed

        with torch.no_grad():
            # Get latent vector from the autoencoder: shape [batch, 256]
            latent = autoencoder.encoder(batch_fmri)  # Use only the fMRI tensor
        
        # Reshape the latent vector to a 2D map: [batch, 1, 16, 16]
        latent = latent.view(latent.shape[0], 1, spatial_dim, spatial_dim)
        
        noise = torch.randn_like(latent).to(DEVICE)
        timesteps = torch.randint(0, 1000, (latent.shape[0],), device=DEVICE).long()
        noisy_latent = scheduler.add_noise(latent, noise, timesteps)
        
        # Forward pass through the diffusion model
        output = diffusion_model(noisy_latent, timesteps)
        # Depending on the diffusers version, the output might be wrapped in an object.
        predicted_noise = output.sample if hasattr(output, 'sample') else output
        
        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_losses.append(loss.item())

        # Update progress bar description with current batch loss
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Compute average epoch loss
    avg_epoch_loss = epoch_loss / len(dataloader)

    print(f"\n📊 Epoch {epoch} Completed - Avg Loss: {avg_epoch_loss:.6f}")

    # Check for best loss improvement
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        counter = 0
        torch.save(diffusion_model.state_dict(), DIFFUSION_CHECKPOINT)
        print("✅ Checkpoint saved (New Best Loss)")
    else:
        counter += 1
        print(f"⏳ No Improvement - Patience {counter}/{patience}")

        if counter >= patience:
            print("🚀 Early stopping triggered - Training Complete! ✅")
            break
