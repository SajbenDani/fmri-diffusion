import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
from models.autoencoder import fMRIAutoencoder
from utils.dataset import get_dataloader
from config import *

# Initialize model
autoencoder = fMRIAutoencoder().to(DEVICE)

# Load checkpoint if available
if os.path.exists(AUTOENCODER_CHECKPOINT):
    autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT))
    print("Loaded checkpoint")

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)


# DataLoader
if __name__ == "__main__":
    dataloader = get_dataloader("data/new_format_config/train.csv", "data", BATCH_SIZE)


# Early stopping
best_loss = float('inf')
patience = 5
counter = 0

print(f"Training for {EPOCHS_AUTOENCODER} epochs...\n")

for epoch in range(EPOCHS_AUTOENCODER):
    epoch_loss = 0
    print(f"Epoch {epoch + 1}/{EPOCHS_AUTOENCODER}")
    for i, batch in enumerate(dataloader):
        fmri_tensor, label = batch  # Unpack tuple
        fmri_tensor = fmri_tensor.to(DEVICE)
        #batch = batch.to(DEVICE)
        
        # this below would solve the issue with that was solve by avearging the time dimension
        # Flatten the first two dimensions: (B, T, 1, D, H, W) -> (B*T, 1, D, H, W)
        # B, T, C, D, H, W = fmri_tensor.shape
        # fmri_tensor = fmri_tensor.view(B * T, C, D, H, W)

        optimizer.zero_grad()
        recon = autoencoder(fmri_tensor)  # Use fmri_tensor instead of batch
        loss = criterion(recon, fmri_tensor)  # Compare reconstruction with input
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Progress indicator
        # sys.stdout.write(f"\rBatch {i + 1}/{len(dataloader)} - Loss: {loss.item():.4f}")
        # sys.stdout.flush()

        # Print batch loss without overwriting previous output
        print(f"Batch {i + 1}/{len(dataloader)} - Loss: {loss.item():.4f}")

    epoch_loss /= len(dataloader)
    # print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    print(f"Epoch {epoch + 1} completed - Avg Loss: {epoch_loss:.4f}\n")
    
    # Checkpointing
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        torch.save(autoencoder.state_dict(), AUTOENCODER_CHECKPOINT)
        print("Checkpoint saved")
    else:
        counter += 1
        print(f"Patience counter: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping triggered")
            break
