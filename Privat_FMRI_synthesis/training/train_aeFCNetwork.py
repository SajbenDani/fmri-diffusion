import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Add parent dir to path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from models.autoencoder import Improved3DAutoencoder
from utils.dataset import FMRIDataModule
from config import (
    DEVICE, BATCH_SIZE, EPOCHS_AUTOENCODER, LEARNING_RATE, PATIENCE, NUM_WORKERS,
    PREFETCH_FACTOR, TRAIN_CSV, VAL_CSV, DATA_DIR, BEST_MODEL_PATH,
    W_MSE, W_L1, W_SSIM
)

# Load model
model = Improved3DAutoencoder().to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

# Freeze convolutional layers
for name, param in model.named_parameters():
    if any(k in name for k in ['enc_conv', 'enc_norm', 'dec_conv', 'dec_norm', 'label_embedding']):
        param.requires_grad = False

# Optimizer (only train unfrozen layers, e.g. FC layers)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# Loss functions
mse_loss_fn = nn.MSELoss()
l1_loss_fn = nn.L1Loss()
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# Data module
data_module = FMRIDataModule(
    train_csv=TRAIN_CSV,
    val_csv=VAL_CSV,
    test_csv=VAL_CSV,
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR
)
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Early stopping variables
best_val_loss = float('inf')
patience_counter = 0

# Training loop
model.train()
for epoch in range(EPOCHS_AUTOENCODER):
    total_train_loss = 0.0
    for fmri_tensor, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS_AUTOENCODER}"):
        fmri_tensor = fmri_tensor.to(DEVICE)

        optimizer.zero_grad()

        # Encode - frozen conv layers
        with torch.no_grad():
            e1 = nn.functional.leaky_relu(model.enc_norm1(model.enc_conv1(fmri_tensor)), 0.2)
            e2 = nn.functional.leaky_relu(model.enc_norm2(model.enc_conv2(e1)), 0.2)
            e3 = nn.functional.leaky_relu(model.enc_norm3(model.enc_conv3(e2)), 0.2)
            flattened = e3.view(e3.size(0), -1)

        # Trainable FC layers
        encoded = nn.functional.leaky_relu(model.enc_fc1(flattened), 0.2)
        encoded = model.enc_dropout(encoded)
        z = model.enc_fc2(encoded)

        # Decoder FC
        d = nn.functional.leaky_relu(model.dec_fc1(z), 0.2)
        d = model.dec_dropout(d)
        reconstructed_flattened = nn.functional.leaky_relu(model.dec_fc2(d), 0.2)

        # Composite loss
        mse_loss = mse_loss_fn(reconstructed_flattened, flattened)
        l1_loss = l1_loss_fn(reconstructed_flattened, flattened)
        loss = W_MSE * mse_loss + W_L1 * l1_loss  # SSIM skipped (not suitable on flattened)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS_AUTOENCODER}, Avg Train Loss: {avg_train_loss:.6f}")

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for fmri_tensor, _ in tqdm(val_loader, desc="Validating"):
            fmri_tensor = fmri_tensor.to(DEVICE)

            # Encode
            e1 = nn.functional.leaky_relu(model.enc_norm1(model.enc_conv1(fmri_tensor)), 0.2)
            e2 = nn.functional.leaky_relu(model.enc_norm2(model.enc_conv2(e1)), 0.2)
            e3 = nn.functional.leaky_relu(model.enc_norm3(model.enc_conv3(e2)), 0.2)
            flattened = e3.view(e3.size(0), -1)

            # FC forward pass
            encoded = nn.functional.leaky_relu(model.enc_fc1(flattened), 0.2)
            encoded = model.enc_dropout(encoded)
            z = model.enc_fc2(encoded)

            # Decoder FC
            d = nn.functional.leaky_relu(model.dec_fc1(z), 0.2)
            d = model.dec_dropout(d)
            reconstructed_flattened = nn.functional.leaky_relu(model.dec_fc2(d), 0.2)

            # Composite loss
            mse_loss = mse_loss_fn(reconstructed_flattened, flattened)
            l1_loss = l1_loss_fn(reconstructed_flattened, flattened)
            loss = W_MSE * mse_loss + W_L1 * l1_loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS_AUTOENCODER}, Avg Validation Loss: {avg_val_loss:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(os.path.dirname(BEST_MODEL_PATH), "Autoencoder_best_FC.pth"))
        print(f" Saved: Autoencoder_best_FC.pth (Val Loss: {avg_val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f" Early stopping triggered after {PATIENCE} epochs with no improvement.")
            break

print(" Training completed.")
