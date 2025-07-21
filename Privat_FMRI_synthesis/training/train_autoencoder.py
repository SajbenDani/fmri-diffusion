#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
import logging
import time
from pathlib import Path

# Add parent directory to path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

# Import the model and dataset
from models.autoencoder import Improved3DAutoencoder
from utils.dataset import FMRIDataModule

USE_MSSSIM = False  # Define this outside the try block
HAS_MSSSIM = False  # Default to False

try:
    if USE_MSSSIM:  # Only import if we want to use it
        from pytorch_msssim import MS_SSIM
        HAS_MSSSIM = True
except ImportError:
    print("Warning: pytorch-msssim not found. Install with: pip install pytorch-msssim")
    HAS_MSSSIM = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Config
BATCH_SIZE = 16  # Can use larger batch size with optimized pipeline
LEARNING_RATE = 1e-4
EPOCHS = 100
PATIENCE = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = True if torch.cuda.is_available() else False
USE_VQ = True

# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_autoencoder():
    # Set random seed
    set_seed(SEED)
    
    # Paths to preprocessed data
    preprocessed_dir = os.path.join(PARENT_DIR, "data_preprocessed")
    
    # Checkpoint directory
    checkpoint_dir = os.path.join(PARENT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize MS-SSIM if available
    if HAS_MSSSIM:
        msssim_module = MS_SSIM(
            data_range=1.0,
            win_size=7,
            win_sigma=1.5,
            channel=1,
            spatial_dims=3,
            K=(0.01, 0.03)
        ).to(DEVICE)
        
        def msssim_loss(x, y):
            return 1 - msssim_module(x, y)
    else:
        def msssim_loss(x, y):
            return torch.tensor(0.0, device=x.device)
    
    # Create data module with patch dataset
    data_module = FMRIDataModule(
        train_csv=os.path.join(preprocessed_dir, "train_patches.csv"),
        val_csv=os.path.join(preprocessed_dir, "val_patches.csv"),
        test_csv=os.path.join(preprocessed_dir, "test_patches.csv"),
        batch_size=BATCH_SIZE,
        num_workers=16,  # Can use more workers with optimized dataset
        prefetch_factor=4,
        view='axial'
    )
    
    # Setup data loaders
    logger.info("Setting up data loaders...")
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Create model
    logger.info(f"Creating Improved 3D Autoencoder on {DEVICE}...")
    model = Improved3DAutoencoder(
        in_channels=1,
        latent_channels=8,
        base_channels=32,
        use_vq=USE_VQ
    ).to(DEVICE)
    
    # Define loss functions
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    
    # Set up optimizer and scaler for mixed precision
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') if USE_MIXED_PRECISION else None
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_ssim_loss = 0.0
        train_vq_loss = 0.0
        
        # Track batch processing time
        batch_times = []
        data_times = []
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")):
            batch_start = time.time()
            data_load_time = batch_start - (batch_times[-1] if batch_times else start_time)
            data_times.append(data_load_time)
            
            data = data.to(DEVICE)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            if USE_MIXED_PRECISION:
                with torch.amp.autocast('cuda'):
                    if USE_VQ:
                        recon, _, vq_loss = model(data)
                    else:
                        recon, _ = model(data)
                        vq_loss = 0
                    
                    # Compute losses
                    mse_loss = mse_criterion(recon, data)
                    l1_loss = l1_criterion(recon, data)
                    ssim_loss = msssim_loss(recon, data) if HAS_MSSSIM else torch.tensor(0.0, device=DEVICE)
                    
                    # Combined loss
                    loss = 0.5 * mse_loss + 0.3 * l1_loss
                    if HAS_MSSSIM:
                        loss += 0.2 * ssim_loss
                    if USE_VQ:
                        loss += 0.1 * vq_loss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                if USE_VQ:
                    recon, _, vq_loss = model(data)
                else:
                    recon, _ = model(data)
                    vq_loss = 0
                
                # Compute losses
                mse_loss = mse_criterion(recon, data)
                l1_loss = l1_criterion(recon, data)
                ssim_loss = msssim_loss(recon, data) if HAS_MSSSIM else torch.tensor(0.0, device=DEVICE)
                
                # Combined loss
                loss = 0.5 * mse_loss + 0.3 * l1_loss
                if HAS_MSSSIM:
                    loss += 0.2 * ssim_loss
                if USE_VQ:
                    loss += 0.1 * vq_loss
                
                # Standard backward pass
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += (mse_loss.item() + l1_loss.item()) / 2
            train_ssim_loss += ssim_loss.item() if HAS_MSSSIM else 0
            train_vq_loss += vq_loss.item() if USE_VQ else 0
            
            # Track batch processing time
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            # Print timing info every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_batch_time = sum(batch_times[-10:]) / 10
                avg_data_time = sum(data_times[-10:]) / 10
                # logger.info(f"Batch {batch_idx+1}: avg batch time {avg_batch_time:.4f}s, avg data time {avg_data_time:.4f}s")
        
        # Calculate average training losses
        num_batches = len(train_loader)
        train_loss /= num_batches
        train_recon_loss /= num_batches
        train_ssim_loss /= num_batches
        train_vq_loss /= num_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_ssim_loss = 0.0
        val_vq_loss = 0.0
        
        with torch.no_grad():
            for data, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                data = data.to(DEVICE)
                
                if USE_VQ:
                    recon, _, vq_loss = model(data)
                else:
                    recon, _ = model(data)
                    vq_loss = 0
                
                # Compute losses
                mse_loss = mse_criterion(recon, data)
                l1_loss = l1_criterion(recon, data)
                ssim_loss = msssim_loss(recon, data) if HAS_MSSSIM else torch.tensor(0.0, device=DEVICE)
                
                # Combined loss
                loss = 0.5 * mse_loss + 0.3 * l1_loss
                if HAS_MSSSIM:
                    loss += 0.2 * ssim_loss
                if USE_VQ:
                    loss += 0.1 * vq_loss
                
                val_loss += loss.item()
                val_recon_loss += (mse_loss.item() + l1_loss.item()) / 2
                val_ssim_loss += ssim_loss.item() if HAS_MSSSIM else 0
                val_vq_loss += vq_loss.item() if USE_VQ else 0
        
        # Calculate average validation losses
        num_val_batches = len(val_loader)
        val_loss /= num_val_batches
        val_recon_loss /= num_val_batches
        val_ssim_loss /= num_val_batches
        val_vq_loss /= num_val_batches
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_data_time = sum(data_times) / len(data_times) if data_times else 0
        
        logger.info(f"Epoch {epoch+1}/{EPOCHS} completed in {epoch_time:.2f}s")
        logger.info(f"Average batch processing time: {avg_batch_time:.4f}s")
        logger.info(f"Average data loading time: {avg_data_time:.4f}s")
        logger.info(f"Train Loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}, SSIM: {train_ssim_loss:.6f}, VQ: {train_vq_loss:.6f})")
        logger.info(f"Val Loss: {val_loss:.6f} (Recon: {val_recon_loss:.6f}, SSIM: {val_ssim_loss:.6f}, VQ: {val_vq_loss:.6f})")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_recon_loss': val_recon_loss,
                'val_ssim_loss': val_ssim_loss,
                'val_vq_loss': val_vq_loss,
            }, os.path.join(checkpoint_dir, 'best_autoencoder.pt'))
            
            logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}")
            
            # Early stopping
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(checkpoint_dir, 'latest_autoencoder.pt'))
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    train_autoencoder()