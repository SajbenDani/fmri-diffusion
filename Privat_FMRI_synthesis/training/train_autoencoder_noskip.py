#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PARENT_DIR))

# Import the models and datasets
from models.autoencoder_noskip import NoSkipAutoencoder
from utils.dataset import FMRIDataModule

# --- Configuration ---
OUTPUT_DIR = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_autoencoder"
PREPROCESSED_DIR = PARENT_DIR / "data_preprocessed"
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
VIEW = 'axial'

# Model hyperparameters
BASE_CHANNELS = 32
LATENT_CHANNELS = 16  # Increased from 8 to 16 for better encoding without skips
USE_VQ = True
NUM_VQ_EMBEDDINGS = 1024  # Increased from 512 to 1024 for more diversity
USE_POSITIONAL_ENCODING = True
BETA = 0.1  # Perceptual loss weight

# Setup logging
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = Path(OUTPUT_DIR) / f"training_noskip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
logger.info(f"Current Date and Time (UTC): 2025-08-08 06:41:23")
logger.info(f"Current User's Login: SajbenDani")

def compute_reconstruction_loss(recon, target):
    """Combined reconstruction loss (MSE + L1)"""
    # Ensure the reconstruction has the same size as the target
    if recon.shape != target.shape:
        recon = F.interpolate(recon, size=target.shape[2:], mode='trilinear', align_corners=False)
    
    mse = nn.MSELoss()(recon, target)
    mae = nn.L1Loss()(recon, target)
    return mse + 0.5 * mae

def main():
    logger.info("Initializing NoSkipAutoencoder training")
    logger.info(f"Latent channels: {LATENT_CHANNELS}, Base channels: {BASE_CHANNELS}")
    logger.info(f"Using VQ: {USE_VQ}, VQ Embeddings: {NUM_VQ_EMBEDDINGS}")
    logger.info(f"Using Positional Encoding: {USE_POSITIONAL_ENCODING}")
    
    # Initialize model
    model = NoSkipAutoencoder(
        in_channels=1,
        latent_channels=LATENT_CHANNELS,
        base_channels=BASE_CHANNELS,
        use_vq=USE_VQ,
        num_vq_embeddings=NUM_VQ_EMBEDDINGS,
        use_positional_encoding=USE_POSITIONAL_ENCODING
    ).to(device)
    
    # Setup optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    
    # Setup data loaders
    try:
        train_csv = os.path.join(PREPROCESSED_DIR, "train_patches.csv")
        val_csv = os.path.join(PREPROCESSED_DIR, "val_patches.csv")
        test_csv = os.path.join(PREPROCESSED_DIR, "test_patches.csv")
        
        logger.info(f"Loading datasets from {PREPROCESSED_DIR}...")
        data_module = FMRIDataModule(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            batch_size=BATCH_SIZE,
            num_workers=16,
            prefetch_factor=2,
            view=VIEW
        )
        
        data_module.setup()
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        logger.info(f"Successfully set up data module with {len(data_module.train_dataset)} training samples and {len(data_module.val_dataset)} validation samples")
        
    except Exception as e:
        logger.error(f"Error setting up data loaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Check for existing checkpoint to resume training
    last_checkpoint_path = os.path.join(OUTPUT_DIR, "Last_autoencoder_without_skips.pt")
    if os.path.exists(last_checkpoint_path):
        try:
            logger.info(f"Loading checkpoint from {last_checkpoint_path}")
            checkpoint = torch.load(last_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            logger.info(f"Resuming training from epoch {start_epoch} with best validation loss: {best_val_loss:.6f}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting training from scratch")
    
    # Record start time
    start_time = datetime.now()
    logger.info(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        recon_losses = 0.0
        vq_losses = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (images, _) in enumerate(pbar):
            # Move data to device
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if USE_VQ:
                recon, _, vq_loss = model(images)
            else:
                recon, _ = model(images)
                vq_loss = torch.tensor(0.0, device=device)
            
            # Compute reconstruction loss
            recon_loss = compute_reconstruction_loss(recon, images)
            
            # Combine losses
            total_loss = recon_loss
            if USE_VQ:
                total_loss += vq_loss
            
            # Backward pass and optimize
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update statistics
            train_loss += total_loss.item()
            recon_losses += recon_loss.item()
            if USE_VQ:
                vq_losses += vq_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(), 
                'recon_loss': recon_loss.item(),
                'vq_loss': vq_loss.item() if USE_VQ else 0
            })
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_recon_loss = recon_losses / len(train_loader)
        avg_vq_loss = vq_losses / len(train_loader) if USE_VQ else 0
        
        logger.info(f"Epoch {epoch+1}/{EPOCHS}:")
        logger.info(f"  Train Loss: {avg_train_loss:.6f}")
        logger.info(f"  Recon Loss: {avg_recon_loss:.6f}")
        if USE_VQ:
            logger.info(f"  VQ Loss: {avg_vq_loss:.6f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_vq_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(val_loader, desc="Validation")):
                # Move data to device
                images = images.to(device)
                
                # Forward pass
                if USE_VQ:
                    recon, _, vq_loss = model(images)
                else:
                    recon, _ = model(images)
                    vq_loss = torch.tensor(0.0, device=device)
                
                # Compute reconstruction loss
                recon_loss = compute_reconstruction_loss(recon, images)
                
                # Combine losses
                total_loss = recon_loss
                if USE_VQ:
                    total_loss += vq_loss
                
                # Update statistics
                val_loss += total_loss.item()
                val_recon_loss += recon_loss.item()
                if USE_VQ:
                    val_vq_loss += vq_loss.item()
        
        # Calculate average validation losses
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_vq_loss = val_vq_loss / len(val_loader) if USE_VQ else 0
        
        logger.info(f"  Validation Loss: {avg_val_loss:.6f}")
        logger.info(f"  Validation Recon Loss: {avg_val_recon_loss:.6f}")
        if USE_VQ:
            logger.info(f"  Validation VQ Loss: {avg_val_vq_loss:.6f}")
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save last checkpoint
        last_checkpoint_path = os.path.join(OUTPUT_DIR, "Last_autoencoder_without_skips.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'latent_channels': LATENT_CHANNELS,
            'base_channels': BASE_CHANNELS,
            'use_vq': USE_VQ,
            'num_vq_embeddings': NUM_VQ_EMBEDDINGS,
            'use_positional_encoding': USE_POSITIONAL_ENCODING
        }, last_checkpoint_path)
        
        logger.info(f"  Saved latest model checkpoint to {last_checkpoint_path}")
        
        # Save best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            best_checkpoint_path = os.path.join(OUTPUT_DIR, "Best_autoencoder_without_skips.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'latent_channels': LATENT_CHANNELS,
                'base_channels': BASE_CHANNELS,
                'use_vq': USE_VQ,
                'num_vq_embeddings': NUM_VQ_EMBEDDINGS,
                'use_positional_encoding': USE_POSITIONAL_ENCODING
            }, best_checkpoint_path)
            
            logger.info(f"  Saved best model checkpoint to {best_checkpoint_path} (Val Loss: {avg_val_loss:.6f})")

    # Calculate total training time
    end_time = datetime.now()
    training_time = end_time - start_time
    logger.info(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total training time: {training_time}")
    logger.info("Training complete!")

if __name__ == "__main__":
    main()