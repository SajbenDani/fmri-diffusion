#!/usr/bin/env python3
"""
3D Autoencoder Training Module for fMRI Latent Space Learning.

This module implements the complete training pipeline for the 3D autoencoder
that learns compressed latent representations of functional MRI (fMRI) data.
The autoencoder serves as the foundation for the diffusion model pipeline,
creating an efficient latent space for downstream generation tasks.

Training Objectives:
    1. Learn to compress 3D fMRI volumes into compact latent representations
    2. Achieve high-fidelity reconstruction from compressed latents
    3. Optionally learn discrete latent spaces through Vector Quantization (VQ)
    4. Create stable latent representations suitable for diffusion modeling

Key Features:
    - Mixed precision training for memory efficiency and speed
    - Advanced loss functions including MS-SSIM for perceptual quality
    - Vector quantization support for discrete latent spaces
    - Comprehensive validation and checkpointing
    - Memory-optimized batch processing
    - Robust training with early stopping

Architecture Training Strategy:
    The autoencoder is trained in a standard reconstruction paradigm:
    - Forward: Input Volume -> Encoder -> Latent -> Decoder -> Reconstruction
    - Loss: Combination of reconstruction loss + optional VQ regularization
    - Validation: Monitor reconstruction quality on held-out data
    - Checkpointing: Save best model based on validation performance

Loss Components:
    1. Reconstruction Loss: MSE between input and reconstructed volumes
    2. Perceptual Loss (optional): MS-SSIM for structural similarity
    3. VQ Loss (optional): Codebook and commitment losses for quantization
    4. Regularization: Weight decay for model complexity control

Training Phases:
    - Phase 1: Initial convergence with high learning rate
    - Phase 2: Fine-tuning with reduced learning rate (early stopping)
    - Phase 3: Optional VQ codebook optimization

Usage:
    Direct execution:
    ```bash
    python train_autoencoder.py
    ```
    
    Programmatic usage:
    ```python
    from training.train_autoencoder import train_autoencoder
    model = train_autoencoder(config)
    ```

Output:
    - Trained autoencoder model checkpoint
    - Training logs with loss curves and metrics
    - Validation performance statistics
    - Model architecture summaries

Requirements:
    - Preprocessed fMRI patch datasets (train/val/test splits)
    - CUDA GPU recommended (>8GB VRAM for batch_size=16)
    - Optional: pytorch-msssim for advanced perceptual losses
"""

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
from typing import Dict, Tuple, Optional, Any

# Add parent directory to path for relative imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

# Import custom modules
from models.autoencoder import Improved3DAutoencoder
from utils.dataset import FMRIDataModule

# =============================================================================
# OPTIONAL ADVANCED LOSS FUNCTIONS
# =============================================================================

# MS-SSIM configuration - provides better perceptual reconstruction quality
USE_MSSSIM = False  # Enable for perceptual loss (requires additional dependency)
HAS_MSSSIM = False  # Will be set based on import success

try:
    if USE_MSSSIM:
        from pytorch_msssim import MS_SSIM
        HAS_MSSSIM = True
        logging.info("MS-SSIM loss function available")
except ImportError:
    logging.warning("pytorch-msssim not found. Install with: pip install pytorch-msssim")
    logging.warning("Falling back to MSE-only loss function")
    HAS_MSSSIM = False

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("autoencoder_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Model training hyperparameters
BATCH_SIZE = 16               # Batch size (adjust based on GPU memory)
LEARNING_RATE = 1e-4          # Initial learning rate for Adam optimizer
EPOCHS = 100                  # Maximum training epochs
PATIENCE = 10                 # Early stopping patience (epochs without improvement)
SEED = 42                     # Random seed for reproducibility

# Hardware and optimization settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = True if torch.cuda.is_available() else False  # AMP for efficiency
USE_VQ = True                 # Enable Vector Quantization for discrete latents

# Model architecture parameters
LATENT_CHANNELS = 8           # Dimension of latent space
BASE_CHANNELS = 32            # Base channel count for autoencoder
VQ_EMBEDDINGS = 512           # Size of VQ codebook if VQ enabled

# Loss function weights
RECONSTRUCTION_WEIGHT = 1.0   # Weight for MSE reconstruction loss
VQ_WEIGHT = 0.1              # Weight for VQ regularization loss
PERCEPTUAL_WEIGHT = 0.1      # Weight for MS-SSIM perceptual loss (if available)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducible training across multiple runs.
    
    This function ensures deterministic behavior across PyTorch, NumPy,
    and Python's random module. Critical for comparing different model
    configurations and reproducing experimental results.
    
    Args:
        seed (int): Random seed value
        
    Note:
        Even with seed setting, some CUDA operations may introduce
        non-determinism for performance reasons. For complete determinism,
        additional CUDA flags may be needed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For complete reproducibility (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def compute_reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor, 
                              use_perceptual: bool = False) -> torch.Tensor:
    """
    Compute comprehensive reconstruction loss between predicted and target volumes.
    
    This function computes a combination of pixel-wise and perceptual losses
    to encourage both accurate reconstruction and visual quality. The loss
    combination helps the model learn meaningful latent representations.
    
    Args:
        reconstruction (torch.Tensor): Reconstructed fMRI volume from decoder
            Shape: (batch_size, channels, depth, height, width)
        target (torch.Tensor): Ground truth fMRI volume
            Shape: same as reconstruction
        use_perceptual (bool): Whether to include MS-SSIM perceptual loss
        
    Returns:
        torch.Tensor: Combined reconstruction loss (scalar)
        
    Loss Components:
        1. MSE Loss: Pixel-wise reconstruction accuracy L2(recon, target)
        2. MS-SSIM Loss (optional): Multi-scale structural similarity
        
    The combination ensures both accurate reconstruction and preservation
    of important structural patterns in the fMRI data.
    """
    # Primary reconstruction loss: Mean Squared Error
    mse_loss = nn.functional.mse_loss(reconstruction, target)
    total_loss = RECONSTRUCTION_WEIGHT * mse_loss
    
    # Optional perceptual loss using Multi-Scale SSIM
    if use_perceptual and HAS_MSSSIM:
        try:
            # MS-SSIM requires specific data format and normalization
            # Ensure values are in [0,1] range
            recon_norm = torch.clamp((reconstruction - reconstruction.min()) / 
                                   (reconstruction.max() - reconstruction.min() + 1e-8), 0, 1)
            target_norm = torch.clamp((target - target.min()) / 
                                    (target.max() - target.min() + 1e-8), 0, 1)
            
            # Initialize MS-SSIM loss function if not exists
            if not hasattr(compute_reconstruction_loss, 'ms_ssim_loss'):
                compute_reconstruction_loss.ms_ssim_loss = MS_SSIM(
                    data_range=1.0, 
                    size_average=True, 
                    channel=reconstruction.shape[1]
                ).to(reconstruction.device)
            
            # Compute MS-SSIM loss (1 - MS-SSIM for loss minimization)
            ms_ssim_val = compute_reconstruction_loss.ms_ssim_loss(recon_norm, target_norm)
            ms_ssim_loss = 1.0 - ms_ssim_val
            
            total_loss += PERCEPTUAL_WEIGHT * ms_ssim_loss
            
        except Exception as e:
            logger.warning(f"MS-SSIM computation failed, using MSE only: {e}")
    
    return total_loss


def validate_model(model: nn.Module, val_loader: torch.utils.data.DataLoader, 
                  device: torch.device) -> Dict[str, float]:
    """
    Perform comprehensive model validation on held-out data.
    
    This function evaluates the trained autoencoder on validation data to
    monitor training progress and detect overfitting. It computes multiple
    metrics to assess both reconstruction quality and latent space properties.
    
    Args:
        model (nn.Module): Trained autoencoder model
        val_loader (DataLoader): Validation data loader
        device (torch.device): Computation device
        
    Returns:
        Dict[str, float]: Validation metrics
            - 'val_loss': Average validation loss
            - 'val_mse': Average MSE reconstruction error
            - 'val_vq_loss': Average VQ loss (if applicable)
            - 'val_perplexity': VQ codebook usage (if applicable)
            
    Process:
        1. Set model to evaluation mode (disable dropout, etc.)
        2. Process validation batches without gradient computation
        3. Compute reconstruction and regularization losses
        4. Aggregate statistics across all validation samples
        5. Return comprehensive metrics dictionary
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_vq_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (fmri_data, labels) in enumerate(tqdm(val_loader, desc="Validation")):
            try:
                # Move data to device
                fmri_data = fmri_data.to(device, non_blocking=True)
                batch_size = fmri_data.size(0)
                
                # Forward pass through autoencoder
                if USE_VQ:
                    reconstruction, latent, vq_loss = model(fmri_data)
                    
                    # Compute losses
                    recon_loss = compute_reconstruction_loss(
                        reconstruction, fmri_data, use_perceptual=USE_MSSSIM
                    )
                    total_loss += (recon_loss + VQ_WEIGHT * vq_loss).item() * batch_size
                    total_vq_loss += vq_loss.item() * batch_size
                    
                else:
                    reconstruction, latent = model(fmri_data)
                    recon_loss = compute_reconstruction_loss(
                        reconstruction, fmri_data, use_perceptual=USE_MSSSIM
                    )
                    total_loss += recon_loss.item() * batch_size
                
                # Compute MSE for monitoring (always computed)
                mse = nn.functional.mse_loss(reconstruction, fmri_data)
                total_mse += mse.item() * batch_size
                total_samples += batch_size
                
            except Exception as e:
                logger.warning(f"Validation batch {batch_idx} failed: {e}")
                continue
    
    # Compute average metrics
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_mse = total_mse / total_samples
        avg_vq_loss = total_vq_loss / total_samples if USE_VQ else 0.0
        
        metrics = {
            'val_loss': avg_loss,
            'val_mse': avg_mse,
            'val_vq_loss': avg_vq_loss
        }
        
        logger.info(f"Validation - Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}")
        if USE_VQ:
            logger.info(f"Validation - VQ Loss: {avg_vq_loss:.6f}")
            
        return metrics
    else:
        logger.error("No valid validation samples processed")
        return {'val_loss': float('inf'), 'val_mse': float('inf'), 'val_vq_loss': 0.0}
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