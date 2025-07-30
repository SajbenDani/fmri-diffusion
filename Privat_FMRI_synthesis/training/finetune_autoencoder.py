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

# Add parent directory to path to find the models module
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PARENT_DIR))

# Import the models and datasets
from models.autoencoder import Improved3DAutoencoder
from utils.dataset import FMRIDataModule

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
CHECKPOINT_PATH = str(PARENT_DIR / "checkpoints_autoencoder" / "best_autoencoder.pt")
PREPROCESSED_DIR = str(PARENT_DIR / "data_preprocessed")
OUTPUT_DIR = str(PARENT_DIR / "checkpoints_finetuned")
BATCH_SIZE = 4  # Reduced batch size to avoid OOM
EPOCHS = 30
LEARNING_RATE = 5e-6
NO_SKIP_PROB = 0.2
NO_SKIP_WEIGHT = 0.5
USE_NO_SKIP_ANNEALING = True
VIEW = 'axial'
BASE_CHANNELS = 32
LATENT_CHANNELS = 8
USE_VQ = True

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_model_with_custom_mapping():
    """Load model with custom weight mapping for the modified architecture."""
    print(f"Loading pretrained model from {CHECKPOINT_PATH}")
    
    # Initialize the new model architecture
    model = Improved3DAutoencoder(
        in_channels=1,
        latent_channels=LATENT_CHANNELS,
        base_channels=BASE_CHANNELS,
        use_vq=USE_VQ
    ).to(device)
    
    try:
        # Load the old state dictionary
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        old_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Create a new state dictionary with correct mappings
        new_state_dict = {}
        
        # Copy weights for components that haven't changed
        for key in old_state_dict:
            # Skip the decoder blocks that have changed
            if any(x in key for x in ['dec1.res_block', 'dec2.res_block', 'dec3.res_block', 'final_upsample.res_block']):
                continue
            
            # Keep all other weights as they are
            new_state_dict[key] = old_state_dict[key]
        
        # Map old decoder block weights to new res_block_with_skip
        for block in ['dec1', 'dec2', 'dec3', 'final_upsample']:
            for param in ['conv1.weight', 'conv1.bias', 'norm1.weight', 'norm1.bias', 
                         'conv2.weight', 'conv2.bias', 'norm2.weight', 'norm2.bias']:
                old_key = f"{block}.res_block.{param}"
                new_key = f"{block}.res_block_with_skip.{param}"
                if old_key in old_state_dict:
                    new_state_dict[new_key] = old_state_dict[old_key]
            
            # Map skip connection weights if they exist
            for param in ['skip.0.weight', 'skip.0.bias', 'skip.1.weight', 'skip.1.bias']:
                old_key = f"{block}.res_block.{param}"
                new_key = f"{block}.res_block_with_skip.{param}"
                if old_key in old_state_dict:
                    new_state_dict[new_key] = old_state_dict[old_key]
            
            # Initialize res_block_no_skip with the same weights as res_block_with_skip
            # but with adjusted input dimensions
            for param in ['conv1.weight', 'conv1.bias', 'norm1.weight', 'norm1.bias',
                         'conv2.weight', 'conv2.bias', 'norm2.weight', 'norm2.bias']:
                old_key = f"{block}.res_block.{param}"
                new_key = f"{block}.res_block_no_skip.{param}"
                
                if old_key in old_state_dict:
                    # For weights, need to handle the input dimension difference
                    if 'weight' in param and 'conv1' in param:
                        # Get the original weight tensor
                        orig_weight = old_state_dict[old_key]
                        
                        # For the no_skip path, the input channels are fewer (no skip connection)
                        # So we need to resize the weight tensor
                        if block == 'dec3':
                            # dec3: base_channels*8 instead of base_channels*8 + base_channels*8
                            new_weight = orig_weight[:, :BASE_CHANNELS*8, :, :, :]
                        elif block == 'dec2':
                            # dec2: base_channels*4 instead of base_channels*4 + base_channels*4
                            new_weight = orig_weight[:, :BASE_CHANNELS*4, :, :, :]
                        elif block == 'dec1':
                            # dec1: base_channels*2 instead of base_channels*2 + base_channels*2
                            new_weight = orig_weight[:, :BASE_CHANNELS*2, :, :, :]
                        else:  # final_upsample
                            # final_upsample: base_channels instead of base_channels + base_channels
                            new_weight = orig_weight[:, :BASE_CHANNELS, :, :, :]
                        
                        new_state_dict[new_key] = new_weight
                    else:
                        # For biases and normalization, copy directly
                        new_state_dict[new_key] = old_state_dict[old_key]
                
            # Handle skip connection in the no_skip path
            for param in ['skip.0.weight', 'skip.0.bias', 'skip.1.weight', 'skip.1.bias']:
                old_key = f"{block}.res_block.{param}"
                new_key = f"{block}.res_block_no_skip.{param}"
                
                if old_key in old_state_dict:
                    if 'weight' in param and '0.weight' in param:
                        # Adjust the input dimension for the skip connection
                        orig_weight = old_state_dict[old_key]
                        if block == 'dec3':
                            new_weight = orig_weight[:, :BASE_CHANNELS*8, :, :]
                        elif block == 'dec2':
                            new_weight = orig_weight[:, :BASE_CHANNELS*4, :, :]
                        elif block == 'dec1':
                            new_weight = orig_weight[:, :BASE_CHANNELS*2, :, :]
                        else:  # final_upsample
                            new_weight = orig_weight[:, :BASE_CHANNELS, :, :]
                        
                        new_state_dict[new_key] = new_weight
                    else:
                        # Copy biases directly
                        new_state_dict[new_key] = old_state_dict[old_key]
        
        # Load the new state dict into the model
        model.load_state_dict(new_state_dict, strict=False)
        
        print("Model loaded successfully with custom weight mapping")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

# Loss functions
def compute_reconstruction_loss(recon, target):
    """Combined reconstruction loss (MSE + L1)"""
    # Ensure the reconstruction has the same size as the target
    if recon.shape != target.shape:
        recon = F.interpolate(recon, size=target.shape[2:], mode='trilinear', align_corners=False)
    
    mse = nn.MSELoss()(recon, target)
    mae = nn.L1Loss()(recon, target)
    return mse + 0.5 * mae

def main():
    # Load model with custom weight mapping
    model = load_model_with_custom_mapping()
    if model is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)

    model.to(device)
    model.eval()  # Start in eval mode

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Setup data loaders
    try:
        # Check if the preprocessed directory exists
        if not os.path.exists(PREPROCESSED_DIR):
            print(f"ERROR: Preprocessed directory not found at {PREPROCESSED_DIR}")
            print("Please check your project structure and update PREPROCESSED_DIR in the script.")
            sys.exit(1)
        
        train_csv = os.path.join(PREPROCESSED_DIR, "train_patches.csv")
        val_csv = os.path.join(PREPROCESSED_DIR, "val_patches.csv")
        test_csv = os.path.join(PREPROCESSED_DIR, "test_patches.csv")
        
        # Check if CSV files exist
        for csv_file, name in [(train_csv, "train"), (val_csv, "validation")]:
            if not os.path.exists(csv_file):
                print(f"ERROR: {name.capitalize()} CSV file not found at {csv_file}")
                print(f"Please make sure the {name} CSV file exists in {PREPROCESSED_DIR}")
                sys.exit(1)
        
        print(f"Loading datasets from {PREPROCESSED_DIR}...")
        data_module = FMRIDataModule(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            batch_size=BATCH_SIZE,
            num_workers=4,  # Reduced to avoid memory issues
            prefetch_factor=2,
            view=VIEW
        )
        
        data_module.setup()
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"Successfully set up data module with {len(data_module.train_dataset)} training samples and {len(data_module.val_dataset)} validation samples")
        
    except Exception as e:
        print(f"Error setting up data loaders: {e}")
        print("Please check your data paths and dataset configuration.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Training loop
    print("Starting fine-tuning...")
    best_val_loss = float('inf')
    best_val_no_skip_loss = float('inf')

    # Record start time
    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        skip_losses = 0.0
        no_skip_losses = 0.0
        no_skip_count = 0
        
        # Determine no-skip probability and weight for this epoch if using annealing
        if USE_NO_SKIP_ANNEALING:
            # Gradually increase from 0.05 to max value
            no_skip_prob = min(0.05 + epoch * 0.05, NO_SKIP_PROB)
            no_skip_weight = min(0.1 + epoch * 0.1, NO_SKIP_WEIGHT)
        else:
            no_skip_prob = NO_SKIP_PROB
            no_skip_weight = NO_SKIP_WEIGHT
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (images, _) in enumerate(pbar):
            # Move data to device
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # === Part 1: Main pass with skips ===
            # Normal forward pass with skip connections
            if model.use_vq:
                recon_with_skips, z, vq_loss = model(images)
            else:
                recon_with_skips, z = model(images)
                vq_loss = torch.tensor(0.0, device=device)
            
            # Compute reconstruction loss with skip connections
            loss_with_skips = compute_reconstruction_loss(recon_with_skips, images)
            
            # Combine losses for the skip pass
            total_loss_part1 = loss_with_skips
            if model.use_vq:
                total_loss_part1 += vq_loss
            
            # Backpropagate this part first
            total_loss_part1.backward()
            
            # === Part 2: Regularization pass without skips ===
            loss_no_skips = torch.tensor(0.0, device=device)
            
            if np.random.rand() < no_skip_prob:
                # We must detach 'z' from the computation graph of the first pass
                # to prevent gradients from flowing back through the encoder twice.
                recon_no_skips = model.decode(z.detach(), skip_features=None)
                
                # Compute reconstruction loss without skip connections
                # (the size matching happens inside compute_reconstruction_loss)
                loss_no_skips = compute_reconstruction_loss(recon_no_skips, images)
                
                # Backpropagate the no-skip loss separately
                (no_skip_weight * loss_no_skips).backward()
                no_skip_count += 1
            
            # === Part 3: Update weights ===
            # The optimizer now has gradients from both backward passes
            optimizer.step()
            
            # Update statistics
            total_loss = total_loss_part1.item() + (no_skip_weight * loss_no_skips.item())
            train_loss += total_loss
            skip_losses += loss_with_skips.item()
            no_skip_losses += loss_no_skips.item() if loss_no_skips > 0 else 0
            
            # Update progress bar
            pbar.set_postfix({
                'train_loss': total_loss, 
                'skip_loss': loss_with_skips.item(),
                'no_skip_loss': loss_no_skips.item() if loss_no_skips > 0 else 0
            })
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_skip_loss = skip_losses / len(train_loader)
        avg_no_skip_loss = no_skip_losses / no_skip_count if no_skip_count > 0 else 0
        
        logger.info(f"Epoch {epoch+1}/{EPOCHS}:")
        logger.info(f"  Train Loss: {avg_train_loss:.6f}")
        logger.info(f"  Skip Loss: {avg_skip_loss:.6f}")
        logger.info(f"  No-Skip Loss: {avg_no_skip_loss:.6f}")
        logger.info(f"  No-Skip Prob: {no_skip_prob:.3f}, Weight: {no_skip_weight:.3f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_skip_loss = 0.0
        val_no_skip_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(val_loader, desc="Validation")):
                # Move data to device
                images = images.to(device)
                
                # Forward pass with skip connections
                if model.use_vq:
                    recon_with_skips, z, vq_loss = model(images)
                else:
                    recon_with_skips, z = model(images)
                    vq_loss = torch.tensor(0.0, device=device)
                
                # Compute reconstruction loss with skip connections
                loss_with_skips = compute_reconstruction_loss(recon_with_skips, images)
                
                # Forward pass without skip connections
                recon_no_skips = model.decode(z, skip_features=None)
                
                # Compute reconstruction loss without skip connections
                loss_no_skips = compute_reconstruction_loss(recon_no_skips, images)
                
                # Combine losses for tracking (use same weight as training)
                val_total_loss = loss_with_skips + no_skip_weight * loss_no_skips
                if model.use_vq:
                    val_total_loss += vq_loss
                
                # Update statistics
                val_loss += val_total_loss.item()
                val_skip_loss += loss_with_skips.item()
                val_no_skip_loss += loss_no_skips.item()
        
        # Calculate average validation losses
        avg_val_loss = val_loss / len(val_loader)
        avg_val_skip_loss = val_skip_loss / len(val_loader)
        avg_val_no_skip_loss = val_no_skip_loss / len(val_loader)
        
        logger.info(f"  Validation Loss: {avg_val_loss:.6f}")
        logger.info(f"  Validation Skip Loss: {avg_val_skip_loss:.6f}")
        logger.info(f"  Validation No-Skip Loss: {avg_val_no_skip_loss:.6f}")
        
        # Save checkpoint if validation loss improved
        if avg_val_no_skip_loss < best_val_no_skip_loss:  # Changed from avg_val_loss < best_val_loss
            best_val_no_skip_loss = avg_val_no_skip_loss  # Changed from best_val_loss = avg_val_loss
            
            checkpoint_path = os.path.join(OUTPUT_DIR, f'best_finetuned_autoencoder.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_skip_loss': avg_val_skip_loss,
                'val_no_skip_loss': avg_val_no_skip_loss,
            }, checkpoint_path)
    
            logger.info(f"  Saved best model to {checkpoint_path} (No-Skip Loss: {avg_val_no_skip_loss:.6f})")
        
        # Also save the latest model
        checkpoint_path = os.path.join(OUTPUT_DIR, f'latest_finetuned_autoencoder.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_skip_loss': avg_val_skip_loss,
            'val_no_skip_loss': avg_val_no_skip_loss,
        }, checkpoint_path)

    # Calculate total training time
    end_time = datetime.now()
    training_time = end_time - start_time
    logger.info(f"Fine-tuning completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total training time: {training_time}")

    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    main()