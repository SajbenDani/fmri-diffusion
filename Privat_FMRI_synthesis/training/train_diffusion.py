import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import logging

# --- Setup Paths ---
PARENT_DIR = Path(__file__).parent.parent
sys.path.append(str(PARENT_DIR))

from utils.dataset import FMRIDataModule
from models.autoencoder import Improved3DAutoencoder
from models.diffusion import DiffusionUNet3D

from diffusers import DDPMScheduler
from accelerate import Accelerator

# Create logs directory before setting up logging
logs_dir = PARENT_DIR / "logs"
logs_dir.mkdir(exist_ok=True)

# Setup logging after directory is created
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(logs_dir / "diffusion_training.log")
    ]
)
logger = logging.getLogger(__name__)

def train_diffusion():
    # --- Configuration ---
    config = {
        'preprocessed_data_dir': PARENT_DIR / "data_preprocessed",
        'autoencoder_checkpoint': PARENT_DIR / "checkpoints" / "best_autoencoder.pt",
        'output_dir': PARENT_DIR / "checkpoints_diffusion",
        'best_model_path': PARENT_DIR / "checkpoints_diffusion" / "best_diffusion.pt",
        'batch_size': 4,
        'num_workers': 8,
        'learning_rate': 1e-4,
        'num_epochs': 200,
        'scale_factor': 2, # For 2x super-resolution
        'latent_channels': 8, # Must match autoencoder
        'early_stopping_patience': 15,  # Stop after this many epochs without improvement
        'early_stopping_min_delta': 1e-4,  # Minimum change to qualify as improvement
    }
    
    # Create output directory
    config['output_dir'].mkdir(exist_ok=True)
    
    # --- Accelerator for mixed precision and multi-GPU ---
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    
    # --- Load Pre-trained Autoencoder ---
    logger.info("Loading pre-trained autoencoder...")
    autoencoder = Improved3DAutoencoder(latent_channels=config['latent_channels']).to(device)
    autoencoder.load_state_dict(torch.load(config['autoencoder_checkpoint'])['model_state_dict'])
    autoencoder.eval() # Freeze autoencoder
    for param in autoencoder.parameters():
        param.requires_grad = False

    # --- Data Setup ---
    data_module = FMRIDataModule(
        train_csv=config['preprocessed_data_dir'] / "train_patches.csv",
        val_csv=config['preprocessed_data_dir'] / "val_patches.csv",
        test_csv=config['preprocessed_data_dir'] / "test_patches.csv",
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # --- Model, Optimizer, and Scheduler Setup ---
    unet = DiffusionUNet3D(latent_channels=config['latent_channels'])
    
    # Initialize variables for tracking training progress
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Check if we're continuing training from a checkpoint
    if os.path.exists(config['best_model_path']):
        logger.info(f"Loading existing model from {config['best_model_path']}")
        checkpoint = torch.load(config['best_model_path'], map_location=device)
        
        if 'model_state_dict' in checkpoint:
            unet.load_state_dict(checkpoint['model_state_dict'])
            
            # If we have training state information, restore it
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Continuing from epoch {start_epoch}")
            
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                logger.info(f"Previous best validation loss: {best_val_loss:.6f}")
        else:
            # Handle case where checkpoint just contains model weights
            unet.load_state_dict(checkpoint)
            logger.info("Loaded model weights (no training state)")
    
    optimizer = optim.AdamW(unet.parameters(), lr=config['learning_rate'])
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # --- Prepare with Accelerator ---
    unet, optimizer, train_loader, val_loader = accelerator.prepare(
        unet, optimizer, train_loader, val_loader
    )
    
    # Function for validation
    def validate():
        unet.eval()
        val_losses = []
        
        with torch.no_grad():
            for hr_patches, _ in tqdm(val_loader, desc="Validation", leave=False):
                # Create LR/HR pairs and encode
                lr_patches = F.interpolate(
                    hr_patches, 
                    scale_factor=1/config['scale_factor'], 
                    mode='trilinear',
                    align_corners=False
                )
                
                # Encode to latent space
                z_hr, _, _ = autoencoder.encode(hr_patches)
                z_lr, _, _ = autoencoder.encode(lr_patches)
                
                # Upsample z_lr to match z_hr shape for conditioning
                z_lr_upsampled = F.interpolate(z_lr, size=z_hr.shape[2:], mode='trilinear')
                
                # Add noise
                noise = torch.randn_like(z_hr)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                        (hr_patches.shape[0],), device=device)
                z_hr_noisy = noise_scheduler.add_noise(z_hr, noise, timesteps)
                
                # Predict noise
                predicted_noise = unet(z_hr_noisy, timesteps, z_lr_upsampled)
                loss = F.mse_loss(predicted_noise, noise)
                
                val_losses.append(loss.item())
        
        # Calculate average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        return avg_val_loss

    # --- Training Loop ---
    logger.info(f"Starting training for {config['num_epochs']} epochs")
    
    for epoch in range(start_epoch, config['num_epochs']):
        # Training phase
        unet.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for hr_patches, _ in pbar:
            with torch.no_grad():
                # Step 1: Create LR/HR pairs and encode
                lr_patches = F.interpolate(
                    hr_patches, 
                    scale_factor=1/config['scale_factor'], 
                    mode='trilinear',
                    align_corners=False
                )
                
                # Encode to latent space
                z_hr, _, _ = autoencoder.encode(hr_patches)
                z_lr, _, _ = autoencoder.encode(lr_patches)
                
                # Upsample z_lr to match z_hr shape for conditioning
                z_lr_upsampled = F.interpolate(z_lr, size=z_hr.shape[2:], mode='trilinear')

            # Step 2: Diffusion process
            noise = torch.randn_like(z_hr)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                     (hr_patches.shape[0],), device=device)
            z_hr_noisy = noise_scheduler.add_noise(z_hr, noise, timesteps)
            
            # Step 3: Predict noise and calculate loss
            with accelerator.accumulate(unet):
                optimizer.zero_grad()
                
                predicted_noise = unet(z_hr_noisy, timesteps, z_lr_upsampled)
                loss = F.mse_loss(predicted_noise, noise)
                
                accelerator.backward(loss)
                optimizer.step()
            
            train_losses.append(loss.item())
            current_loss = sum(train_losses[-100:]) / min(len(train_losses), 100)
            pbar.set_postfix({"loss": current_loss})
        
        # Calculate average training loss for the epoch
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation phase
        val_loss = validate()
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} - "
                   f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss - config['early_stopping_min_delta']:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save the best model
            if accelerator.is_main_process:
                unwrapped_unet = accelerator.unwrap_model(unet)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': unwrapped_unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, config['best_model_path'])
                logger.info(f"Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs")
            
            # Early stopping
            if epochs_without_improvement >= config['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    logger.info("Training completed!")
    return best_val_loss

if __name__ == "__main__":
    train_diffusion()