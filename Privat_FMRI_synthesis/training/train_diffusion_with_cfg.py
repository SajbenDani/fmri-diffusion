#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import logging
from datetime import datetime

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
log_file = f"diffusion_cfg_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(logs_dir / log_file)
    ]
)
logger = logging.getLogger(__name__)

def train_diffusion_cfg():
    # --- Configuration ---
    config = {
        'preprocessed_data_dir': PARENT_DIR / "data_preprocessed",
        'autoencoder_checkpoint': PARENT_DIR / "checkpoints_finetuned" / "best_finetuned_autoencoder.pt",  # Use FINETUNED model
        'previous_diffusion_model': PARENT_DIR / "checkpoints_diffusion" / "best_diffusion.pt",  # Load previous model
        'output_dir': PARENT_DIR / "checkpoints_diffusion_cfg",
        'best_model_path': PARENT_DIR / "checkpoints_diffusion_cfg" / "best_diffusion_cfg.pt",
        'batch_size': 4,
        'num_workers': 8,
        'learning_rate': 5e-5,  # Slightly lower LR for fine-tuning
        'num_epochs': 100,      # Fewer epochs for fine-tuning
        'scale_factor': 2,      # For 2x super-resolution
        'latent_channels': 8,   # Must match autoencoder
        'base_channels': 128,   # Must match previous UNet
        'cfg_dropout_prob': 0.1, # Probability of dropping conditioning (10% is common)
        'early_stopping_patience': 15,
        'early_stopping_min_delta': 1e-4,
        'view': 'axial'         # View for the dataset
    }
    
    # Create output directory
    config['output_dir'].mkdir(exist_ok=True)
    
    # --- Accelerator for mixed precision and multi-GPU ---
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    
    logger.info(f"Using device: {device}")
    logger.info(f"CFG dropout probability: {config['cfg_dropout_prob']}")
    
    # --- Load Pre-trained Autoencoder ---
    logger.info("Loading pre-trained finetuned autoencoder...")
    autoencoder = Improved3DAutoencoder(
        in_channels=1,
        latent_channels=config['latent_channels'],
        base_channels=32,  # Default value used in your model
        use_vq=True
    ).to(device)
    
    autoencoder.load_state_dict(torch.load(config['autoencoder_checkpoint'], map_location=device)['model_state_dict'])
    autoencoder.eval()  # Freeze autoencoder
    for param in autoencoder.parameters():
        param.requires_grad = False
    logger.info("Autoencoder loaded successfully")

    # --- Data Setup ---
    data_module = FMRIDataModule(
        train_csv=os.path.join(config['preprocessed_data_dir'], "train_patches.csv"),
        val_csv=os.path.join(config['preprocessed_data_dir'], "val_patches.csv"),
        test_csv=os.path.join(config['preprocessed_data_dir'], "test_patches.csv"),
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        prefetch_factor=2,
        view=config['view']
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    logger.info(f"Data loaded: {len(data_module.train_dataset)} training samples, {len(data_module.val_dataset)} validation samples")

    # --- Model, Optimizer, and Scheduler Setup ---
    # Initialize the UNet with the same parameters as before
    unet = DiffusionUNet3D(
        latent_channels=config['latent_channels'],
        base_channels=config['base_channels'],
        time_emb_dim=256  # Default value used in your model
    )
    
    # Initialize variables for tracking training progress
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Load the previous diffusion model for fine-tuning
    logger.info(f"Loading previous diffusion model from {config['previous_diffusion_model']}")
    try:
        checkpoint = torch.load(config['previous_diffusion_model'], map_location=device)
        
        if 'model_state_dict' in checkpoint:
            unet.load_state_dict(checkpoint['model_state_dict'])
            
            # If we have training state information, we can reference it
            if 'best_val_loss' in checkpoint:
                prev_best_val_loss = checkpoint['best_val_loss']
                logger.info(f"Previous best validation loss: {prev_best_val_loss:.6f}")
        else:
            # Handle case where checkpoint just contains model weights
            unet.load_state_dict(checkpoint)
        
        logger.info("Successfully loaded previous diffusion model")
    except Exception as e:
        logger.error(f"Error loading previous model: {e}")
        logger.warning("Starting training from scratch!")
    
    unet = unet.to(device)
    optimizer = optim.AdamW(unet.parameters(), lr=config['learning_rate'])
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # --- Prepare with Accelerator ---
    unet, optimizer, train_loader, val_loader = accelerator.prepare(
        unet, optimizer, train_loader, val_loader
    )
    
    # Function for validation with CFG
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
                z_lr_upsampled = F.interpolate(z_lr, size=z_hr.shape[2:], mode='trilinear', align_corners=False)
                
                # Add noise
                noise = torch.randn_like(z_hr)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                        (hr_patches.shape[0],), device=device)
                z_hr_noisy = noise_scheduler.add_noise(z_hr, noise, timesteps)
                
                # For validation, we evaluate both conditional and unconditional paths
                batch_size = hr_patches.shape[0]
                
                # Conditional prediction (with conditioning)
                pred_cond = unet(z_hr_noisy, timesteps, z_lr_upsampled)
                
                # Unconditional prediction (no conditioning)
                pred_uncond = unet(z_hr_noisy, timesteps, torch.zeros_like(z_lr_upsampled))
                
                # Calculate losses for both paths
                loss_cond = F.mse_loss(pred_cond, noise)
                loss_uncond = F.mse_loss(pred_uncond, noise)
                
                # Combined loss (weighted toward conditional)
                loss = 0.9 * loss_cond + 0.1 * loss_uncond
                val_losses.append(loss.item())
        
        # Calculate average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        return avg_val_loss

    # --- Training Loop with CFG ---
    logger.info(f"Starting CFG training for {config['num_epochs']} epochs")
    
    for epoch in range(start_epoch, config['num_epochs']):
        # Training phase
        unet.train()
        train_losses = []
        train_losses_cond = []
        train_losses_uncond = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for hr_patches, _ in pbar:
            with torch.no_grad():
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
                z_lr_upsampled = F.interpolate(z_lr, size=z_hr.shape[2:], mode='trilinear', align_corners=False)

            # Diffusion process
            noise = torch.randn_like(z_hr)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                     (hr_patches.shape[0],), device=device)
            z_hr_noisy = noise_scheduler.add_noise(z_hr, noise, timesteps)
            
            # --- CFG TRAINING LOGIC ---
            # Randomly decide which samples in the batch will have conditioning dropped
            batch_size = hr_patches.shape[0]
            mask = torch.rand(batch_size, device=device) > config['cfg_dropout_prob']
            mask = mask.view(batch_size, 1, 1, 1, 1)  # Reshape for broadcasting
            
            # Create conditioned version - selectively zero out some conditioning inputs
            z_lr_conditioned = z_lr_upsampled * mask
            
            with accelerator.accumulate(unet):
                optimizer.zero_grad()
                
                # Predict noise
                predicted_noise = unet(z_hr_noisy, timesteps, z_lr_conditioned)
                loss = F.mse_loss(predicted_noise, noise)
                
                # Also compute conditional and unconditional losses for monitoring
                with torch.no_grad():
                    pred_cond = unet(z_hr_noisy, timesteps, z_lr_upsampled)
                    pred_uncond = unet(z_hr_noisy, timesteps, torch.zeros_like(z_lr_upsampled))
                    loss_cond = F.mse_loss(pred_cond, noise).item()
                    loss_uncond = F.mse_loss(pred_uncond, noise).item()
                
                accelerator.backward(loss)
                optimizer.step()
            
            train_losses.append(loss.item())
            train_losses_cond.append(loss_cond)
            train_losses_uncond.append(loss_uncond)
            
            current_loss = sum(train_losses[-100:]) / min(len(train_losses), 100)
            pbar.set_postfix({
                "loss": f"{current_loss:.6f}",
                "cond": f"{loss_cond:.6f}",
                "uncond": f"{loss_uncond:.6f}"
            })
        
        # Calculate average training losses for the epoch
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_loss_cond = sum(train_losses_cond) / len(train_losses_cond)
        avg_train_loss_uncond = sum(train_losses_uncond) / len(train_losses_uncond)
        
        # Validation phase
        val_loss = validate()
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        logger.info(f"  Train Loss: {avg_train_loss:.6f} (Cond: {avg_train_loss_cond:.6f}, Uncond: {avg_train_loss_uncond:.6f})")
        logger.info(f"  Val Loss: {val_loss:.6f}")
        
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
                    'best_val_loss': best_val_loss,
                    'cfg_dropout_prob': config['cfg_dropout_prob'],
                    'training_method': 'classifier-free-guidance'
                }
                torch.save(checkpoint, config['best_model_path'])
                logger.info(f"  Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No improvement for {epochs_without_improvement} epochs")
            
            # Early stopping
            if epochs_without_improvement >= config['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Also save latest model
        if accelerator.is_main_process and (epoch + 1) % 5 == 0:
            unwrapped_unet = accelerator.unwrap_model(unet)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': unwrapped_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'cfg_dropout_prob': config['cfg_dropout_prob'],
                'training_method': 'classifier-free-guidance'
            }
            torch.save(checkpoint, config['output_dir'] / f"diffusion_cfg_epoch_{epoch+1}.pt")
    
    logger.info("CFG training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    return best_val_loss

if __name__ == "__main__":
    start_time = time.time()
    best_loss = train_diffusion_cfg()
    total_time = time.time() - start_time
    
    logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    logger.info(f"Best validation loss: {best_loss:.6f}")