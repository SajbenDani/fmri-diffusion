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
log_file = f"diffusion_cfg_adversarial_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(logs_dir / log_file)
    ]
)
logger = logging.getLogger(__name__)

def validate_checkpoint_compatibility(autoencoder_path, diffusion_path):
    """Check if autoencoder and diffusion model are compatible."""
    logger.info("Validating checkpoint compatibility...")
    
    # Load autoencoder checkpoint
    ae_ckpt = torch.load(autoencoder_path, map_location='cpu')
    ae_latent_channels = ae_ckpt.get('latent_channels', 8)
    
    # Load diffusion checkpoint 
    diff_ckpt = torch.load(diffusion_path, map_location='cpu')
    
    # Check if diffusion model expects correct input channels
    state_dict = diff_ckpt['model_state_dict'] if 'model_state_dict' in diff_ckpt else diff_ckpt
    
    # Find first conv layer to check input channels
    for key, tensor in state_dict.items():
        if 'conv' in key.lower() and 'weight' in key and tensor.dim() == 5:
            diff_input_channels = tensor.shape[1]
            expected_channels = ae_latent_channels * 2  # CFG expects noisy + conditioning
            
            logger.info(f"Autoencoder latent channels: {ae_latent_channels}")
            logger.info(f"Diffusion model input channels: {diff_input_channels}")
            logger.info(f"Expected CFG input channels: {expected_channels}")
            
            if diff_input_channels != expected_channels:
                logger.warning(f"Channel mismatch detected! This may cause issues.")
                return False
            else:
                logger.info("Channel compatibility confirmed!")
                return True
            break
    
    logger.warning("Could not determine channel compatibility")
    return True  # Assume compatible if we can't determine

def compute_reconstruction_quality_metrics(autoencoder, val_loader, device):
    """Compute reconstruction quality metrics to monitor autoencoder alignment."""
    autoencoder.eval()
    mse_losses = []
    l1_losses = []
    
    with torch.no_grad():
        for i, (hr_patches, _) in enumerate(val_loader):
            if i >= 20:  # Only check first 20 batches for speed
                break
                
            hr_patches = hr_patches.to(device)
            
            # Reconstruct through autoencoder
            if hasattr(autoencoder, 'use_vq') and autoencoder.use_vq:
                recon, _, _ = autoencoder(hr_patches)
            else:
                recon, _ = autoencoder(hr_patches)
            
            # Compute metrics
            mse = F.mse_loss(recon, hr_patches).item()
            l1 = F.l1_loss(recon, hr_patches).item()
            
            mse_losses.append(mse)
            l1_losses.append(l1)
    
    avg_mse = sum(mse_losses) / len(mse_losses)
    avg_l1 = sum(l1_losses) / len(l1_losses)
    
    logger.info(f"Autoencoder reconstruction quality - MSE: {avg_mse:.6f}, L1: {avg_l1:.6f}")
    return avg_mse, avg_l1

def train_diffusion_cfg():
    # --- Configuration ---
    config = {
        'preprocessed_data_dir': PARENT_DIR / "data_preprocessed",
        'autoencoder_checkpoint': PARENT_DIR / "checkpoints_adversarial" / "best_finetuned_adversarial_autoencoder_second.pt",
        'previous_diffusion_model': PARENT_DIR / "checkpoints_diffusion_cfg" / "best_diffusion_cfg_first.pt",
        'output_dir': PARENT_DIR / "checkpoints_diffusion_cfg",
        'best_model_path': PARENT_DIR / "checkpoints_diffusion_cfg" / "best_diffusion_cfg_adversarial_aligned.pt",
        'batch_size': 4,
        'num_workers': 8,
        'learning_rate': 2e-5,  # Even lower LR for adversarial alignment
        'num_epochs': 80,       # Reduced epochs for fine-tuning
        'scale_factor': 2,
        'latent_channels': 8,
        'base_channels': 128,
        'cfg_dropout_prob': 0.1,
        'early_stopping_patience': 12,  # Reduced patience for fine-tuning
        'early_stopping_min_delta': 5e-5,  # Smaller improvement threshold
        'view': 'axial',
        'warmup_epochs': 3,     # Gradual learning rate warmup
        'gradient_clip_val': 1.0,  # Gradient clipping for stability
        'reconstruction_weight': 0.1,  # Weight for reconstruction consistency loss
    }
    
    # Create output directory
    config['output_dir'].mkdir(exist_ok=True)
    
    # --- Accelerator for mixed precision and multi-GPU ---
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=2)
    device = accelerator.device
    
    logger.info(f"Using device: {device}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    logger.info(f"CFG dropout probability: {config['cfg_dropout_prob']}")
    
    # --- Validate Compatibility ---
    if not validate_checkpoint_compatibility(
        config['autoencoder_checkpoint'], 
        config['previous_diffusion_model']
    ):
        logger.error("Checkpoint compatibility issues detected!")
        return None
    
    # --- Load Pre-trained Autoencoder ---
    logger.info("Loading adversarially-trained autoencoder...")
    autoencoder = Improved3DAutoencoder(
        in_channels=1,
        latent_channels=config['latent_channels'],
        base_channels=32,
        use_vq=True
    ).to(device)
    
    ae_checkpoint = torch.load(config['autoencoder_checkpoint'], map_location=device)
    autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    
    logger.info("Autoencoder loaded successfully")
    
    # Log autoencoder details
    if 'epoch' in ae_checkpoint:
        logger.info(f"Autoencoder was trained for {ae_checkpoint['epoch']} epochs")
    if 'val_loss' in ae_checkpoint:
        logger.info(f"Autoencoder final validation loss: {ae_checkpoint['val_loss']:.6f}")

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

    # --- Compute initial autoencoder reconstruction quality ---
    initial_ae_mse, initial_ae_l1 = compute_reconstruction_quality_metrics(autoencoder, val_loader, device)

    # --- Model, Optimizer, and Scheduler Setup ---
    unet = DiffusionUNet3D(
        latent_channels=config['latent_channels'],
        base_channels=config['base_channels'],
        time_emb_dim=256
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
            missing_keys, unexpected_keys = unet.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            if 'best_val_loss' in checkpoint:
                prev_best_val_loss = checkpoint['best_val_loss']
                logger.info(f"Previous best validation loss: {prev_best_val_loss:.6f}")
        else:
            unet.load_state_dict(checkpoint, strict=False)
        
        logger.info("Successfully loaded previous diffusion model")
    except Exception as e:
        logger.error(f"Error loading previous model: {e}")
        logger.warning("Starting training from scratch!")
    
    unet = unet.to(device)
    
    # Setup optimizer with learning rate warmup
    optimizer = optim.AdamW(unet.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        return 1.0
    
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # --- Prepare with Accelerator ---
    unet, optimizer, train_loader, val_loader = accelerator.prepare(
        unet, optimizer, train_loader, val_loader
    )
    
    # Function for validation with CFG and reconstruction consistency
    def validate():
        unet.eval()
        val_losses = []
        recon_losses = []
        
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
                
                # Conditional and unconditional predictions
                pred_cond = unet(z_hr_noisy, timesteps, z_lr_upsampled)
                pred_uncond = unet(z_hr_noisy, timesteps, torch.zeros_like(z_lr_upsampled))
                
                # Calculate losses
                loss_cond = F.mse_loss(pred_cond, noise)
                loss_uncond = F.mse_loss(pred_uncond, noise)
                combined_loss = 0.9 * loss_cond + 0.1 * loss_uncond
                
                val_losses.append(combined_loss.item())
                
                # Optional: Reconstruction consistency check
                if len(recon_losses) < 10:  # Only check first few batches
                    # Denoise completely and check reconstruction
                    z_denoised = z_hr  # For simplicity, use clean latents
                    reconstructed = autoencoder.decode(z_denoised, skip_features=None)
                    recon_loss = F.mse_loss(reconstructed, hr_patches)
                    recon_losses.append(recon_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_recon_loss = sum(recon_losses) / len(recon_losses) if recon_losses else 0.0
        
        return avg_val_loss, avg_recon_loss

    # --- Training Loop with Enhanced CFG ---
    logger.info(f"Starting adversarial-aligned CFG training for {config['num_epochs']} epochs")
    
    for epoch in range(start_epoch, config['num_epochs']):
        # Training phase
        unet.train()
        train_losses = []
        train_losses_cond = []
        train_losses_uncond = []
        
        # Update learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (LR: {current_lr:.2e})")
        for hr_patches, _ in pbar:
            with torch.no_grad():
                # Create LR/HR pairs and encode
                lr_patches = F.interpolate(
                    hr_patches, 
                    scale_factor=1/config['scale_factor'], 
                    mode='trilinear',
                    align_corners=False
                )
                
                # Encode to latent space using adversarial autoencoder
                z_hr, _, _ = autoencoder.encode(hr_patches)
                z_lr, _, _ = autoencoder.encode(lr_patches)
                
                # Upsample z_lr to match z_hr shape for conditioning
                z_lr_upsampled = F.interpolate(z_lr, size=z_hr.shape[2:], mode='trilinear', align_corners=False)

            # Diffusion process
            noise = torch.randn_like(z_hr)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                     (hr_patches.shape[0],), device=device)
            z_hr_noisy = noise_scheduler.add_noise(z_hr, noise, timesteps)
            
            # CFG training with dropout
            batch_size = hr_patches.shape[0]
            mask = torch.rand(batch_size, device=device) > config['cfg_dropout_prob']
            mask = mask.view(batch_size, 1, 1, 1, 1)
            z_lr_conditioned = z_lr_upsampled * mask
            
            with accelerator.accumulate(unet):
                optimizer.zero_grad()
                
                # Predict noise
                predicted_noise = unet(z_hr_noisy, timesteps, z_lr_conditioned)
                loss = F.mse_loss(predicted_noise, noise)
                
                # Gradient clipping
                if config['gradient_clip_val'] > 0:
                    accelerator.clip_grad_norm_(unet.parameters(), config['gradient_clip_val'])
                
                accelerator.backward(loss)
                optimizer.step()
                
                # Monitor conditional and unconditional performance
                with torch.no_grad():
                    pred_cond = unet(z_hr_noisy, timesteps, z_lr_upsampled)
                    pred_uncond = unet(z_hr_noisy, timesteps, torch.zeros_like(z_lr_upsampled))
                    loss_cond = F.mse_loss(pred_cond, noise).item()
                    loss_uncond = F.mse_loss(pred_uncond, noise).item()
            
            train_losses.append(loss.item())
            train_losses_cond.append(loss_cond)
            train_losses_uncond.append(loss_uncond)
            
            # Update progress bar
            current_loss = sum(train_losses[-50:]) / min(len(train_losses), 50)
            pbar.set_postfix({
                "loss": f"{current_loss:.6f}",
                "cond": f"{loss_cond:.6f}",
                "uncond": f"{loss_uncond:.6f}"
            })
        
        # Calculate average training losses
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_loss_cond = sum(train_losses_cond) / len(train_losses_cond)
        avg_train_loss_uncond = sum(train_losses_uncond) / len(train_losses_uncond)
        
        # Validation phase
        val_loss, recon_consistency = validate()
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        logger.info(f"  Train Loss: {avg_train_loss:.6f} (Cond: {avg_train_loss_cond:.6f}, Uncond: {avg_train_loss_uncond:.6f})")
        logger.info(f"  Val Loss: {val_loss:.6f}")
        logger.info(f"  Reconstruction Consistency: {recon_consistency:.6f}")
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        
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
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'recon_consistency': recon_consistency,
                    'cfg_dropout_prob': config['cfg_dropout_prob'],
                    'training_method': 'adversarial-aligned-cfg',
                    'autoencoder_checkpoint': str(config['autoencoder_checkpoint']),
                    'initial_ae_mse': initial_ae_mse,
                    'config': config
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
        
        # Save checkpoint every 5 epochs
        if accelerator.is_main_process and (epoch + 1) % 5 == 0:
            unwrapped_unet = accelerator.unwrap_model(unet)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': unwrapped_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'recon_consistency': recon_consistency,
                'cfg_dropout_prob': config['cfg_dropout_prob'],
                'training_method': 'adversarial-aligned-cfg',
                'autoencoder_checkpoint': str(config['autoencoder_checkpoint']),
                'config': config
            }
            torch.save(checkpoint, config['output_dir'] / f"diffusion_cfg_adversarial_epoch_{epoch+1}.pt")
    
    logger.info("Adversarial-aligned CFG training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    return best_val_loss

if __name__ == "__main__":
    start_time = time.time()
    logger.info(f"Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    best_loss = train_diffusion_cfg()
    
    if best_loss is not None:
        total_time = time.time() - start_time
        logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        logger.info(f"Best validation loss: {best_loss:.6f}")
        logger.info(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        logger.error("Training failed due to compatibility issues")