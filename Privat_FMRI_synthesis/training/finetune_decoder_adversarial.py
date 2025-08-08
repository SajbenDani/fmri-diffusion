#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import logging
from datetime import datetime
import numpy as np

# --- Setup Paths ---
PARENT_DIR = Path(__file__).parent.parent
sys.path.append(str(PARENT_DIR))

from utils.dataset import FMRIDataModule
from models.autoencoder import Improved3DAutoencoder

# --- Hardcoded Configuration ---
# Initial autoencoder (not adversarially trained) - starting fresh as recommended
INITIAL_AUTOENCODER_PATH = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_finetuned/best_finetuned_autoencoder.pt"
# Where to save new checkpoints
OUTPUT_DIR = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_adversarial"
# Where to save logs
LOGS_DIR = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/logs"
# Number of workers for data loading
NUM_WORKERS = 16
# Whether to resume training from checkpoint or start fresh
RESUME_TRAINING = False  # Set to False to start fresh as recommended

# Create logs directory
logs_dir = Path(LOGS_DIR)
logs_dir.mkdir(exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = logs_dir / f"adversarial_training_stable_{timestamp}.txt"

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Error tracking to avoid console flooding
error_count = 0
MAX_ERROR_LOGS = 5

# --- 3D Discriminator Architecture with Adaptive Sizing ---
class AdaptiveDiscriminator3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, min_required_size=16):
        super().__init__()
        self.min_required_size = min_required_size
        
        # Initial convolutional layer
        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Downsample blocks with smaller kernels and adaptive number of layers
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(base_channels * (2**i if i > 0 else 1), 
                          base_channels * (2**(i+1)), 
                          kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(8, base_channels * (2**(i+1))), base_channels * (2**(i+1))),
                nn.LeakyReLU(0.2, inplace=True)
            ) for i in range(3)
        ])
        
        # Final classification layer with smaller kernel
        self.final = nn.Conv3d(base_channels * 8, 1, kernel_size=3, stride=1, padding=1)
        
        # Global pooling instead of fixed kernel for final decision
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, x):
        # Check if input is large enough
        if min(x.shape[2:]) < self.min_required_size:
            # Resize if too small
            x = F.interpolate(x, size=(self.min_required_size, self.min_required_size, self.min_required_size),
                             mode='trilinear', align_corners=False)
        
        x = self.initial(x)
        
        # Apply downsample blocks as long as the feature map is large enough
        for block in self.down_blocks:
            if min(x.shape[2:]) <= 3:  # Stop if too small
                break
            x = block(x)
        
        x = self.final(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1).squeeze(1)

def adversarial_finetune_decoder():
    # --- Configuration ---
    config = {
        'preprocessed_data_dir': PARENT_DIR / "data_preprocessed",
        'output_dir': Path(OUTPUT_DIR),
        'best_model_path': Path(OUTPUT_DIR) / "best_finetuned_adversarial_autoencoder.pt",
        'latest_checkpoint_path': Path(OUTPUT_DIR) / "latest_checkpoint.pt",
        'batch_size': 4,
        'num_workers': NUM_WORKERS,
        'ae_learning_rate': 1e-5,  # REDUCED as per suggestion
        'disc_learning_rate': 5e-5,  # REDUCED as per suggestion
        'num_epochs': 50,
        'lambda_anchor': 1.0,  # NEW: Weight for the anchor reconstruction loss
        'lambda_recon': 1.0,    # Weight for no-skip reconstruction
        'lambda_adv': 0.001,    # REDUCED as per suggestion (start small)
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 1e-4,
        'view': 'axial',
        'latent_channels': 8,
        'base_channels': 32,
        'use_vq': True,
        'min_input_size': 16,
        'disc_update_freq': 1   # How often to update discriminator (can be increased)
    }
    
    logger.info(f"Using {config['num_workers']} data loading workers")
    logger.info(f"STABLE TRAINING: Using anchor loss with weight {config['lambda_anchor']}")
    logger.info(f"STABLE TRAINING: Starting with reduced adversarial weight {config['lambda_adv']}")
    
    # Create output directory
    config['output_dir'].mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # --- Load Models and Optimizers ---
    logger.info("Initializing models...")
    autoencoder = Improved3DAutoencoder(
        in_channels=1,
        latent_channels=config['latent_channels'],
        base_channels=config['base_channels'],
        use_vq=config['use_vq']
    ).to(device)
    
    discriminator = AdaptiveDiscriminator3D(
        in_channels=1, 
        base_channels=16,
        min_required_size=config['min_input_size']
    ).to(device)
    
    # Initialize optimizers
    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=config['ae_learning_rate'])
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=config['disc_learning_rate'])
    
    # --- Training state variables ---
    start_epoch = 0
    best_recon_loss = float('inf')
    epochs_without_improvement = 0
    current_lambda_adv = config['lambda_adv']
    
    # --- Load checkpoint if resuming ---
    if RESUME_TRAINING and os.path.exists(config['latest_checkpoint_path']):
        logger.info(f"Resuming training from checkpoint: {config['latest_checkpoint_path']}")
        try:
            checkpoint = torch.load(config['latest_checkpoint_path'], map_location=device)
            
            # Load autoencoder
            autoencoder.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded autoencoder state")
            
            # Load discriminator if available
            if 'discriminator_state_dict' in checkpoint:
                discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                logger.info("Loaded discriminator state")
            else:
                logger.warning("No discriminator state found in checkpoint. Discriminator will start from scratch.")
            
            # Load optimizer states if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer_ae.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded autoencoder optimizer state")
            
            if 'disc_optimizer_state_dict' in checkpoint:
                optimizer_disc.load_state_dict(checkpoint['disc_optimizer_state_dict'])
                logger.info("Loaded discriminator optimizer state")
            
            # Load other training state variables
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                logger.info(f"Resuming from epoch {start_epoch}")
            
            if 'best_recon_loss' in checkpoint:
                best_recon_loss = checkpoint['best_recon_loss']
                logger.info(f"Previous best reconstruction loss: {best_recon_loss:.6f}")
            
            if 'lambda_adv' in checkpoint:
                current_lambda_adv = checkpoint['lambda_adv']
                logger.info(f"Using adversarial weight: {current_lambda_adv}")
            
            if 'epochs_without_improvement' in checkpoint:
                epochs_without_improvement = checkpoint['epochs_without_improvement']
                logger.info(f"Epochs without improvement: {epochs_without_improvement}")
                
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting training from scratch")
    else:
        # Load the initial autoencoder checkpoint if not resuming
        try:
            logger.info(f"Loading initial autoencoder from: {INITIAL_AUTOENCODER_PATH}")
            checkpoint = torch.load(INITIAL_AUTOENCODER_PATH, map_location=device)
            if 'model_state_dict' in checkpoint:
                autoencoder.load_state_dict(checkpoint['model_state_dict'])
            else:
                autoencoder.load_state_dict(checkpoint)
            logger.info("Autoencoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load initial autoencoder: {e}")
            return
    
    logger.info("Models and optimizers initialized")

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

    # --- Loss Functions ---
    recon_criterion = nn.L1Loss()  # L1 loss for reconstruction
    adv_criterion = nn.BCEWithLogitsLoss()  # BCE loss for adversarial training
    
    # Function to ensure tensors have the same size
    def ensure_same_size(source, target):
        if source.shape[2:] != target.shape[2:]:
            source = F.interpolate(source, size=target.shape[2:], mode='trilinear', align_corners=False)
        return source
    
    # --- Validation function ---
    def validate():
        autoencoder.eval()
        discriminator.eval()
        
        val_anchor_losses = []  # NEW: tracking anchor losses
        val_recon_losses = []
        val_adv_losses = []
        val_total_losses = []
        val_disc_losses = []
        
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc="Validation", leave=False):
                try:
                    images = images.to(device, non_blocking=True)
                    
                    # Get latent representation and with-skips reconstruction
                    if config['use_vq']:
                        recon_with_skips, z, vq_loss = autoencoder(images)
                    else:
                        recon_with_skips, z = autoencoder(images)
                        vq_loss = torch.tensor(0.0, device=device)
                    
                    # Generate without skip connections
                    recon_no_skips = autoencoder.decode(z, skip_features=None)
                    recon_no_skips = ensure_same_size(recon_no_skips, images)
                    
                    # Compute losses - NOW INCLUDES ANCHOR LOSS
                    anchor_loss = recon_criterion(recon_with_skips, images)
                    recon_loss = recon_criterion(recon_no_skips, images)
                    
                    # Discriminator scores
                    real_preds = discriminator(images)
                    fake_preds = discriminator(recon_no_skips)
                    
                    # Generator adversarial loss
                    adv_loss = adv_criterion(fake_preds, torch.ones_like(fake_preds, device=device))
                    
                    # Discriminator loss
                    disc_real_loss = adv_criterion(real_preds, torch.ones_like(real_preds, device=device))
                    disc_fake_loss = adv_criterion(fake_preds, torch.zeros_like(fake_preds, device=device))
                    disc_loss = (disc_real_loss + disc_fake_loss) / 2
                    
                    # Total loss - NOW INCLUDES ANCHOR LOSS
                    total_loss = (
                        config['lambda_anchor'] * anchor_loss + 
                        config['lambda_recon'] * recon_loss + 
                        current_lambda_adv * adv_loss
                    )
                    
                    # Record metrics
                    val_anchor_losses.append(anchor_loss.item())
                    val_recon_losses.append(recon_loss.item())
                    val_adv_losses.append(adv_loss.item())
                    val_total_losses.append(total_loss.item())
                    val_disc_losses.append(disc_loss.item())
                    
                except Exception as e:
                    # Limit error logging
                    global error_count
                    if error_count < MAX_ERROR_LOGS:
                        logger.error(f"Error during validation: {e}")
                        error_count += 1
                    elif error_count == MAX_ERROR_LOGS:
                        logger.error("Too many errors, suppressing further error logs")
                        error_count += 1
                    continue
        
        # Calculate averages
        if val_recon_losses:
            avg_anchor_loss = sum(val_anchor_losses) / len(val_anchor_losses)
            avg_recon_loss = sum(val_recon_losses) / len(val_recon_losses)
            avg_adv_loss = sum(val_adv_losses) / len(val_adv_losses)
            avg_total_loss = sum(val_total_losses) / len(val_total_losses)
            avg_disc_loss = sum(val_disc_losses) / len(val_disc_losses)
            return avg_total_loss, avg_anchor_loss, avg_recon_loss, avg_adv_loss, avg_disc_loss
        else:
            logger.error("No valid batches during validation")
            return float('inf'), float('inf'), float('inf'), float('inf'), float('inf')

    # --- Training Loop ---
    training_start_time = time.time()
    logger.info(f"Starting STABLE adversarial fine-tuning from epoch {start_epoch+1} to {config['num_epochs']}")
    
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        autoencoder.train()
        discriminator.train()
        
        # Tracking metrics
        train_anchor_losses = []  # NEW: tracking anchor losses
        train_recon_losses = []
        train_adv_losses = []
        train_total_losses = []
        train_disc_losses = []
        train_disc_real_accs = []
        train_disc_fake_accs = []
        
        # Reset error count each epoch
        global error_count
        error_count = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, (images, _) in enumerate(pbar):
            try:
                images = images.to(device, non_blocking=True)
                batch_size = images.shape[0]
                
                # ------------------------
                # Train Discriminator
                # ------------------------
                if batch_idx % config['disc_update_freq'] == 0:
                    optimizer_disc.zero_grad(set_to_none=True)
                    
                    # Generate reconstructions without skip connections
                    with torch.no_grad():
                        # Full forward pass to get latent representation
                        if config['use_vq']:
                            recon_with_skips, z, vq_loss = autoencoder(images)
                        else:
                            recon_with_skips, z = autoencoder(images)
                            vq_loss = torch.tensor(0.0, device=device)
                        
                        # Generate without skip connections
                        recon_no_skips = autoencoder.decode(z, skip_features=None)
                        recon_no_skips = ensure_same_size(recon_no_skips, images)
                    
                    # Process real images with discriminator
                    real_preds = discriminator(images)
                    real_labels = torch.ones_like(real_preds, device=device)
                    loss_disc_real = adv_criterion(real_preds, real_labels)
                    
                    # Process fake images with discriminator
                    fake_preds = discriminator(recon_no_skips.detach())
                    fake_labels = torch.zeros_like(fake_preds, device=device)
                    loss_disc_fake = adv_criterion(fake_preds, fake_labels)
                    
                    # Total discriminator loss
                    loss_disc = (loss_disc_real + loss_disc_fake) / 2
                    loss_disc.backward()
                    optimizer_disc.step()
                    
                    # Calculate discriminator accuracy
                    disc_real_acc = (real_preds > 0).float().mean().item()
                    disc_fake_acc = (fake_preds < 0).float().mean().item()
                    
                    # Record discriminator metrics
                    train_disc_losses.append(loss_disc.item())
                    train_disc_real_accs.append(disc_real_acc)
                    train_disc_fake_accs.append(disc_fake_acc)
                
                # ------------------------
                # Train Generator (Autoencoder) - MODIFIED WITH ANCHOR LOSS
                # ------------------------
                optimizer_ae.zero_grad(set_to_none=True)
                
                # --- STEP 1: ANCHORING LOSS (Prevents Catastrophic Forgetting) ---
                # Full forward pass to get high-quality reconstruction with skip connections
                if config['use_vq']:
                    recon_with_skips, z, vq_loss = autoencoder(images)
                else:
                    recon_with_skips, z = autoencoder(images)
                    vq_loss = torch.tensor(0.0, device=device)
                    
                # Calculate anchoring loss - this preserves the primary reconstruction ability
                loss_anchor_recon = recon_criterion(recon_with_skips, images)
                
                # --- STEP 2: ADVERSARIAL & NO-SKIP LOSS ---
                # Generate without skip connections for adversarial training
                recon_no_skips = autoencoder.decode(z, skip_features=None)
                recon_no_skips = ensure_same_size(recon_no_skips, images)
                
                # Reconstruction loss on the no-skip path
                loss_no_skip_recon = recon_criterion(recon_no_skips, images)
                
                # Adversarial loss (fool the discriminator)
                fake_preds = discriminator(recon_no_skips)
                loss_adv = adv_criterion(fake_preds, torch.ones_like(fake_preds, device=device))
                
                # --- STEP 3: COMBINED LOSS ---
                # The total loss now includes the anchoring loss, which is critical
                loss_ae = (
                    config['lambda_anchor'] * loss_anchor_recon +  # Anchor loss preserves skip-connection reconstruction
                    config['lambda_recon'] * loss_no_skip_recon +  # No-skip reconstruction loss
                    current_lambda_adv * loss_adv                   # Adversarial loss
                )
                
                # Add VQ loss if present
                if vq_loss is not None:
                    loss_ae += vq_loss
                
                loss_ae.backward()
                optimizer_ae.step()
                
                # Record losses
                train_anchor_losses.append(loss_anchor_recon.item())
                train_recon_losses.append(loss_no_skip_recon.item())
                train_adv_losses.append(loss_adv.item())
                train_total_losses.append(loss_ae.item())
                
                # Update progress bar
                avg_anchor = np.mean(train_anchor_losses[-100:])
                avg_recon = np.mean(train_recon_losses[-100:])
                avg_adv = np.mean(train_adv_losses[-100:])
                
                # Calculate discriminator metrics if available
                if train_disc_losses:
                    avg_disc = np.mean(train_disc_losses[-100:])
                    avg_real_acc = np.mean(train_disc_real_accs[-100:])
                    avg_fake_acc = np.mean(train_disc_fake_accs[-100:])
                else:
                    avg_disc, avg_real_acc, avg_fake_acc = 0, 0, 0
                
                pbar.set_postfix({
                    "anchor": f"{avg_anchor:.4f}",  # NEW: show anchor loss
                    "noskip": f"{avg_recon:.4f}",
                    "adv": f"{avg_adv:.4f}",
                    "disc": f"{avg_disc:.4f}",
                    "r_acc": f"{avg_real_acc:.2f}",
                    "f_acc": f"{avg_fake_acc:.2f}"
                })
                
            except Exception as e:
                # Limit error logging to avoid console flooding
                if error_count < MAX_ERROR_LOGS:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    error_count += 1
                elif error_count == MAX_ERROR_LOGS:
                    logger.error("Too many errors, suppressing further error logs")
                    error_count += 1
                continue
        
        # Skip epoch if no valid batches
        if not train_recon_losses:
            logger.error("No valid batches this epoch, skipping validation")
            continue
        
        # Calculate average training metrics
        avg_train_anchor = sum(train_anchor_losses) / len(train_anchor_losses)
        avg_train_recon = sum(train_recon_losses) / len(train_recon_losses)
        avg_train_adv = sum(train_adv_losses) / len(train_adv_losses)
        avg_train_total = sum(train_total_losses) / len(train_total_losses)
        
        if train_disc_losses:
            avg_train_disc = sum(train_disc_losses) / len(train_disc_losses)
            avg_train_real_acc = sum(train_disc_real_accs) / len(train_disc_real_accs)
            avg_train_fake_acc = sum(train_disc_fake_accs) / len(train_disc_fake_accs)
        else:
            avg_train_disc, avg_train_real_acc, avg_train_fake_acc = 0, 0, 0
        
        # Run validation
        val_total, val_anchor, val_recon, val_adv, val_disc = validate()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} completed in {epoch_time:.2f}s")
        logger.info(f"Train - Anchor: {avg_train_anchor:.6f}, No-Skip: {avg_train_recon:.6f}, Adv: {avg_train_adv:.6f}")
        logger.info(f"Train - Disc Loss: {avg_train_disc:.6f}, Real Acc: {avg_train_real_acc:.4f}, Fake Acc: {avg_train_fake_acc:.4f}")
        logger.info(f"Val   - Anchor: {val_anchor:.6f}, No-Skip: {val_recon:.6f}, Adv: {val_adv:.6f}")
        
        # Save latest checkpoint after EVERY epoch (for resuming training)
        latest_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': autoencoder.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_ae.state_dict(),
            'disc_optimizer_state_dict': optimizer_disc.state_dict(),
            'val_loss': val_total,
            'val_anchor_loss': val_anchor,
            'val_recon_loss': val_recon,
            'val_adv_loss': val_adv,
            'best_recon_loss': best_recon_loss,
            'lambda_adv': current_lambda_adv,
            'lambda_anchor': config['lambda_anchor'],
            'epochs_without_improvement': epochs_without_improvement,
            'training_method': 'adversarial_stable'
        }
        torch.save(latest_checkpoint, config['latest_checkpoint_path'])
        logger.info(f"Saved latest checkpoint for resuming training")
        
        # Check if model improved - focus on no-skip reconstruction loss
        if val_recon < best_recon_loss - config['early_stopping_min_delta']:
            best_recon_loss = val_recon
            epochs_without_improvement = 0
            
            # Save the best model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': autoencoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer_ae.state_dict(),
                'disc_optimizer_state_dict': optimizer_disc.state_dict(),
                'val_loss': val_total,
                'val_anchor_loss': val_anchor,
                'val_recon_loss': val_recon,
                'val_adv_loss': val_adv,
                'lambda_adv': current_lambda_adv,
                'lambda_anchor': config['lambda_anchor'],
                'best_recon_loss': best_recon_loss,
                'training_method': 'adversarial_stable'
            }
            torch.save(checkpoint, config['best_model_path'])
            logger.info(f"Saved best model (val_recon_loss: {val_recon:.6f})")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs")
            
            # Early stopping
            if epochs_without_improvement >= config['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Adjust adversarial weight if discriminator is too strong - but more cautiously
        if avg_train_fake_acc > 0.9 and epoch > 10:  # Wait longer before increasing
            current_lambda_adv *= 1.1  # Increase more gradually (1.1x instead of 1.2x)
            logger.info(f"Increased adversarial weight to {current_lambda_adv:.6f}")
        
        # Monitor anchor loss to ensure it stays low
        if val_anchor > 0.1 and epoch > 5:  # If anchor loss starts getting high
            logger.warning("Anchor loss increasing - model may be forgetting primary task")
            # Increase anchor weight to compensate
            config['lambda_anchor'] *= 1.2
            logger.info(f"Increased anchor weight to {config['lambda_anchor']:.6f}")
    
    # Training complete
    total_time = time.time() - training_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info("Stable adversarial fine-tuning completed!")
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Best no-skip reconstruction loss: {best_recon_loss:.6f}")
    
    return best_recon_loss

if __name__ == "__main__":
    best_loss = adversarial_finetune_decoder()
    logger.info(f"Final best no-skip reconstruction loss: {best_loss:.6f}")
    logger.info(f"Current Date and Time (UTC): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"User: SajbenDani")