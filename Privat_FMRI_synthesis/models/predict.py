#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from diffusers import DDPMScheduler
from monai.inferers import sliding_window_inference

# --- Setup Paths ---
PARENT_DIR = Path(__file__).parent.parent
sys.path.append(str(PARENT_DIR))

# Import the models
from models.autoencoder import Improved3DAutoencoder
from models.diffusion import DiffusionUNet3D

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use the adversarially trained autoencoder
AUTOENCODER_PATH = PARENT_DIR / "checkpoints_adversarial" / "best_finetuned_adversarial_autoencoder_second.pt"
# Use CFG-finetuned diffusion model
DIFFUSION_PATH = PARENT_DIR / "checkpoints_diffusion_cfg" / "best_diffusion_cfg_adversarial_aligned.pt"
OUTPUT_DIR = PARENT_DIR / "logs"
ORIGINAL_DATA_DIR = Path("/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri")

# Model parameters
LATENT_CHANNELS = 8
BASE_CHANNELS = 32
SCALE_FACTOR = 2
NUM_INFERENCE_STEPS = 1000  # Full 1000 steps for highest quality
ROI_SIZE_HR = (64, 64, 64) 
GUIDANCE_SCALE = 7.5 #7.5  # CFG guidance scale - a higher value can increase sharpness

# Define custom colormap
colors = [(0, 0, 0), (0, 1, 0.5)]
cm = LinearSegmentedColormap.from_list('custom_green', colors, N=256)

def plot_slices(data_np, output_path, title):
    """Plot 2x4 slices of 3D data. Assumes input shape is (H, W, D)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    depth = data_np.shape[2]
    slice_indices = np.linspace(0, depth - 1, 8, dtype=int)
    
    for i, slice_idx in enumerate(slice_indices):
        axes[i].imshow(data_np[:, :, slice_idx], cmap=cm, vmin=0, vmax=1)
        axes[i].set_title(f"Slice {slice_idx}")
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved visualization to: {output_path}")

def load_models():
    print("Loading models...")
    autoencoder = Improved3DAutoencoder(in_channels=1, latent_channels=LATENT_CHANNELS, base_channels=BASE_CHANNELS, use_vq=True).to(DEVICE)
    checkpoint = torch.load(AUTOENCODER_PATH, map_location=DEVICE)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()
    print(f"Loaded adversarial autoencoder from: {AUTOENCODER_PATH}")

    diffusion_model = DiffusionUNet3D(latent_channels=LATENT_CHANNELS, base_channels=128, time_emb_dim=256).to(DEVICE)
    diffusion_ckpt = torch.load(DIFFUSION_PATH, map_location=DEVICE)
    if 'model_state_dict' in diffusion_ckpt:
        diffusion_model.load_state_dict(diffusion_ckpt['model_state_dict'])
    else:
        diffusion_model.load_state_dict(diffusion_ckpt)
    diffusion_model.eval()
    print(f"Loaded CFG-trained diffusion model from: {DIFFUSION_PATH}")

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    print("Models loaded successfully.")
    return autoencoder, diffusion_model, noise_scheduler


class CFGDiffusionSR:
    """Super-resolution pipeline class with Classifier-Free Guidance."""
    def __init__(self, autoencoder, diffusion_model, noise_scheduler, num_steps=1000, guidance_scale=7.5):
        self.autoencoder = autoencoder
        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler.set_timesteps(num_steps)
        self.guidance_scale = guidance_scale
        print(f"Using guidance scale: {guidance_scale}")
        print(f"Using {num_steps} diffusion steps")
    
    @torch.no_grad()
    def __call__(self, lr_patch):
        # ... (logic before the loop is correct) ...
        z_lr, _, _ = self.autoencoder.encode(lr_patch)
        hr_latent_shape = (
            lr_patch.shape[2] * SCALE_FACTOR // 8,
            lr_patch.shape[3] * SCALE_FACTOR // 8,
            lr_patch.shape[4] * SCALE_FACTOR // 8
        )
        z_lr_upsampled = F.interpolate(z_lr, size=hr_latent_shape, mode='trilinear')
        null_conditioning = torch.zeros_like(z_lr_upsampled)
        latents = torch.randn((lr_patch.shape[0], LATENT_CHANNELS, *hr_latent_shape), device=DEVICE)
        
        for t in tqdm(self.noise_scheduler.timesteps, desc="Generating SR Patch", leave=False):
            # Convert scalar timestep to tensor
            t_tensor = torch.tensor([t], device=DEVICE)
            
            # Get unconditional prediction
            noise_pred_uncond = self.diffusion_model(latents, t_tensor, null_conditioning)
            
            # Get conditional prediction
            noise_pred_cond = self.diffusion_model(latents, t_tensor, z_lr_upsampled)
            
            # Combine with guidance scale
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Perform denoising step
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        sr_patch = self.autoencoder.decode(latents, skip_features=None)
        
        return sr_patch

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    autoencoder, diffusion_model, noise_scheduler = load_models()
    if autoencoder is None:
        return

    # Load original HR file
    sample_base_path = ORIGINAL_DATA_DIR / '100307' / 'tfMRI_MOTOR_RL'
    hr_file_path = None
    for ext in ['.nii', '.nii.gz']:
        potential_path = sample_base_path.with_suffix(ext)
        if potential_path.exists():
            hr_file_path = potential_path
            break
    if not hr_file_path:
        print(f"Error: Original fMRI file not found at {sample_base_path}")
        return

    print(f"Loading original HR file: {hr_file_path}")
    nii_img = nib.load(hr_file_path)
    hr_data = nii_img.get_fdata(dtype=np.float32)
    hr_data = np.mean(hr_data, axis=3) if hr_data.ndim == 4 else hr_data
    hr_data = (hr_data - hr_data.min()) / (hr_data.max() - hr_data.min())
    print(f"Original full volume shape (W, H, D): {hr_data.shape}")

    # Convert to tensor with shape (B, C, W, H, D)
    hr_tensor = torch.from_numpy(hr_data.copy()).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    print("Creating synthetic low-resolution volume...")
    lr_tensor = F.interpolate(hr_tensor, scale_factor=1/SCALE_FACTOR, mode='trilinear', align_corners=False)
    
    # Create upscaled low-resolution for comparison
    lr_upscaled = F.interpolate(lr_tensor, size=hr_tensor.shape[2:], mode='trilinear', align_corners=False)
    
    # Initialize the pipeline
    sr_pipeline = CFGDiffusionSR(
        autoencoder, 
        diffusion_model, 
        noise_scheduler, 
        NUM_INFERENCE_STEPS, 
        GUIDANCE_SCALE
    )
    
    # Extract a central portion to process - full volume is too memory intensive
    print("Processing central portion of the volume...")
    center = [dim // 2 for dim in hr_data.shape]
    region_size = min(96, min(hr_data.shape))  # Reasonable size for processing
    
    half_size = region_size // 2
    hr_region = hr_tensor[:, :, 
                          max(0, center[0] - half_size):min(hr_data.shape[0], center[0] + half_size),
                          max(0, center[1] - half_size):min(hr_data.shape[1], center[1] + half_size),
                          max(0, center[2] - half_size):min(hr_data.shape[2], center[2] + half_size)]
    
    # Create low-resolution version of the region
    lr_region = F.interpolate(hr_region, scale_factor=1/SCALE_FACTOR, mode='trilinear', align_corners=False)
    
    # Define the patch size for the *low-resolution* input volume
    roi_size_lr = tuple(s // SCALE_FACTOR for s in ROI_SIZE_HR)
    
    print(f"Running sliding window inference on region with LR patch size: {roi_size_lr}...")
    try:
        sr_region = sliding_window_inference(
            inputs=lr_region,
            roi_size=roi_size_lr,
            sw_batch_size=1,  # Process one window at a time to save memory
            predictor=sr_pipeline,
            overlap=0.5,
            mode="gaussian"
        )
        
        # Prepare images for plotting
        hr_np = hr_region.squeeze().cpu().numpy()
        lr_np = F.interpolate(lr_region, size=hr_region.shape[2:], mode='trilinear').squeeze().cpu().numpy()
        sr_np = sr_region.squeeze().cpu().numpy()
        
    except Exception as e:
        print(f"Error during sliding window inference: {e}")
        import traceback
        traceback.print_exc()
        
        # Fall back to a single central patch
        print("Falling back to single central patch...")
        
        # Extract central patch
        patch_size = min(64, min(hr_data.shape))
        patch_half = patch_size // 2
        
        hr_patch = hr_tensor[:, :, 
                             center[0] - patch_half:center[0] + patch_half,
                             center[1] - patch_half:center[1] + patch_half,
                             center[2] - patch_half:center[2] + patch_half]
        
        lr_patch = F.interpolate(hr_patch, scale_factor=1/SCALE_FACTOR, mode='trilinear', align_corners=False)
        
        # Process the single patch
        sr_patch = sr_pipeline(lr_patch)
        
        # Prepare images for plotting
        hr_np = hr_patch.squeeze().cpu().numpy()
        lr_np = F.interpolate(lr_patch, size=hr_patch.shape[2:], mode='trilinear').squeeze().cpu().numpy()
        sr_np = sr_patch.squeeze().cpu().numpy()
    
    # Save visualizations
    basename = hr_file_path.name.split('.')[0]
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_slices(hr_np, OUTPUT_DIR / f"{basename}_{timestamp}_1_original_HR.png", 
                "Original High-Resolution")
    plot_slices(lr_np, OUTPUT_DIR / f"{basename}_{timestamp}_2_input_LR.png", 
                "Low-Resolution Input (Upscaled)")
    plot_slices(sr_np, OUTPUT_DIR / f"{basename}_{timestamp}_3_superresolved.png", 
                f"Super-Resolved with CFG (Guidance: {GUIDANCE_SCALE})")
    
    print("\n=== PREDICTION COMPLETE ===")
    print("Generated three images for visual comparison:")
    print(f"1. Original HR: {OUTPUT_DIR}/{basename}_{timestamp}_1_original_HR.png")
    print(f"2. Low-res:     {OUTPUT_DIR}/{basename}_{timestamp}_2_input_LR.png")
    print(f"3. Super-res:   {OUTPUT_DIR}/{basename}_{timestamp}_3_superresolved.png")
    print(f"Current Date and Time (UTC): 2025-08-07 08:32:19")
    print(f"Current User's Login: SajbenDani")

if __name__ == "__main__":
    main()