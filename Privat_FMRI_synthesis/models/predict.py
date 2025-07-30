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

# Add parent directory to path
PARENT_DIR = Path(__file__).parent.parent
sys.path.append(str(PARENT_DIR))

# Import the models
from models.autoencoder import Improved3DAutoencoder
from models.diffusion import DiffusionUNet3D

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTOENCODER_PATH = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_finetuned/best_finetuned_autoencoder.pt"
DIFFUSION_PATH = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_diffusion/best_diffusion.pt"
OUTPUT_DIR = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/logs"
ORIGINAL_DATA_DIR = Path("/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri")

# Model parameters
LATENT_CHANNELS = 8
BASE_CHANNELS = 32
SCALE_FACTOR = 2
NUM_INFERENCE_STEPS = 20  # Reduced for faster inference
PATCH_SIZE = 64  # Single value for cubic patches

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

def load_models_with_custom_mapping():
    """Load autoencoder with custom weight mapping for the new architecture."""
    print("Loading models...")
    
    # Initialize the new model architecture
    autoencoder = Improved3DAutoencoder(
        in_channels=1,
        latent_channels=LATENT_CHANNELS,
        base_channels=BASE_CHANNELS,
        use_vq=True
    ).to(DEVICE)
    
    try:
        # Load the old state dictionary
        checkpoint = torch.load(AUTOENCODER_PATH, map_location=DEVICE, weights_only=False)
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
        autoencoder.load_state_dict(new_state_dict, strict=False)
        autoencoder.eval()
        print("Autoencoder loaded with custom weight mapping.")
        
        # Load diffusion model normally
        diffusion_model = DiffusionUNet3D(
            latent_channels=LATENT_CHANNELS,
            base_channels=128,
            time_emb_dim=256
        ).to(DEVICE)
        
        diffusion_ckpt = torch.load(DIFFUSION_PATH, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in diffusion_ckpt:
            diffusion_model.load_state_dict(diffusion_ckpt['model_state_dict'])
        else:
            diffusion_model.load_state_dict(diffusion_ckpt)
        diffusion_model.eval()
        print("Diffusion model loaded.")
        
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        
        print("Models loaded successfully.")
        return autoencoder, diffusion_model, noise_scheduler
        
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

class PureDiffusionSR:
    """Super-resolution pipeline class for sliding window inference with no skip connections."""
    def __init__(self, autoencoder, diffusion_model, noise_scheduler, num_steps=20):
        self.autoencoder = autoencoder
        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler.set_timesteps(num_steps)
        self.num_steps = num_steps
    
    @torch.no_grad()
    def __call__(self, x):
        """Process a low-resolution patch and return super-resolved patch.
        
        Args:
            x: Low-resolution patch tensor [B, C, D, H, W]
            
        Returns:
            Super-resolved patch tensor [B, C, D*2, H*2, W*2]
        """
        # 1. Encode the LR image to get the latent
        z_lr, _, _ = self.autoencoder.encode(x)
        
        # 2. Calculate the target HR latent shape (latent is 8x smaller than image space)
        hr_latent_shape = (
            x.shape[2] * SCALE_FACTOR // 8,  # depth
            x.shape[3] * SCALE_FACTOR // 8,  # height
            x.shape[4] * SCALE_FACTOR // 8,  # width
        )
        
        # 3. Upsample LR latent to HR latent size for conditioning
        z_lr_upsampled = F.interpolate(z_lr, size=hr_latent_shape, mode='trilinear')
        
        # 4. Initialize with random noise
        latents = torch.randn((x.shape[0], LATENT_CHANNELS, *hr_latent_shape), device=x.device)
        
        # 5. Denoise with diffusion model 
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.diffusion_model(latents, t.unsqueeze(0).to(x.device), z_lr_upsampled)
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        # 6. Decode latents to image space WITHOUT any skip connections
        # Pass None to use the no-skip path in the decoder
        sr_patch = self.autoencoder.decode(latents, skip_features=None)
        
        return sr_patch

def extract_patch(data, center=None, size=64):
    """Extract a cubic patch from the data."""
    if center is None:
        center = [dim // 2 for dim in data.shape]
    
    half_size = size // 2
    h_start = max(0, center[0] - half_size)
    h_end = min(data.shape[0], center[0] + half_size)
    
    w_start = max(0, center[1] - half_size)
    w_end = min(data.shape[1], center[1] + half_size)
    
    d_start = max(0, center[2] - half_size)
    d_end = min(data.shape[2], center[2] + half_size)
    
    # Extract patch
    patch = data[h_start:h_end, w_start:w_end, d_start:d_end]
    
    # Pad if necessary
    if patch.shape[0] < size or patch.shape[1] < size or patch.shape[2] < size:
        padded = np.zeros((size, size, size), dtype=patch.dtype)
        padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
        return padded
    
    return patch

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load models with custom weight mapping
    autoencoder, diffusion_model, noise_scheduler = load_models_with_custom_mapping()
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
    
    # Average over time if 4D
    if hr_data.ndim == 4:
        hr_data = np.mean(hr_data, axis=3)
    
    # Normalize to [0, 1]
    hr_data = (hr_data - hr_data.min()) / (hr_data.max() - hr_data.min())
    print(f"Original full volume shape: {hr_data.shape}")

    # Extract patch to make processing faster
    print("Extracting central patch for processing...")
    hr_patch_data = extract_patch(hr_data, size=PATCH_SIZE)
    print(f"Extracted patch shape: {hr_patch_data.shape}")

    # Convert to tensor
    hr_tensor = torch.from_numpy(hr_patch_data).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    print(f"HR Tensor Shape (B, C, H, W, D): {hr_tensor.shape}")

    # Create low-resolution version
    print("Creating synthetic low-resolution image...")
    lr_tensor = F.interpolate(hr_tensor, scale_factor=1/SCALE_FACTOR, mode='trilinear', align_corners=False)
    print(f"LR Tensor Shape: {lr_tensor.shape}")

    # Create upscaled low-resolution for comparison
    lr_upscaled = F.interpolate(lr_tensor, size=hr_tensor.shape[2:], mode='trilinear', align_corners=False)
    
    # Initialize SR pipeline
    print("Initializing diffusion super-resolution pipeline...")
    sr_pipeline = PureDiffusionSR(autoencoder, diffusion_model, noise_scheduler, NUM_INFERENCE_STEPS)

    # Process the patch
    print("Running diffusion super-resolution...")
    sr_tensor = sr_pipeline(lr_tensor)
    print(f"SR Tensor Shape: {sr_tensor.shape}")

    # Convert tensors to numpy for visualization
    hr_np = hr_tensor.squeeze().cpu().numpy()
    lr_np = lr_upscaled.squeeze().cpu().numpy()
    sr_np = sr_tensor.squeeze().cpu().numpy()
    
    # Save visualizations
    basename = hr_file_path.name.split('.')[0]
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_slices(hr_np, Path(OUTPUT_DIR) / f"{basename}_{timestamp}_1_original_hr.png", 
                "Original High-Resolution")
    plot_slices(lr_np, Path(OUTPUT_DIR) / f"{basename}_{timestamp}_2_lowres.png", 
                "Low-Resolution (Upscaled)")
    plot_slices(sr_np, Path(OUTPUT_DIR) / f"{basename}_{timestamp}_3_superres.png", 
                "Super-Resolved with Diffusion (No Skip Connections)")
    
    print("Prediction complete! Check the logs directory for the three comparison images.")

if __name__ == "__main__":
    main()