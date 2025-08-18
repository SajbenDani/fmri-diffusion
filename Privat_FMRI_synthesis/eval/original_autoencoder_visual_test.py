#!/usr/bin/env python3
import os
import sys
import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn.functional as F
import nibabel as nib
from monai.inferers import sliding_window_inference

# --- Paths & Imports ---
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PARENT_DIR))

# Models and dataset
from models.autoencoder import Improved3DAutoencoder
from utils.dataset import FMRIDataModule

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_autoencoder/best_autoencoder.pt")

DATA_DIR = Path("/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/data_preprocessed/test")
TEST_CSV = Path("/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/data_preprocessed/test_patches.csv")
ORIGINAL_DATA_DIR = Path("/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri")

OUTPUT_DIR = PARENT_DIR / "eval" / "brain_reconstructions_original_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sliding window settings (same as predict)
ROI_SIZE = (64, 64, 64)
SW_BATCH_SIZE = 1
OVERLAP = 0.5
SW_MODE = "gaussian"

# Colormap
colors = [(0, 0, 0), (0, 1, 0.5)]
cm = LinearSegmentedColormap.from_list('custom_green', colors, N=256)

def plot_slices(data_np, output_path, title):
    """Plot 2x4 slices of 3D data. Assumes input shape is (H, W, D) for proper axial view."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    depth = data_np.shape[2]
    slice_indices = np.linspace(0, depth - 1, 8, dtype=int)
    
    for i, slice_idx in enumerate(slice_indices):
        # Rotate 90 degrees counter-clockwise for proper brain orientation (top view)
        slice_data = np.rot90(data_np[:, :, slice_idx], k=1)
        axes[i].imshow(slice_data, cmap=cm, vmin=0, vmax=1)
        axes[i].set_title(f"Slice {slice_idx}")
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved visualization to: {output_path}")


# Create a version of the old autoencoder that matches the checkpoint structure
class OriginalAutoencoder(torch.nn.Module):
    """Recreation of the original autoencoder that matches the checkpoint."""
    def __init__(self, in_channels=1, latent_channels=8, base_channels=32, use_vq=True, num_vq_embeddings=512):
        super().__init__()
        from models.autoencoder import ResidualBlock3D, DownBlock3D, VectorQuantizer
        
        # Same encoder as current model
        self.initial_conv = torch.nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc1 = ResidualBlock3D(base_channels, base_channels)
        self.enc2 = DownBlock3D(base_channels, base_channels*2)
        self.enc3 = DownBlock3D(base_channels*2, base_channels*4)
        self.enc4 = DownBlock3D(base_channels*4, base_channels*8)
        self.bottleneck = ResidualBlock3D(base_channels*8, latent_channels)
        
        self.use_vq = use_vq
        if use_vq:
            self.vq = VectorQuantizer(num_embeddings=num_vq_embeddings, embedding_dim=latent_channels)
        
        # Decoder with the OLD skip connection structure
        self.dec4 = ResidualBlock3D(latent_channels, base_channels*8)
        
        # OLD UpBlock structure - always expects skip connections
        self.dec3 = OldUpBlock3D(base_channels*8, base_channels*4, base_channels*8)  # expects skip
        self.dec2 = OldUpBlock3D(base_channels*4, base_channels*2, base_channels*4)  # expects skip  
        self.dec1 = OldUpBlock3D(base_channels*2, base_channels, base_channels*2)    # expects skip
        self.final_upsample = OldUpBlock3D(base_channels, base_channels, base_channels) # expects skip
        
        self.final = torch.nn.Sequential(
            ResidualBlock3D(base_channels, base_channels),
            torch.nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.initial_conv(x)
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        z = self.bottleneck(enc4_out)
        
        if self.use_vq:
            z_q, vq_loss, _ = self.vq(z)
            return z_q, vq_loss, [enc1_out, enc2_out, enc3_out, enc4_out]
        else:
            return z, None, [enc1_out, enc2_out, enc3_out, enc4_out]
    
    def decode(self, z, skip_features):
        dec4_out = self.dec4(z)
        
        if skip_features is not None and len(skip_features) == 4:
            enc1_out, enc2_out, enc3_out, enc4_out = skip_features
            dec3_out = self.dec3(dec4_out, enc4_out)
            dec2_out = self.dec2(dec3_out, enc3_out)
            dec1_out = self.dec1(dec2_out, enc2_out)
            final_dec = self.final_upsample(dec1_out, enc1_out)
        else:
            # This won't work well with old model, but let's try with zeros
            dummy_skip = torch.zeros_like(dec4_out)
            dec3_out = self.dec3(dec4_out, dummy_skip)
            dec2_out = self.dec2(dec3_out, torch.zeros_like(dec3_out))
            dec1_out = self.dec1(dec2_out, torch.zeros_like(dec2_out))
            final_dec = self.final_upsample(dec1_out, torch.zeros_like(dec1_out))
        
        out = self.final(final_dec)
        return out
    
    def forward(self, x):
        if self.use_vq:
            z, vq_loss, skip_features = self.encode(x)
        else:
            z, _, skip_features = self.encode(x)
            vq_loss = None
        
        recon = self.decode(z, skip_features)
        
        if self.use_vq:
            return recon, z, vq_loss
        else:
            return recon, z


class OldUpBlock3D(torch.nn.Module):
    """Old upsampling block that always expects skip connections."""
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        from models.autoencoder import ResidualBlock3D
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.res_block = ResidualBlock3D(in_channels + skip_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res_block(x)


def load_original_model():
    print(f"Loading original model from: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    # Read hyperparams if present, else use defaults
    latent_channels = ckpt.get('latent_channels', 8)
    base_channels = ckpt.get('base_channels', 32)
    use_vq = ckpt.get('use_vq', True)
    num_vq_embeddings = ckpt.get('num_vq_embeddings', 512)

    model = OriginalAutoencoder(
        in_channels=1,
        latent_channels=latent_channels,
        base_channels=base_channels,
        use_vq=use_vq,
        num_vq_embeddings=num_vq_embeddings
    ).to(DEVICE)

    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Load the state dict - should work perfectly now
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print(f"Successfully loaded model")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    model.eval()

    print(f"Model loaded. "
          f"latent_channels={latent_channels}, base_channels={base_channels}, "
          f"use_vq={use_vq}, num_vq_embeddings={num_vq_embeddings}")
    return model


def main():
    print(f"Using device: {DEVICE}")
    model = load_original_model()

    # Load original HR file (same approach as predict script)
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

    # Extract a central portion to process - full volume is too memory intensive
    print("Processing central portion of the volume...")
    center = [dim // 2 for dim in hr_data.shape]
    region_size = min(96, min(hr_data.shape))  # Reasonable size for processing
    
    half_size = region_size // 2
    hr_region = hr_tensor[:, :, 
                          max(0, center[0] - half_size):min(hr_data.shape[0], center[0] + half_size),
                          max(0, center[1] - half_size):min(hr_data.shape[1], center[1] + half_size),
                          max(0, center[2] - half_size):min(hr_data.shape[2], center[2] + half_size)]

    print(f"Processing region shape (B,C,W,H,D): {tuple(hr_region.shape)}")

    # Predictor for sliding_window_inference - using the original model with skip connections
    @torch.no_grad()
    def predictor_with_skips(x):
        # Use the original model WITH skip connections
        if model.use_vq:
            recon, _, _ = model(x)
        else:
            recon, _ = model(x)
        return recon
    
    # Also create a predictor without skip connections (using dummy zeros)
    @torch.no_grad()
    def predictor_no_skips(x):
        # Encode to get latent
        if model.use_vq:
            z, vq_loss, skip_features = model.encode(x)
        else:
            z, _, skip_features = model.encode(x)
        
        # Decode WITHOUT skip connections (pass None or zeros)
        recon = model.decode(z, skip_features=None)
        return recon

    print(f"Running sliding window inference with ROI size: {ROI_SIZE}...")
    
    # Test with skip connections
    try:
        print("Testing reconstruction WITH skip connections...")
        with torch.no_grad():
            recon_region_with_skips = sliding_window_inference(
                inputs=hr_region,
                roi_size=ROI_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=predictor_with_skips,
                overlap=OVERLAP,
                mode=SW_MODE
            )
        skip_success = True
    except Exception as e:
        print(f"Error with skip connections: {e}")
        skip_success = False
    
    # Test without skip connections  
    try:
        print("Testing reconstruction WITHOUT skip connections...")
        with torch.no_grad():
            recon_region_no_skips = sliding_window_inference(
                inputs=hr_region,
                roi_size=ROI_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=predictor_no_skips,
                overlap=OVERLAP,
                mode=SW_MODE
            )
        no_skip_success = True
    except Exception as e:
        print(f"Error without skip connections: {e}")
        no_skip_success = False
    
    if not skip_success and not no_skip_success:
        print("Both approaches failed, falling back to single patch...")
        # Fall back to single patch
        patch_size = min(64, min(hr_region.shape[2:]))
        patch_half = patch_size // 2
        region_center = [dim // 2 for dim in hr_region.shape[2:]]
        
        hr_patch = hr_region[:, :, 
                             max(0, region_center[0] - patch_half):min(hr_region.shape[2], region_center[0] + patch_half),
                             max(0, region_center[1] - patch_half):min(hr_region.shape[3], region_center[1] + patch_half),
                             max(0, region_center[2] - patch_half):min(hr_region.shape[4], region_center[2] + patch_half)]
        
        recon_patch = predictor_with_skips(hr_patch)
        hr_region = hr_patch
        recon_region_with_skips = recon_patch
        skip_success = True

    # Prepare images for plotting
    hr_np = hr_region.squeeze().cpu().numpy()  # (W, H, D)

    # Save visualizations
    basename = hr_file_path.name.split('.')[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_slices(hr_np, OUTPUT_DIR / f"{basename}_{timestamp}_1_original.png", 
                "Original fMRI")
    
    if skip_success:
        recon_with_skips_np = recon_region_with_skips.squeeze().cpu().numpy()
        recon_with_skips_np = np.clip(recon_with_skips_np, 0.0, 1.0)
        plot_slices(recon_with_skips_np, OUTPUT_DIR / f"{basename}_{timestamp}_2_reconstructed_with_skips.png", 
                    "Original Model WITH Skip Connections")
    
    if no_skip_success:
        recon_no_skips_np = recon_region_no_skips.squeeze().cpu().numpy()
        recon_no_skips_np = np.clip(recon_no_skips_np, 0.0, 1.0)
        plot_slices(recon_no_skips_np, OUTPUT_DIR / f"{basename}_{timestamp}_3_reconstructed_no_skips.png", 
                    "Original Model WITHOUT Skip Connections")
    
    print("\n=== AUTOENCODER VISUALIZATION COMPLETE ===")
    print("Generated images for comparison:")
    print(f"1. Original:        {OUTPUT_DIR}/{basename}_{timestamp}_1_original.png")
    if skip_success:
        print(f"2. With Skips:      {OUTPUT_DIR}/{basename}_{timestamp}_2_reconstructed_with_skips.png")
    if no_skip_success:
        print(f"3. Without Skips:   {OUTPUT_DIR}/{basename}_{timestamp}_3_reconstructed_no_skips.png")
    print(f"Current Date and Time (UTC): 2025-08-11 14:14:28")
    print(f"Current User's Login: SajbenDani")

if __name__ == "__main__":
    main()