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

MODEL_PATH = Path("/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_finetuned/best_finetuned_autoencoder.pt")

DATA_DIR = Path("/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/data_preprocessed/test")
TEST_CSV = Path("/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/data_preprocessed/test_patches.csv")
ORIGINAL_DATA_DIR = Path("/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri")

OUTPUT_DIR = PARENT_DIR / "eval" / "brain_reconstructions_adversial"
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

def load_model():
    print(f"Loading model from: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    # Read hyperparams if present, else use defaults for Improved3DAutoencoder
    latent_channels = ckpt.get('latent_channels', 8)
    base_channels = ckpt.get('base_channels', 32)
    use_vq = ckpt.get('use_vq', True)
    num_vq_embeddings = ckpt.get('num_vq_embeddings', 512)

    model = Improved3DAutoencoder(
        in_channels=1,
        latent_channels=latent_channels,
        base_channels=base_channels,
        use_vq=use_vq,
        num_vq_embeddings=num_vq_embeddings
    ).to(DEVICE)

    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Model loaded. "
          f"latent_channels={latent_channels}, base_channels={base_channels}, "
          f"use_vq={use_vq}, num_vq_embeddings={num_vq_embeddings}")
    return model

def main():
    print(f"Using device: {DEVICE}")
    model = load_model()

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

    # Predictor for sliding_window_inference
    @torch.no_grad()
    def predictor(x):
        # x is (B,C,W,H,D) patch
        # Encode to get latent representation
        if model.use_vq:
            z, vq_loss, skip_features = model.encode(x)
        else:
            z, _, skip_features = model.encode(x)
        
        # Decode WITHOUT skip connections by passing None
        recon = model.decode(z, skip_features=None)
        return recon

    print(f"Running sliding window inference with ROI size: {ROI_SIZE}...")
    try:
        # Full-region reconstruction via sliding window inference
        with torch.no_grad():
            recon_region = sliding_window_inference(
                inputs=hr_region,
                roi_size=ROI_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=predictor,
                overlap=OVERLAP,
                mode=SW_MODE
            )

        # Prepare images for plotting
        hr_np = hr_region.squeeze().cpu().numpy()  # (W, H, D)
        recon_np = recon_region.squeeze().cpu().numpy()  # (W, H, D)

    except Exception as e:
        print(f"Error during sliding window inference: {e}")
        import traceback
        traceback.print_exc()
        
        # Fall back to a single central patch
        print("Falling back to single central patch...")
        
        # Extract central patch
        patch_size = min(64, min(hr_region.shape[2:]))
        patch_half = patch_size // 2
        region_center = [dim // 2 for dim in hr_region.shape[2:]]
        
        hr_patch = hr_region[:, :, 
                             max(0, region_center[0] - patch_half):min(hr_region.shape[2], region_center[0] + patch_half),
                             max(0, region_center[1] - patch_half):min(hr_region.shape[3], region_center[1] + patch_half),
                             max(0, region_center[2] - patch_half):min(hr_region.shape[4], region_center[2] + patch_half)]
        
        # Process the single patch
        recon_patch = predictor(hr_patch)
        
        # Prepare images for plotting
        hr_np = hr_patch.squeeze().cpu().numpy()
        recon_np = recon_patch.squeeze().cpu().numpy()

    # Clamp reconstruction to [0,1]
    recon_np = np.clip(recon_np, 0.0, 1.0)

    # Save visualizations (same style as predict script)
    basename = hr_file_path.name.split('.')[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_slices(hr_np, OUTPUT_DIR / f"{basename}_{timestamp}_1_original.png", 
                "Original fMRI")
    plot_slices(recon_np, OUTPUT_DIR / f"{basename}_{timestamp}_2_reconstructed_skip_model_no_skips.png", 
                "Autoencoder Reconstruction (Skip Model - No Skip Connections Used)")
    
    print("\n=== AUTOENCODER VISUALIZATION COMPLETE ===")
    print("Generated two images for comparison:")
    print(f"1. Original:      {OUTPUT_DIR}/{basename}_{timestamp}_1_original.png")
    print(f"2. Reconstructed: {OUTPUT_DIR}/{basename}_{timestamp}_2_reconstructed_skip_model_no_skips.png")
    print(f"Current Date and Time (UTC): 2025-08-11 14:07:37")
    print(f"Current User's Login: SajbenDani")

if __name__ == "__main__":
    main()