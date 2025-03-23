import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps



# Add parent directory to the system path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)
# Import the provided model classes
from models.diffusion import LatentDiffusion
from models.autoencoder import Improved3DAutoencoder
from models.skipPredictor import SkipPredictor

# Define custom colormap: black to light green
colors = [(0, 0, 0), (0, 1, 0.5)]  # [black, light green]
n_bins = 256
cmap_name = 'custom_green'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
colormaps.register(cm)  # Register the colormap correctly

# Set device to CPU as specified
device = torch.device('cpu')

# Initialize and load the models
models_dir = "C:\\Users\\sajbe\\Documents\\onLab\\fmri-diffusion\\Privat_FMRI_synthesis\\models"
checkpoints_dir = "C:\\Users\\sajbe\\Documents\\onLab\\fmri-diffusion\\Privat_FMRI_synthesis\\checkpoints"

autoencoder = Improved3DAutoencoder(latent_dims=(8,8,8), num_classes=5).to(device)
autoencoder.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'finetuned_autoencoder_best.pth'), map_location=device))
autoencoder.eval()

diffusion = LatentDiffusion(latent_shape=(8,8,8), num_classes=5, device=device)
diffusion.model.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'latent_diffusion.pth'), map_location=device))
diffusion.model.eval()

skip_predictor = SkipPredictor(latent_dims=(8,8,8)).to(device)
skip_predictor.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'skip_predictor_best.pth'), map_location=device))
skip_predictor.eval()

# Set the original size for the autoencoder decoding
autoencoder.original_size = (96, 112, 96)  # [D, H, W]

# Define visualization directory with current datetime
vis_dir = os.path.join('visualizations', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(vis_dir, exist_ok=True)

# Diffusion sampling parameters
steps = 50

# Skip connection target shapes
target_shape_e1 = (48, 56, 48)  # For e1
target_shape_e2 = (24, 28, 24)  # For e2

# 2x4 grid: 8 evenly spaced slices along W (originally D, post-permutation)
depth_size = 96  # W dimension after permutation
slice_indices = np.linspace(8, depth_size-8, 8, dtype=int)  # e.g., [8, 19, 31, 42, 54, 65, 77, 88]

# Process all 5 labels
for label in range(5):
    # Create subdirectory for the label
    label_dir = os.path.join(vis_dir, f'label_{label}')
    os.makedirs(label_dir, exist_ok=True)

    # Create one-hot encoded label tensor
    labels = torch.zeros(1, 5, device=device)
    labels[0, label] = 1

    # Generate final latent sample and visualize
    with torch.no_grad():
        latent = diffusion.sample(labels, steps=steps)  # [1, 1, 8, 8, 8]

        # Predict skip connections
        pred_e1, pred_e2 = skip_predictor(latent, target_shape_e1, target_shape_e2)

        # Prepare latent for decoding
        z = latent.squeeze(1).view(1, -1)  # [1, 512]
        label_features = autoencoder.label_embedding(labels)
        modulated_z = z * torch.sigmoid(label_features)

        # Reconstruct
        recon = autoencoder.decode(modulated_z, pred_e1, pred_e2)  # [1, 1, 96, 112, 96]

        # Permute spatial dimensions: (2, 0, 1) for [D, H, W] -> [W, D, H]
        recon_permuted = recon.permute(0, 1, 4, 2, 3)  # [1, 1, 96, 96, 112]
        recon_np = recon_permuted[0, 0].detach().cpu().numpy()  # [96, 96, 112] = [W, D, H]

        # 1. MIP-like representation (maximum projection along W)
        mip = np.max(recon_np, axis=0)  # [96, 112] = [D, H]
        mip = (mip - mip.min()) / (mip.max() - mip.min() + 1e-8)  # Normalize to [0, 1]
        plt.figure(figsize=(5, 5))
        plt.imshow(mip, cmap='custom_green', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(os.path.join(label_dir, f'label_{label}_mip.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        # 2. 2x4 Grid of slices
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for idx, (ax, slice_idx) in enumerate(zip(axes.flat, slice_indices)):
            slice_data = recon_np[slice_idx, :, :]  # [D, H] = [96, 112]
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
            ax.imshow(slice_data, cmap='custom_green', vmin=0, vmax=1)
            ax.set_title(f'Depth {slice_idx}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(label_dir, f'label_{label}_grid.png'), bbox_inches='tight')
        plt.close()

print(f"Visualization completed. Results saved in {vis_dir}")