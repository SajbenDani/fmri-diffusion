import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
import torch
from diffusers import DDPMScheduler
import sys

# Add parent directory to the system path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)
# Import models from their respective files
from models.diffusion import ConditionalDiffusion
from models.autoencoder import fMRIAutoencoder

# Define custom colormap: black to light green
colors = [(0, 0, 0), (0, 1, 0.5)]  # [black, light green]
n_bins = 256
cmap_name = 'custom_green'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
colormaps.register(cm)

# Set device to CPU
device = torch.device('cpu')

# Define constants
latent_dim = 256
spatial_dim = int(latent_dim ** 0.5)  # 16
CHECKPOINTS_DIR = "C:\\Users\\sajbe\\Documents\\onLab\\fmri-diffusion\\Privat_FMRI_synthesis\\checkpoints"
DIFFUSION_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'diffusion_model.pth')
AUTOENCODER_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'autoencoder.pth')

# Initialize models
diffusion_model = ConditionalDiffusion().to(device)
autoencoder = fMRIAutoencoder(latent_dim=latent_dim).to(device)

# Load checkpoints
if os.path.exists(DIFFUSION_CHECKPOINT):
    diffusion_model.load_state_dict(torch.load(DIFFUSION_CHECKPOINT, map_location=device))
    print("Loaded diffusion checkpoint")
if os.path.exists(AUTOENCODER_CHECKPOINT):
    autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT, map_location=device))
    print("Loaded autoencoder checkpoint")

diffusion_model.eval()
autoencoder.eval()

# Initialize scheduler for diffusion sampling
scheduler = DDPMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(50)  # 50 inference steps

# Visualization setup
vis_dir = os.path.join('visualizations', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(vis_dir, exist_ok=True)

# 2x4 grid: 8 evenly spaced slices along W (originally D, post-permutation)
depth_size = 91  # W dimension after permutation
slice_indices = np.linspace(7, depth_size-7, 8, dtype=int)  # e.g., [7, 17, 27, 37, 47, 57, 67, 77]

# Sampling function for ConditionalDiffusion
def sample_diffusion(model, labels, steps=50):
    with torch.no_grad():
        x = torch.randn(1, 1, spatial_dim, spatial_dim, device=device)  # [1, 1, 16, 16]
        for t in scheduler.timesteps:
            timestep = torch.tensor([t], device=device).expand(1)
            noise_pred = model(x, timestep, labels)
            x = scheduler.step(noise_pred, t, x).prev_sample
        return x

# Process all 5 labels
for label in range(5):
    label_dir = os.path.join(vis_dir, f'label_{label}')
    os.makedirs(label_dir, exist_ok=True)

    labels = torch.tensor([label], device=device)  # For embedding, not one-hot

    with torch.no_grad():
        # Generate latent via diffusion
        latent = sample_diffusion(diffusion_model, labels)  # [1, 1, 16, 16]
        latent = latent.view(1, -1)  # [1, 256]

        # Dummy input for encoder (only need latent for decoding)
        dummy_input = torch.zeros(1, 1, 91, 109, 91, device=device)
        recon = autoencoder(dummy_input, labels)  # Just to set up graph
        recon = autoencoder.decoder(torch.cat([latent, autoencoder.label_embed(labels)], dim=-1))  # [1, 1, 91, 109, 91]

        # Permute: (2, 0, 1) for [D, H, W] -> [W, D, H]
        recon_permuted = recon.permute(0, 1, 4, 2, 3)  # [1, 1, 91, 91, 109]
        recon_np = recon_permuted[0, 0].detach().cpu().numpy()  # [91, 91, 109] = [W, D, H]

        # 1. MIP-like representation (maximum projection along W)
        mip = np.max(recon_np, axis=0)  # [91, 109] = [D, H]
        mip = (mip - mip.min()) / (mip.max() - mip.min() + 1e-8)
        plt.figure(figsize=(5, 5))
        plt.imshow(mip, cmap='custom_green', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(os.path.join(label_dir, f'label_{label}_mip.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        # 2. 2x4 Grid of slices
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for idx, (ax, slice_idx) in enumerate(zip(axes.flat, slice_indices)):
            slice_data = recon_np[slice_idx, :, :]  # [D, H] = [91, 109]
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
            ax.imshow(slice_data, cmap='custom_green', vmin=0, vmax=1)
            ax.set_title(f'Depth {slice_idx}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(label_dir, f'label_{label}_grid.png'), bbox_inches='tight')
        plt.close()

print(f"Visualization completed. Results saved in {vis_dir}")