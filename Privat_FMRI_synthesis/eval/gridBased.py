import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
import torch
from diffusers import DDPMScheduler

# Add parent directory to system path for relative imports
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

# Custom modules
from models.diffusion import ConditionalDiffusion
from models.autoencoder import fMRIAutoencoder
from config import *

# Custom colormap: black to light green
colors = [(0, 0, 0), (0, 1, 0.5)]
cmap_name = 'custom_green'
colormaps.register(LinearSegmentedColormap.from_list(cmap_name, colors, N=256))

# ----- Use device from config -----
device = torch.device("cpu")  # force CPU, can use `DEVICE` if GPU is desired

# ----- Load models using config -----
diffusion_model = ConditionalDiffusion(num_classes=NUM_CLASSES).to(device)
autoencoder = fMRIAutoencoder(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)

if os.path.exists(DIFFUSION_CHECKPOINT):
    diffusion_model.load_state_dict(torch.load(DIFFUSION_CHECKPOINT, map_location=device))
    print(" Loaded diffusion checkpoint")
else:
    raise FileNotFoundError(f"Diffusion checkpoint not found at {DIFFUSION_CHECKPOINT}")

if os.path.exists(AUTOENCODER_CHECKPOINT):
    autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT, map_location=device))
    print(" Loaded autoencoder checkpoint")
else:
    raise FileNotFoundError(f"Autoencoder checkpoint not found at {AUTOENCODER_CHECKPOINT}")

diffusion_model.eval()
autoencoder.eval()

# ----- Diffusion Scheduler -----
scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
scheduler.set_timesteps(50)

# ----- Visualization Output Directory -----
vis_dir = os.path.join("visualizations", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(vis_dir, exist_ok=True)

# Slice locations (2x4 grid along W, which is originally D)
depth_size = 91  # After permutation: W axis
slice_indices = np.linspace(7, depth_size - 7, 8, dtype=int)

# ----- Sampling function -----
def sample_diffusion(model, labels, steps=50):
    with torch.no_grad():
        x = torch.randn(1, 1, SPATIAL_DIM, SPATIAL_DIM, device=device)
        for t in scheduler.timesteps:
            timestep = torch.tensor([t], device=device).expand(1)
            noise_pred = model(x, timestep, labels)
            x = scheduler.step(noise_pred, t, x).prev_sample
        return x

# ----- Visualize all class labels -----
for label in range(NUM_CLASSES):
    label_dir = os.path.join(vis_dir, f'label_{label}')
    os.makedirs(label_dir, exist_ok=True)

    labels = torch.tensor([label], device=device)

    with torch.no_grad():
        latent = sample_diffusion(diffusion_model, labels)
        latent = latent.view(1, -1)

        dummy_input = torch.zeros(1, 1, 91, 109, 91, device=device)
        _ = autoencoder(dummy_input, labels)  # warm up
        recon = autoencoder.decoder(torch.cat([latent, autoencoder.label_embed(labels)], dim=-1))

        recon_permuted = recon.permute(0, 1, 4, 2, 3)
        recon_np = recon_permuted[0, 0].detach().cpu().numpy()

        # Maximum intensity projection
        mip = np.max(recon_np, axis=0)
        mip = (mip - mip.min()) / (mip.max() - mip.min() + 1e-8)
        plt.figure(figsize=(5, 5))
        plt.imshow(mip, cmap='custom_green', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(os.path.join(label_dir, f'label_{label}_mip.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        # 2x4 slice grid
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, slice_idx in zip(axes.flat, slice_indices):
            slice_data = recon_np[slice_idx, :, :]
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
            ax.imshow(slice_data, cmap='custom_green', vmin=0, vmax=1)
            ax.set_title(f'Depth {slice_idx}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(label_dir, f'label_{label}_grid.png'), bbox_inches='tight')
        plt.close()

print(f"🎨 Visualization complete. Saved to: {vis_dir}")
