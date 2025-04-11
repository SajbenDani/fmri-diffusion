import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps

import torch
from torch.utils.data import DataLoader

# Add parent directory to system path for imports
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

# Local imports
from models.autoencoder import fMRIAutoencoder
from utils.dataset import FMRIDataModule
from config import (
    DEVICE, NUM_CLASSES, LATENT_DIM, BATCH_SIZE,
    BASE_LOG_DIR, CHECKPOINT_DIR, TEST_CSV,
    DATA_DIR, AUTOENCODER_CHECKPOINT, PERMUTE_ORDER
)

# Define and register a custom colormap: black to light green
colors = [(0, 0, 0), (0, 1, 0.5)]
n_bins = 256
cmap_name = 'custom_green'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
colormaps.register(cm)

# Plot multiple slices at selected depths
def plot_slices(data, ax_row):
    depth = data.shape[0]  # W after permutation
    step = depth // 8
    indices = [i * step for i in range(8)]  # 8 evenly spaced slices
    for i, idx in enumerate(indices):
        slice_data = data[idx]
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
        ax_row[i].imshow(slice_data, cmap='custom_green', vmin=0, vmax=1)
        ax_row[i].set_title(f"Depth {idx}")
        ax_row[i].axis('off')

# Plot maximum intensity projection
def plot_mip(data, ax):
    mip = np.max(data, axis=0)  # Collapse along W-axis -> shape [91, 109]
    mip = (mip - mip.min()) / (mip.max() - mip.min() + 1e-8)
    ax.imshow(mip, cmap='custom_green', vmin=0, vmax=1)
    ax.set_title("MIP")
    ax.axis('off')

def main():
    # Create timestamped log directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_DIR = os.path.join(BASE_LOG_DIR, f"autoencoder_test_{timestamp}")
    os.makedirs(LOG_DIR, exist_ok=True)

    # Subdirectories for original and reconstructed images
    ORIGINAL_DIR = os.path.join(LOG_DIR, "original_plots")
    RECONSTRUCTED_DIR = os.path.join(LOG_DIR, "reconstructed_autoencoder_plots")
    os.makedirs(ORIGINAL_DIR, exist_ok=True)
    os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)
    print(f"Logging visualizations to {LOG_DIR}")

    # Load trained autoencoder model
    autoencoder = fMRIAutoencoder(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT, map_location=DEVICE))
    autoencoder.eval()
    print(f"Loaded autoencoder from {AUTOENCODER_CHECKPOINT}")

    # Load dataset for testing
    data_module = FMRIDataModule(
        train_csv=TEST_CSV,
        val_csv=TEST_CSV,
        test_csv=TEST_CSV,
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0  # Set to 0 for testing to avoid multiprocessing issues
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # Take first test sample
    fmri_tensor, labels = next(iter(test_loader))
    fmri_tensor = fmri_tensor.to(DEVICE)  # Shape: [1, 1, 91, 109, 91]
    labels = labels.to(DEVICE)
    label_value = labels.item()  # Scalar for display

    # Get autoencoder reconstruction
    with torch.no_grad():
        recon_autoencoder = autoencoder(fmri_tensor, labels)

    # Convert tensors to NumPy arrays and remove singleton dimensions
    original = fmri_tensor.squeeze(0).squeeze(0).cpu().numpy()  # [D, H, W]
    recon_autoencoder = recon_autoencoder.squeeze(0).squeeze(0).cpu().numpy()

    # Permute axes for visualization: [D, H, W] -> [W, D, H]
    original = np.transpose(original, PERMUTE_ORDER)
    recon_autoencoder = np.transpose(recon_autoencoder, PERMUTE_ORDER)

    # Plot original fMRI
    fig, axes = plt.subplots(1, 9, figsize=(20, 2.5), gridspec_kw={'width_ratios': [1]*8 + [1.5]})
    plot_slices(original, axes[:-1])
    plot_mip(original, axes[-1])
    plt.suptitle(f"Original fMRI (Label: {label_value})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    original_file = os.path.join(ORIGINAL_DIR, f"original_label_{label_value}.png")
    plt.savefig(original_file)
    plt.close()

    # Plot reconstructed fMRI
    fig, axes = plt.subplots(1, 9, figsize=(20, 2.5), gridspec_kw={'width_ratios': [1]*8 + [1.5]})
    plot_slices(recon_autoencoder, axes[:-1])
    plot_mip(recon_autoencoder, axes[-1])
    plt.suptitle(f"Reconstructed fMRI - Autoencoder (Label: {label_value})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    recon_file = os.path.join(RECONSTRUCTED_DIR, f"recon_autoencoder_label_{label_value}.png")
    plt.savefig(recon_file)
    plt.close()

    print("Visual evaluation completed.")
    print(f"Original plot saved to: {original_file}")
    print(f"Autoencoder reconstructed plot saved to: {recon_file}")

if __name__ == '__main__':
    main()
