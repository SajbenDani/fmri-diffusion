import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
import torch
from torch.utils.data import DataLoader
import sys

# Add parent directory to the system path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

# Import necessary components
from models.autoencoder import fMRIAutoencoder
from utils.dataset import FMRIDataModule
from config import *

# Define custom colormap
colors = [(0, 0, 0), (0, 1, 0.5)]  # Black to light green
n_bins = 256
cm = LinearSegmentedColormap.from_list('custom_green', colors, N=n_bins)
colormaps.register(cm)

# Permutation for visualization (swap dimensions)
PERMUTE_ORDER = (2, 0, 1)  # [D, H, W] → [W, D, H]

# Plotting Functions
def plot_slices(data, ax_row):
    """Plots 8 evenly spaced slices from the fMRI scan."""
    depth = data.shape[0]
    step = depth // 8
    indices = [i * step for i in range(8)]
    for i, idx in enumerate(indices):
        slice_data = (data[idx] - data[idx].min()) / (data[idx].max() - data[idx].min() + 1e-8)
        ax_row[i].imshow(slice_data, cmap='custom_green', vmin=0, vmax=1)
        ax_row[i].set_title(f"Depth {idx}")
        ax_row[i].axis('off')

def plot_mip(data, ax):
    """Plots Maximum Intensity Projection (MIP)."""
    mip = np.max(data, axis=0)
    mip = (mip - mip.min()) / (mip.max() - mip.min() + 1e-8)
    ax.imshow(mip, cmap='custom_green', vmin=0, vmax=1)
    ax.set_title("MIP")
    ax.axis('off')

def main():
    """Evaluates the autoencoder on test data and saves visualizations."""
    # Create timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_DIR = os.path.join(LOGS_DIR, f"autoencoder_test_{timestamp}")
    os.makedirs(LOG_DIR, exist_ok=True)
    ORIGINAL_DIR = os.path.join(LOG_DIR, "original_plots")
    RECONSTRUCTED_DIR = os.path.join(LOG_DIR, "reconstructed_autoencoder_plots")
    os.makedirs(ORIGINAL_DIR, exist_ok=True)
    os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)
    print(f"Logging visualizations to {LOG_DIR}")

    # Load the trained autoencoder
    autoencoder = fMRIAutoencoder().to(DEVICE)
    autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT, map_location=DEVICE))
    autoencoder.eval()
    print(f"Loaded autoencoder from {AUTOENCODER_CHECKPOINT}")

    # DataModule for Test Data
    data_module = FMRIDataModule(
        train_csv=TEST_DATA,
        val_csv=TEST_DATA,
        test_csv=TEST_DATA,
        data_dir=DATA_DIR,
        batch_size=1,  # Always use batch size 1 for visualization
        num_workers=0  
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # Get one sample
    fmri_tensor, labels = next(iter(test_loader))
    fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
    label_value = labels.item()

    # Autoencoder Reconstruction
    with torch.no_grad():
        recon_autoencoder = autoencoder(fmri_tensor, labels)

    # Convert to NumPy for plotting
    original = np.transpose(fmri_tensor.squeeze().cpu().numpy(), PERMUTE_ORDER)
    recon_autoencoder = np.transpose(recon_autoencoder.squeeze().cpu().numpy(), PERMUTE_ORDER)

    # Plot Original fMRI
    fig, axes = plt.subplots(1, 9, figsize=(20, 2.5), gridspec_kw={'width_ratios': [1]*8 + [1.5]})
    plot_slices(original, axes[:-1])
    plot_mip(original, axes[-1])
    plt.suptitle(f"Original fMRI (Label: {label_value})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    original_file = os.path.join(ORIGINAL_DIR, f"original_label_{label_value}.png")
    plt.savefig(original_file)
    plt.close()

    # Plot Reconstructed fMRI
    fig, axes = plt.subplots(1, 9, figsize=(20, 2.5), gridspec_kw={'width_ratios': [1]*8 + [1.5]})
    plot_slices(recon_autoencoder, axes[:-1])
    plot_mip(recon_autoencoder, axes[-1])
    plt.suptitle(f"Reconstructed fMRI - Autoencoder (Label: {label_value})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    recon_autoencoder_file = os.path.join(RECONSTRUCTED_DIR, f"recon_autoencoder_label_{label_value}.png")
    plt.savefig(recon_autoencoder_file)
    plt.close()

    print(f"Original plot saved to {original_file}")
    print(f"Autoencoder reconstructed plot saved to {recon_autoencoder_file}")
    print("Evaluation complete.")

if __name__ == '__main__':
    main()
