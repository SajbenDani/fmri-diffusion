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
from models.autoencoder import Improved3DAutoencoder  # Import from file
from models.skipPredictor import SkipPredictor  # Import from file
from utils.dataset import FMRIDataModule
from config import *


# Define custom colormap: black to light green
colors = [(0, 0, 0), (0, 1, 0.5)]
n_bins = 256
cmap_name = 'custom_green'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
colormaps.register(cm)

# Plotting Functions
def plot_slices(data, ax_row):
    depth = data.shape[0]  # W after permutation (91)
    step = depth // 8
    indices = [i * step for i in range(8)]  # e.g., [0, 11, 22, 33, 45, 56, 67, 78]
    for i, idx in enumerate(indices):
        slice_data = data[idx]
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
        ax_row[i].imshow(slice_data, cmap='custom_green', vmin=0, vmax=1)
        ax_row[i].set_title(f"Depth {idx}")
        ax_row[i].axis('off')

def plot_mip(data, ax):
    mip = np.max(data, axis=0)  # Max along W (91) -> [91, 109]
    mip = (mip - mip.min()) / (mip.max() - mip.min() + 1e-8)
    ax.imshow(mip, cmap='custom_green', vmin=0, vmax=1)
    ax.set_title("MIP")
    ax.axis('off')

def main():
    # Create timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_DIR = os.path.join(BASE_LOG_DIR, f"autoencoder_test_{timestamp}")
    os.makedirs(LOG_DIR, exist_ok=True)
    ORIGINAL_DIR = os.path.join(LOG_DIR, "original_plots")
    RECONSTRUCTED_DIR = os.path.join(LOG_DIR, "reconstructed_autoencoder_plots")
    PREDICTED_DIR = os.path.join(LOG_DIR, "reconstructed_predicted_plots")
    os.makedirs(ORIGINAL_DIR, exist_ok=True)
    os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)
    os.makedirs(PREDICTED_DIR, exist_ok=True)
    print(f"Logging visualizations to {LOG_DIR}")

    # Load the Trained Autoencoder
    autoencoder = Improved3DAutoencoder(latent_dims=LATENT_SHAPE, num_classes=NUM_CLASSES).to(DEVICE)
    autoencoder.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    autoencoder.eval()
    print(f"Loaded autoencoder from {BEST_MODEL_PATH}")

    # Load the Trained Skip Predictor
    skip_predictor = SkipPredictor(latent_dims=LATENT_SHAPE).to(DEVICE)
    skip_predictor.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    skip_predictor.eval()
    print(f"Loaded skip predictor from {BEST_MODEL_PATH}")

    # DataModule for Test Data
    data_module = FMRIDataModule(
        train_csv=TEST_CSV,
        val_csv=TEST_CSV,
        test_csv=TEST_CSV,
        data_dir=DATA_DIR,
        batch_size=1,
        num_workers=0  # Single-threaded to avoid multiprocessing issues
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # Get one sample
    fmri_tensor, labels = next(iter(test_loader))  # Take the first batch (batch_size=1)
    fmri_tensor = fmri_tensor.to(DEVICE)  # [1, 1, 91, 109, 91]
    labels = labels.to(DEVICE)  # [1]
    label_value = labels.item()  # For labeling the plot

    # Autoencoder Reconstruction (with true skip connections)
    with torch.no_grad():
        # Convert labels to one-hot for Improved3DAutoencoder
        labels_one_hot = torch.zeros(1, NUM_CLASSES, device=DEVICE)
        labels_one_hot[0, label_value] = 1
        recon_autoencoder, _ = autoencoder(fmri_tensor, labels_one_hot)  # [1, 1, 91, 109, 91]

    # Reconstruction with Skip Predictor
    with torch.no_grad():
        # Encode to get latent and true skip connections
        z, latent_3d, e1_true, e2_true = autoencoder.encode(fmri_tensor)
        # Predict skip connections
        pred_e1, pred_e2 = skip_predictor(
            latent_3d.unsqueeze(1),  # [1, 1, 8, 8, 8]
            target_shape_e1=e1_true.shape[2:],  # e1 shape: (46, 55, 46)
            target_shape_e2=e2_true.shape[2:]   # e2 shape: (23, 28, 23)
        )
        # Modulate latent with label features
        label_features = autoencoder.label_embedding(labels_one_hot)
        modulated_z = z * torch.sigmoid(label_features)
        # Decode with predicted skip connections
        recon_predicted = autoencoder.decode(modulated_z, pred_e1, pred_e2)  # [1, 1, 91, 109, 91]

    # Convert to NumPy for plotting (apply permutation for visualization)
    original = fmri_tensor.squeeze(0).squeeze(0).cpu().numpy()  # [91, 109, 91] = [D, H, W]
    recon_autoencoder = recon_autoencoder.squeeze(0).squeeze(0).cpu().numpy()  # [91, 109, 91] = [D, H, W]
    recon_predicted = recon_predicted.squeeze(0).squeeze(0).cpu().numpy()  # [91, 109, 91] = [D, H, W]

    # Apply permutation (2, 0, 1) -> [W, D, H]
    original = np.transpose(original, PERMUTE_ORDER)  # [91, 91, 109]
    recon_autoencoder = np.transpose(recon_autoencoder, PERMUTE_ORDER)  # [91, 91, 109]
    recon_predicted = np.transpose(recon_predicted, PERMUTE_ORDER)  # [91, 91, 109]

    # Plot Original
    fig, axes = plt.subplots(1, 9, figsize=(20, 2.5), gridspec_kw={'width_ratios': [1]*8 + [1.5]})
    plot_slices(original, axes[:-1])
    plot_mip(original, axes[-1])
    plt.suptitle(f"Original fMRI (Label: {label_value})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    original_file = os.path.join(ORIGINAL_DIR, f"original_label_{label_value}.png")
    plt.savefig(original_file)
    plt.close()

    # Plot Autoencoder Reconstruction (true skip connections)
    fig, axes = plt.subplots(1, 9, figsize=(20, 2.5), gridspec_kw={'width_ratios': [1]*8 + [1.5]})
    plot_slices(recon_autoencoder, axes[:-1])
    plot_mip(recon_autoencoder, axes[-1])
    plt.suptitle(f"Reconstructed fMRI - Autoencoder (Label: {label_value})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    recon_autoencoder_file = os.path.join(RECONSTRUCTED_DIR, f"recon_autoencoder_label_{label_value}.png")
    plt.savefig(recon_autoencoder_file)
    plt.close()

    # Plot Predicted Skip Connection Reconstruction
    fig, axes = plt.subplots(1, 9, figsize=(20, 2.5), gridspec_kw={'width_ratios': [1]*8 + [1.5]})
    plot_slices(recon_predicted, axes[:-1])
    plot_mip(recon_predicted, axes[-1])
    plt.suptitle(f"Reconstructed fMRI - Predicted Skip Connections (Label: {label_value})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    recon_predicted_file = os.path.join(PREDICTED_DIR, f"recon_predicted_label_{label_value}.png")
    plt.savefig(recon_predicted_file)
    plt.close()

    print("Visual evaluation completed.")
    print(f"Original plot saved to {original_file}")
    print(f"Autoencoder reconstructed plot saved to {recon_autoencoder_file}")
    print(f"Predicted skip reconstructed plot saved to {recon_predicted_file}")

if __name__ == '__main__':
    main()