#!/usr/bin/env python3
import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import nibabel as nib
from monai.inferers import sliding_window_inference

# Add parent directory to path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

# Import the model
from models.autoencoder import Improved3DAutoencoder

# Define custom colormap (black to light green)
colors = [(0, 0, 0), (0, 1, 0.5)]
n_bins = 256
cm = LinearSegmentedColormap.from_list('custom_green', colors, N=n_bins)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(PARENT_DIR, "checkpoints", "best_autoencoder.pt")
# THIS IS THE DATA_DIR in the original script. Let's make sure it's correct.
# The path in the error message is '/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri'
DATA_DIR = "/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri" 
OUTPUT_DIR = os.path.join(PARENT_DIR, "eval", "brain_reconstructions")
os.makedirs(OUTPUT_DIR, exist_ok=True)
USE_VQ = True
PATCH_SIZE = (64, 64, 64)

def plot_comparison(original, reconstructed, output_path, title):
    """Plot original and reconstructed brain side by side"""
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    
    depth = original.shape[2]
    slice_indices = np.linspace(0, depth-1, 8, dtype=int)
    
    for i, slice_idx in enumerate(slice_indices):
        axes[0, i].imshow(original[:, :, slice_idx], cmap=cm, vmin=0, vmax=1)
        axes[0, i].set_title(f"Slice {slice_idx}")
        axes[0, i].axis('off')
    
    for i, slice_idx in enumerate(slice_indices):
        axes[1, i].imshow(reconstructed[:, :, slice_idx], cmap=cm, vmin=0, vmax=1)
        axes[1, i].axis('off')
    
    fig.text(0.01, 0.75, "Original", ha='left', va='center', fontsize=14, rotation=90)
    fig.text(0.01, 0.25, "Reconstructed", ha='left', va='center', fontsize=14, rotation=90)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0.02, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

def main():
    # Load the trained model
    print(f"Loading model from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    model = Improved3DAutoencoder(
        in_channels=1,
        latent_channels=8,
        base_channels=32,
        use_vq=USE_VQ
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    
    # --- THIS IS THE CORRECTED SECTION ---
    # Define the base path for the sample file
    sample_base_path = os.path.join(DATA_DIR, '100307', 'tfMRI_MOTOR_RL')
    sample_file = None
    
    # Check for both .nii and .nii.gz extensions
    for ext in ['.nii', '.nii.gz']:
        potential_path = sample_base_path + ext
        if os.path.exists(potential_path):
            sample_file = potential_path
            print(f"Found sample file: {sample_file}")
            break
    # --- END OF CORRECTION ---

    if sample_file is None:
        print(f"Sample file not found at '{sample_base_path}.nii' or '{sample_base_path}.nii.gz'.")
        print("Please verify your DATA_DIR path and that the subject '100307' exists.")
        return

    print(f"Processing file: {os.path.basename(sample_file)}")
    
    try:
        # Load fMRI data using nibabel
        nii_img = nib.load(sample_file)
        fmri_data = nii_img.get_fdata(dtype=np.float32)
        
        # Average over time if 4D
        brain_data = np.mean(fmri_data, axis=3) if fmri_data.ndim == 4 else fmri_data
        
        # Normalize to [0, 1]
        brain_data = (brain_data - brain_data.min()) / (brain_data.max() - brain_data.min())
        
        # Convert to a PyTorch tensor with batch and channel dimensions
        original_tensor = torch.from_numpy(brain_data).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # --- SLIDING WINDOW INFERENCE ---
        with torch.no_grad():
            # The predictor function must return only the reconstruction for sliding_window_inference
            def predictor(x):
                recon, _, _ = model(x)
                return recon

            reconstructed_tensor = sliding_window_inference(
                inputs=original_tensor,
                roi_size=PATCH_SIZE,
                sw_batch_size=4,
                predictor=predictor,
                overlap=0.5,
                mode="gaussian"
            )

        # Convert tensors back to numpy for plotting
        original_np = original_tensor.squeeze().cpu().numpy()
        reconstructed_np = reconstructed_tensor.squeeze().cpu().numpy()
        
        print(f"Original shape (DxHxW): {original_np.shape}")
        print(f"Reconstructed shape (DxHxW): {reconstructed_np.shape}")
        
        # Create unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{os.path.basename(sample_file).split('.')[0]}_{timestamp}"
        
        # Plot comparison
        print("Creating comparison view...")
        comparison_path = os.path.join(OUTPUT_DIR, f"comparison_{filename_base}.png")
        
        # Nibabel loads as (H, W, D), but our PyTorch pipeline expects (D, H, W).
        # We need to transpose back for plotting to match the original orientation.
        # Let's assume the original nibabel data is (H, W, D) and our tensor is (D, H, W)
        # So we transpose the numpy arrays for plotting
        original_for_plot = np.transpose(original_np, (1, 2, 0))
        reconstructed_for_plot = np.transpose(reconstructed_np, (1, 2, 0))

        plot_comparison(
            original_for_plot, 
            reconstructed_for_plot,
            comparison_path,
            title=f"Brain Comparison - {os.path.basename(sample_file)}"
        )
        print(f"Comparison saved to {comparison_path}")
            
    except Exception as e:
        print(f"Error processing file {sample_file}: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()