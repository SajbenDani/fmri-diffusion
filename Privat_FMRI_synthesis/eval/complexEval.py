import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt  # Optional, if you want to plot results
import torch.nn.functional as F
import torchvision.utils as vutils

# Get the parent directory of the current script (evaluation/)
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

# Import your FMRIDataModule (which reads CSVs and loads fMRI data)
from utils.dataset import FMRIDataModule  
# Import your pre-trained autoencoder architecture
from models.autoencoder import Improved3DAutoencoder  
# Import the diffusion model definition
from models.diffusion import LatentDiffusion
from torchmetrics.image import StructuralSimilarityIndexMeasure
from config import *

# Utility function: one-hot encoding
def one_hot_encode(labels, num_classes=5):
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot

# ----- Load Pre-trained Autoencoder -----
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f'Autoencoder checkpoint not found: {BEST_MODEL_PATH}')
autoencoder = Improved3DAutoencoder(latent_dims=LATENT_SHAPE, num_classes=NUM_CLASSES).to(DEVICE)
autoencoder.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
autoencoder.eval()
print("Loaded pre-trained autoencoder.")

# ----- Initialize DataModule for Test Data -----
data_module = FMRIDataModule(
    train_csv=TRAIN_CSV, 
    val_csv=VAL_CSV,
    test_csv=TEST_CSV,
    data_dir=DATA_DIR,
    batch_size=1,  # Batch size of 1 for evaluation
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR
)
data_module.setup()
test_loader = data_module.test_dataloader()

# ----- Initialize Diffusion Model -----
diffusion_model = LatentDiffusion(latent_shape=LATENT_SHAPE, num_classes=NUM_CLASSES, device=DEVICE)
# Load the diffusion checkpoint (if it exists) so we evaluate the best model
if os.path.exists(DIFFUSION_CHECKPOINT):
    state = torch.load(DIFFUSION_CHECKPOINT, map_location=DEVICE)
    diffusion_model.model.load_state_dict(state)
    print(f"Loaded diffusion model checkpoint from {DIFFUSION_CHECKPOINT}")
else:
    print("No diffusion checkpoint found. Exiting evaluation.")
    exit()

diffusion_model.model.eval()

# Loss metrics (in image space)
mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

total_mse = 0.0
total_l1 = 0.0
total_ssim = 0.0
num_batches = 0

# Evaluate the diffusion model on the test dataset
with torch.no_grad():
    for fmri_tensor, labels in tqdm(test_loader, desc="Evaluating on Test Data"):
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
        labels_one_hot = one_hot_encode(labels, num_classes=NUM_CLASSES)
        
        # Get latent representation from the pretrained autoencoder
        _, latent_3d, e1, e2 = autoencoder.encode(fmri_tensor)
        latent_3d = latent_3d.unsqueeze(1)  # Shape: [B, 1, D, H, W]
        
        # Use diffusion model sampling to generate latent codes
        sampled_latent = diffusion_model.sample(labels_one_hot, steps=20)
        # Flatten latent if required by decoder; our autoencoder.decode expects a flattened latent:
        z_flat = sampled_latent.view(sampled_latent.size(0), -1)
        # Decode latent code to obtain the reconstructed image
        recon = autoencoder.decode(z_flat, e1, e2)
        
        # Compute losses in image space
        mse_loss = mse_criterion(recon, fmri_tensor)
        l1_loss = l1_criterion(recon, fmri_tensor)
        ssim_loss = 1 - ssim_metric(recon, fmri_tensor)
        
        total_mse += mse_loss.item()
        total_l1 += l1_loss.item()
        total_ssim += ssim_loss.item()
        num_batches += 1

avg_mse = total_mse / num_batches
avg_l1 = total_l1 / num_batches
avg_ssim = total_ssim / num_batches

print(f"Test MSE Loss: {avg_mse:.6f}")
print(f"Test L1 Loss: {avg_l1:.6f}")
print(f"Test SSIM Loss: {avg_ssim:.6f}")

# Optionally, you can plot a few examples
import torchvision.utils as vutils

# Visualize first batch reconstructions vs originals (if desired)
with torch.no_grad():
    fmri_tensor, labels = next(iter(test_loader))
    fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
    labels_one_hot = one_hot_encode(labels, num_classes=NUM_CLASSES)
    _, latent_3d, e1, e2 = autoencoder.encode(fmri_tensor)
    latent_3d = latent_3d.unsqueeze(1)
    sampled_latent = diffusion_model.sample(labels_one_hot, steps=20)
    z_flat = sampled_latent.view(sampled_latent.size(0), -1)
    recon = autoencoder.decode(z_flat, e1, e2)
    
    # Ensure the reconstructed images have the same spatial dimensions as the original
    target_size = fmri_tensor.shape[2:]  # e.g. (D, H, W) of the original image
    if recon.shape[2:] != target_size:
        recon = F.interpolate(recon, size=target_size, mode='trilinear', align_corners=False)

    recon_grid = vutils.make_grid(recon.cpu(), normalize=True, scale_each=True)
    orig_grid = vutils.make_grid(fmri_tensor.cpu(), normalize=True, scale_each=True)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Images")
    plt.imshow(orig_grid.permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Images")
    plt.imshow(recon_grid.permute(1, 2, 0))
    plt.axis("off")

    plt.savefig(os.path.join(CHECKPOINT_DIR, "test_reconstructions.png"))
    plt.show()
