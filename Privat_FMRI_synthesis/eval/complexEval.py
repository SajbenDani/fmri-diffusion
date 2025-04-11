import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ----- Add parent directory to path for relative imports -----
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

# ----- Custom modules -----
from models.autoencoder import fMRIAutoencoder
from models.diffusion import ConditionalDiffusion
from utils.dataset import FMRIDataModule
from config import *

# ----- Load Autoencoder -----
if not os.path.exists(AUTOENCODER_CHECKPOINT):
    raise FileNotFoundError(f"Autoencoder checkpoint not found at: {AUTOENCODER_CHECKPOINT}")
autoencoder = fMRIAutoencoder(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT, map_location=DEVICE))
autoencoder.eval()
print(" Loaded pre-trained autoencoder.")

# ----- Load Dataset -----
data_module = FMRIDataModule(
    train_csv=TEST_CSV,
    val_csv=TEST_CSV,
    test_csv=TEST_CSV,
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=0  # Keep 0 for CPU compatibility; can change to NUM_WORKERS if needed
)
data_module.setup()
test_loader = data_module.test_dataloader()

# ----- Load Diffusion Model -----
if not os.path.exists(DIFFUSION_CHECKPOINT):
    print(f" Diffusion checkpoint not found at {DIFFUSION_CHECKPOINT}. Exiting.")
    exit()

diffusion_model = ConditionalDiffusion(num_classes=NUM_CLASSES).to(DEVICE)
diffusion_model.load_state_dict(torch.load(DIFFUSION_CHECKPOINT, map_location=DEVICE))
diffusion_model.eval()
print(f" Loaded diffusion model from {DIFFUSION_CHECKPOINT}")

# ----- Sampling Scheduler -----
scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
scheduler.set_timesteps(20)  # Number of inference steps

# ----- Losses & Metrics -----
mse_loss_fn = nn.MSELoss()
l1_loss_fn = nn.L1Loss()
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# ----- Sampling Function -----
def sample_from_diffusion(model, labels, steps=20):
    with torch.no_grad():
        latent = torch.randn(labels.size(0), 1, SPATIAL_DIM, SPATIAL_DIM, device=DEVICE)
        for t in scheduler.timesteps:
            timestep = torch.tensor([t], device=DEVICE).expand(labels.size(0))
            noise_pred = model(latent, timestep, labels)
            latent = scheduler.step(noise_pred, t, latent).prev_sample
        return latent

# ----- Evaluation Loop -----
def evaluate():
    total_mse, total_l1, total_ssim, num_batches = 0, 0, 0, 0

    with torch.no_grad():
        for fmri_tensor, labels in tqdm(test_loader, desc="Evaluating"):
            fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)

            latent = sample_from_diffusion(diffusion_model, labels)
            latent_flat = latent.view(latent.size(0), -1)
            label_embed = autoencoder.label_embed(labels)
            recon = autoencoder.decoder(torch.cat([latent_flat, label_embed], dim=-1))

            total_mse += mse_loss_fn(recon, fmri_tensor).item()
            total_l1 += l1_loss_fn(recon, fmri_tensor).item()
            total_ssim += (1 - ssim_metric(recon, fmri_tensor)).item()
            num_batches += 1

    print(f"\n Evaluation Results:")
    print(f"  MSE  : {total_mse / num_batches:.6f}")
    print(f"  L1   : {total_l1 / num_batches:.6f}")
    print(f"  SSIM : {total_ssim / num_batches:.6f}")

# ----- Visualization -----
def visualize_reconstruction(output_path):
    with torch.no_grad():
        fmri_tensor, labels = next(iter(test_loader))
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)

        latent = sample_from_diffusion(diffusion_model, labels)
        latent_flat = latent.view(latent.size(0), -1)
        recon = autoencoder.decoder(torch.cat([latent_flat, autoencoder.label_embed(labels)], dim=-1))

        if recon.shape[2:] != fmri_tensor.shape[2:]:
            recon = F.interpolate(recon, size=fmri_tensor.shape[2:], mode='trilinear', align_corners=False)

        recon_grid = vutils.make_grid(recon.cpu(), normalize=True, scale_each=True)
        orig_grid = vutils.make_grid(fmri_tensor.cpu(), normalize=True, scale_each=True)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(orig_grid.permute(*PERMUTE_ORDER))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Reconstructed")
        plt.imshow(recon_grid.permute(*PERMUTE_ORDER))
        plt.axis("off")

        plt.savefig(output_path)
        plt.close()
        print(f"📸 Reconstruction saved at: {output_path}")

# ----- Main Entrypoint -----
if __name__ == '__main__':
    evaluate()
    visualize_reconstruction(os.path.join(CHECKPOINT_DIR, "test_reconstructions_2d.png"))
