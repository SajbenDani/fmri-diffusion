import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as vutils
from diffusers import DDPMScheduler
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# Add parent directory to the system path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

# Import models and dataset module
from models.autoencoder import fMRIAutoencoder
from models.diffusion import ConditionalDiffusion
from utils.dataset import FMRIDataModule
from config import *

# ----- Configuration -----
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
NUM_CLASSES = 5
LATENT_DIM = 256
SPATIAL_DIM = int(LATENT_DIM ** 0.5)  # 16
BATCH_SIZE = 16
PERMUTE_ORDER = (2, 0, 1)  # For visualization: [D, H, W] -> [W, D, H]

# Directories and CSV paths
BASE_LOG_DIR = '/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/logs'
CHECKPOINT_DIR = '/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_New'
DIFFUSION_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'diffusion_model.pth')
AUTOENCODER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'autoencoder.pth')
TEST_CSV = '/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/test.csv'
DATA_DIR = '/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri'

# Create timestamped log directory
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = os.path.join(BASE_LOG_DIR, f"eval_2d_{timestamp}")
os.makedirs(LOG_DIR, exist_ok=True)
print(f"Logging results to {LOG_DIR}")

# ----- Load Pre-trained Autoencoder -----
if not os.path.exists(AUTOENCODER_CHECKPOINT):
    raise FileNotFoundError(f'Autoencoder checkpoint not found: {AUTOENCODER_CHECKPOINT}')
try:
    autoencoder = fMRIAutoencoder(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT, map_location=DEVICE, weights_only=False))
    autoencoder.eval()
    print("Loaded pre-trained autoencoder.")
except Exception as e:
    print(f"Failed to load autoencoder checkpoint: {e}")
    exit()

# ----- Initialize DataModule for Test Data -----
data_module = FMRIDataModule(
    train_csv=TEST_CSV,
    val_csv=TEST_CSV,
    test_csv=TEST_CSV,
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=16
)
data_module.setup()
test_loader = data_module.test_dataloader()

# ----- Initialize Diffusion Model -----
diffusion_model = ConditionalDiffusion(num_classes=NUM_CLASSES).to(DEVICE)
if os.path.exists(DIFFUSION_CKPT_PATH):
    diffusion_model.load_state_dict(torch.load(DIFFUSION_CKPT_PATH, map_location=DEVICE, weights_only=False))
    print(f"Loaded diffusion model checkpoint from {DIFFUSION_CKPT_PATH}")
else:
    print("No diffusion checkpoint found. Exiting evaluation.")
    exit()
diffusion_model.eval()

# Initialize scheduler for diffusion sampling
scheduler = DDPMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(20)

# Loss metrics
mse_criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

# Sampling function
def sample_diffusion(model, labels, steps=20):
    with torch.no_grad():
        x = torch.randn(labels.size(0), 1, SPATIAL_DIM, SPATIAL_DIM, device=DEVICE)
        for t in scheduler.timesteps:
            timestep = torch.tensor([t], device=DEVICE).expand(labels.size(0))
            noise_pred = model(x, timestep, labels)
            x = scheduler.step(noise_pred, t, x).prev_sample
        return x

# Evaluation loop
batch_metrics = {
    'recon_mse': [], 'recon_mae': [], 'recon_ssim': [], 'recon_psnr': [],
    'orig_latent_mean': [], 'orig_latent_std': [], 'gen_latent_mean': [], 'gen_latent_std': []
}

with torch.no_grad():
    for fmri_tensor, labels in tqdm(test_loader, desc="Evaluating on Test Data"):
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)

        # Original latent from encoder
        orig_latent = autoencoder.encoder(fmri_tensor)  # [B, 256]

        # Generated latent from diffusion
        gen_latent = sample_diffusion(diffusion_model, labels)  # [B, 1, 16, 16]
        gen_latent_flat = gen_latent.view(gen_latent.size(0), -1)  # [B, 256]

        # Decode with label conditioning
        recon = autoencoder.decoder(
            torch.cat([gen_latent_flat, autoencoder.label_embed(labels)], dim=-1)
        )

        target_size = fmri_tensor.shape[2:]  # [91, 109, 91]
        if recon.shape[2:] != target_size:
            recon = F.interpolate(recon, size=target_size, mode='trilinear', align_corners=False)

        # Reconstruction metrics
        mse_loss = mse_criterion(recon, fmri_tensor)
        mae_loss = mae_criterion(recon, fmri_tensor)
        ssim_value = ssim_metric(recon, fmri_tensor)
        psnr_value = psnr_metric(recon, fmri_tensor)

        # Latent statistics
        orig_mean = orig_latent.mean(dim=1)  # [B]
        orig_std = orig_latent.std(dim=1)    # [B]
        gen_mean = gen_latent_flat.mean(dim=1)  # [B]
        gen_std = gen_latent_flat.std(dim=1)    # [B]

        # Store per-batch metrics
        batch_metrics['recon_mse'].append(mse_loss.item())
        batch_metrics['recon_mae'].append(mae_loss.item())
        batch_metrics['recon_ssim'].append(ssim_value.item())
        batch_metrics['recon_psnr'].append(psnr_value.item())
        batch_metrics['orig_latent_mean'].extend(orig_mean.cpu().tolist())
        batch_metrics['orig_latent_std'].extend(orig_std.cpu().tolist())
        batch_metrics['gen_latent_mean'].extend(gen_mean.cpu().tolist())
        batch_metrics['gen_latent_std'].extend(gen_std.cpu().tolist())

# Compute statistics
stats = {}
for key, values in batch_metrics.items():
    values = np.array(values)
    stats[key] = {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values)
    }

# Print detailed results
print("--- Test Set Evaluation Results ---")
print(f"Reconstruction MSE: mean={stats['recon_mse']['mean']:.6f}, std={stats['recon_mse']['std']:.6f}, min={stats['recon_mse']['min']:.6f}, max={stats['recon_mse']['max']:.6f}")
print(f"Reconstruction MAE: mean={stats['recon_mae']['mean']:.6f}, std={stats['recon_mae']['std']:.6f}, min={stats['recon_mae']['min']:.6f}, max={stats['recon_mae']['max']:.6f}")
print(f"Reconstruction SSIM: mean={stats['recon_ssim']['mean']:.6f}, std={stats['recon_ssim']['std']:.6f}, min={stats['recon_ssim']['min']:.6f}, max={stats['recon_ssim']['max']:.6f}")
print(f"Reconstruction PSNR: mean={stats['recon_psnr']['mean']:.6f}, std={stats['recon_psnr']['std']:.6f}, min={stats['recon_psnr']['min']:.6f}, max={stats['recon_psnr']['max']:.6f}")
print(f"Original Latent Mean: mean={stats['orig_latent_mean']['mean']:.6f}, std={stats['orig_latent_std']['std']:.6f}, min={stats['orig_latent_mean']['min']:.6f}, max={stats['orig_latent_mean']['max']:.6f}")
print(f"Original Latent Std: mean={stats['orig_latent_std']['mean']:.6f}, std={stats['orig_latent_std']['std']:.6f}, min={stats['orig_latent_std']['min']:.6f}, max={stats['orig_latent_std']['max']:.6f}")
print(f"Generated Latent Mean: mean={stats['gen_latent_mean']['mean']:.6f}, std={stats['gen_latent_mean']['std']:.6f}, min={stats['gen_latent_mean']['min']:.6f}, max={stats['gen_latent_mean']['max']:.6f}")
print(f"Generated Latent Std: mean={stats['gen_latent_std']['mean']:.6f}, std={stats['gen_latent_std']['std']:.6f}, min={stats['gen_latent_std']['min']:.6f}, max={stats['gen_latent_std']['max']:.6f}")

# Visualization with (2, 0, 1) permutation
with torch.no_grad():
    fmri_tensor, labels = next(iter(test_loader))
    fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
    latent = sample_diffusion(diffusion_model, labels)
    latent_flat = latent.view(latent.size(0), -1)
    recon = autoencoder.decoder(
        torch.cat([latent_flat, autoencoder.label_embed(labels)], dim=-1)
    )

    target_size = fmri_tensor.shape[2:]  # [91, 109, 91]
    if recon.shape[2:] != target_size:
        recon = F.interpolate(recon, size=target_size, mode='trilinear', align_corners=False)

    # Apply (2, 0, 1) permutation
    fmri_tensor_perm = fmri_tensor.permute(0, 1, 4, 2, 3)  # [B, 1, 91, 91, 109]
    recon_perm = recon.permute(0, 1, 4, 2, 3)              # [B, 1, 91, 91, 109]

    mid_slice_idx = target_size[2] // 2  # 91 // 2 = 45 (after permutation)
    fmri_slice = fmri_tensor_perm[:, :, mid_slice_idx, :, :]  # [B, 1, 91, 109]
    recon_slice = recon_perm[:, :, mid_slice_idx, :, :]       # [B, 1, 91, 109]

    recon_grid = vutils.make_grid(recon_slice.cpu(), normalize=True, scale_each=True)
    orig_grid = vutils.make_grid(fmri_slice.cpu(), normalize=True, scale_each=True)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Images (Middle Slice, Permuted)")
    plt.imshow(orig_grid.permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Images (Middle Slice, Permuted)")
    plt.imshow(recon_grid.permute(1, 2, 0))
    plt.axis("off")

    vis_path = os.path.join(LOG_DIR, "test_reconstructions_2d.png")
    plt.savefig(vis_path)
    plt.close()
    print(f"Visualization saved to {vis_path}")

# Plot meaningful metrics (MSE, SSIM, PSNR)
metrics_to_plot = ['recon_mse', 'recon_ssim', 'recon_psnr']
for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    plt.plot(batch_metrics[metric], label=f'{metric.split("_")[1].upper()} per Batch')
    plt.axhline(stats[metric]['mean'], color='r', linestyle='--', label=f'Avg {metric.split("_")[1].upper()}: {stats[metric]["mean"]:.4f}')
    plt.xlabel('Batch Index')
    plt.ylabel(metric.split('_')[1].upper())
    plt.title(f'{metric.split("_")[1].upper()} Over Test Batches')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(LOG_DIR, f"{metric.split('_')[1]}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"{metric.split('_')[1].upper()} plot saved to {plot_path}")