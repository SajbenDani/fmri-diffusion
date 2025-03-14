import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt

# Szülő könyvtár hozzáadása az elérési úthoz
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.append(PARENT_DIR)
from models.autoencoder import Improved3DAutoencoder
from utils.dataset import FMRIDataModule
from config import *

# Konstansok
NUM_CLASSES = 5
CHECKPOINT_PATH = r'/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_New/finetuned_autoencoder_best.pth'
OUTPUT_DIR = r'/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/test_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# One-hot encoding függvény
def one_hot_encode(labels, num_classes=NUM_CLASSES):
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

# Modell betöltése
autoencoder = Improved3DAutoencoder(latent_dims=(8, 8, 8), num_classes=NUM_CLASSES).to(DEVICE)
if os.path.exists(CHECKPOINT_PATH):
    autoencoder.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print(f"✅ Loaded checkpoint from {CHECKPOINT_PATH}")
else:
    raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
autoencoder.eval()

# Loss funkciók
mse_criterion = nn.MSELoss(reduction='mean')
l1_criterion = nn.L1Loss(reduction='mean')
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# DataModule inicializálása
data_module = FMRIDataModule(
    train_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/train.csv',
    val_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/val.csv',
    test_csv=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/test.csv',
    data_dir=r'/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri',
    batch_size=1,  # Egyenkénti feldolgozás a pontos logolás érdekében
    num_workers=4
)
data_module.setup()
test_loader = data_module.test_dataloader()

# Tracking változók
metrics = {
    'fmri_min': [], 'fmri_max': [], 'fmri_mean': [],
    'recon_min': [], 'recon_max': [], 'recon_mean': [],
    'z_mean': [], 'z_std': [],
    'label_emb_mean': [], 'label_emb_std': [],
    'modulated_z_mean': [], 'modulated_z_std': [],
    'pixel_diff_var': [],
    'mse_loss': [], 'l1_loss': [], 'ssim_loss': []
}

print(f"Evaluating model on test set with {len(test_loader)} samples...")

# Teszt ciklus
with torch.no_grad():
    for batch_idx, (fmri_tensor, labels) in enumerate(tqdm(test_loader, desc="Evaluating Test Set")):
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
        labels_one_hot = one_hot_encode(labels, num_classes=NUM_CLASSES)

        # Forward pass
        recon, latent_3d = autoencoder(fmri_tensor, labels_one_hot)
        z, latent_3d, e1, e2 = autoencoder.encode(fmri_tensor)
        label_emb = autoencoder.label_embedding(labels_one_hot)
        modulated_z = z * torch.sigmoid(label_emb)

        # Metrikák számítása
        mse_loss = mse_criterion(recon, fmri_tensor).item()
        l1_loss = l1_criterion(recon, fmri_tensor).item()
        ssim_loss = 1 - ssim(recon, fmri_tensor).item()

        # Tárolás
        metrics['fmri_min'].append(fmri_tensor.min().item())
        metrics['fmri_max'].append(fmri_tensor.max().item())
        metrics['fmri_mean'].append(fmri_tensor.mean().item())
        metrics['recon_min'].append(recon.min().item())
        metrics['recon_max'].append(recon.max().item())
        metrics['recon_mean'].append(recon.mean().item())
        metrics['z_mean'].append(z.mean().item())
        metrics['z_std'].append(z.std().item())
        metrics['label_emb_mean'].append(label_emb.mean().item())
        metrics['label_emb_std'].append(label_emb.std().item())
        metrics['modulated_z_mean'].append(modulated_z.mean().item())
        metrics['modulated_z_std'].append(modulated_z.std().item())
        metrics['pixel_diff_var'].append((recon - fmri_tensor).var().item())
        metrics['mse_loss'].append(mse_loss)
        metrics['l1_loss'].append(l1_loss)
        metrics['ssim_loss'].append(ssim_loss)

        # Opcionális: mentsük el az első néhány rekonstrukciót vizualizációra
        if batch_idx < 3:
            torch.save(recon.cpu(), os.path.join(OUTPUT_DIR, f'recon_sample_{batch_idx}.pt'))
            torch.save(fmri_tensor.cpu(), os.path.join(OUTPUT_DIR, f'fmri_sample_{batch_idx}.pt'))

# Összesített eredmények számítása
def print_stats(name, values):
    mean = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    print(f"{name}: mean={mean:.6f}, std={std:.6f}, min={min_val:.6f}, max={max_val:.6f}")

print("\n--- Test Set Evaluation Results ---")
print_stats("fmri_tensor min", metrics['fmri_min'])
print_stats("fmri_tensor max", metrics['fmri_max'])
print_stats("fmri_tensor mean", metrics['fmri_mean'])
print_stats("recon min", metrics['recon_min'])
print_stats("recon max", metrics['recon_max'])
print_stats("recon mean", metrics['recon_mean'])
print_stats("z mean", metrics['z_mean'])
print_stats("z std", metrics['z_std'])
print_stats("label_embedding mean", metrics['label_emb_mean'])
print_stats("label_embedding std", metrics['label_emb_std'])
print_stats("modulated_z mean", metrics['modulated_z_mean'])
print_stats("modulated_z std", metrics['modulated_z_std'])
print_stats("pixel diff variance", metrics['pixel_diff_var'])
print_stats("MSE loss", metrics['mse_loss'])
print_stats("L1 loss", metrics['l1_loss'])
print_stats("SSIM loss", metrics['ssim_loss'])

# Latens tér histogram készítése (z és modulated_z)
plt.figure(figsize=(12, 6))
plt.hist(np.array(metrics['z_mean']), bins=20, alpha=0.5, label='z mean')
plt.hist(np.array(metrics['modulated_z_mean']), bins=20, alpha=0.5, label='modulated_z mean')
plt.xlabel('Mean Value')
plt.ylabel('Frequency')
plt.title('Distribution of z and modulated_z Means')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'latent_distribution.png'))
plt.close()

print(f"\nResults and sample reconstructions saved to {OUTPUT_DIR}")