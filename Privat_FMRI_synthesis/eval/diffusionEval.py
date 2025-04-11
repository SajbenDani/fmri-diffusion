import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt  # Optional: for plotting loss curves

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
    train_csv=TEST_CSV,  # Using test.csv for evaluation
    val_csv=TEST_CSV,
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
# Load the diffusion checkpoint if available
if os.path.exists(DIFFUSION_CHECKPOINT):
    state = torch.load(DIFFUSION_CHECKPOINT, map_location=DEVICE)
    diffusion_model.model.load_state_dict(state)
    print(f"Loaded diffusion model checkpoint from {DIFFUSION_CHECKPOINT}")
else:
    print("No diffusion checkpoint found. Exiting evaluation.")
    exit()

diffusion_model.model.eval()

# Loss criteria: latent-space composite loss
mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)



total_loss = 0.0
total_mse = 0.0
total_l1 = 0.0
total_ssim = 0.0
num_batches = 0

# Evaluate the diffusion model on the test dataset in latent space
with torch.no_grad():
    for fmri_tensor, labels in tqdm(test_loader, desc="Evaluating in Latent Space"):
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
        labels_one_hot = one_hot_encode(labels, num_classes=NUM_CLASSES)
        
        # Get latent representation from autoencoder encoder
        # We only use the latent output here (without decoding)
        _, latent_3d, _, _ = autoencoder.encode(fmri_tensor)
        latent_3d = latent_3d.unsqueeze(1)  # Shape: [B, 1, D, H, W]
        
        # Forward diffusion: add noise to the latent code
        noisy_latent, t, true_noise = diffusion_model.forward_diffusion(latent_3d)
        
        # Predict noise using the diffusion model in latent space
        pred_noise = diffusion_model.model(noisy_latent, t, labels_one_hot)
        
        # Compute latent-space losses
        mse_loss = mse_criterion(pred_noise, true_noise)
        l1_loss = l1_criterion(pred_noise, true_noise)
        ssim_loss = 1 - ssim_metric(pred_noise, true_noise)
        
        composite_loss = W_MSE * mse_loss + W_L1 * l1_loss + W_SSIM * ssim_loss
        
        total_loss += composite_loss.item()
        total_mse += mse_loss.item()
        total_l1 += l1_loss.item()
        total_ssim += ssim_loss.item()
        num_batches += 1

avg_loss = total_loss / num_batches
avg_mse = total_mse / num_batches
avg_l1 = total_l1 / num_batches
avg_ssim = total_ssim / num_batches

print(f"Test Composite Loss: {avg_loss:.6f}")
print(f"Test MSE Loss: {avg_mse:.6f}")
print(f"Test L1 Loss: {avg_l1:.6f}")
print(f"Test SSIM Loss: {avg_ssim:.6f}")

# Optionally, plot the loss distribution (e.g., as a bar chart)
plt.figure(figsize=(6,4))
plt.bar(['Composite', 'MSE', 'L1', 'SSIM'], [avg_loss, avg_mse, avg_l1, avg_ssim], color=['blue','green','red','purple'])
plt.title('Average Latent-Space Losses on Test Data')
plt.ylabel('Loss')
plt.savefig(os.path.join(CHECKPOINT_DIR, 'test_latent_losses.png'))
plt.show()
