import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt  # For plotting

# Get the parent directory of the current script (training/)
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add parent directory to sys.path
sys.path.append(PARENT_DIR)

# Import your FMRIDataModule (which reads CSVs and loads fMRI data)
from utils.dataset import FMRIDataModule  
# Import your pre-trained autoencoder architecture
from models.autoencoder import Improved3DAutoencoder  
# Import the diffusion model definition
from models.diffusion import LatentDiffusion

# Utility function: one-hot encoding
def one_hot_encode(labels, num_classes=5):
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot

# ----- Configuration -----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 5
LATENT_SHAPE = (8, 8, 8)  # As defined in your autoencoder
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 10

# Directories and CSV paths (update these paths as needed)
CHECKPOINT_DIR = r'/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_New'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DIFFUSION_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'latent_diffusion.pth')
AUTOENCODER_CHECKPOINT = r'/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_New/finetuned_autoencoder_best.pth'
TRAIN_CSV = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/train.csv'
VAL_CSV = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/val.csv'
TEST_CSV = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/test.csv'
DATA_DIR = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri'

# ----- Load Pre-trained Autoencoder -----
if not os.path.exists(AUTOENCODER_CHECKPOINT):
    raise FileNotFoundError(f'Autoencoder checkpoint not found: {AUTOENCODER_CHECKPOINT}')
autoencoder = Improved3DAutoencoder(latent_dims=LATENT_SHAPE, num_classes=NUM_CLASSES).to(DEVICE)
autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT, map_location=DEVICE))
autoencoder.eval()
print("Loaded pre-trained autoencoder.")

# ----- Initialize DataModule -----
data_module = FMRIDataModule(
    train_csv=TRAIN_CSV,
    val_csv=VAL_CSV,
    test_csv=TEST_CSV,  # test_csv not used here
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=20,
    prefetch_factor=4
)
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# ----- Initialize Diffusion Model -----
diffusion_model = LatentDiffusion(latent_shape=LATENT_SHAPE, num_classes=NUM_CLASSES, device=DEVICE)

# Attempt to load an existing diffusion checkpoint to resume training
if os.path.exists(DIFFUSION_CKPT_PATH):
    state = torch.load(DIFFUSION_CKPT_PATH, map_location=DEVICE)
    diffusion_model.model.load_state_dict(state)
    print(f"Loaded diffusion model checkpoint from {DIFFUSION_CKPT_PATH}")

diffusion_model.model.train()

# Loss criteria: We'll compute a composite loss in the latent space.
mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()
# Use torchmetrics' SSIM measure; assuming it can handle your 3D latent tensors.
from torchmetrics.image import StructuralSimilarityIndexMeasure
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# Weights for the composite loss
w_mse = 0.8
w_l1 = 0.1
w_ssim = 0.1

best_val_loss = float('inf')
stopping_counter = 0

# Prepare log file
log_file = os.path.join(CHECKPOINT_DIR, 'diffusion_training_log.txt')
with open(log_file, 'w') as f:
    f.write("Epoch,TrainLoss,ValLoss\n")

# Lists to track losses for plotting
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    train_loss = 0.0
    diffusion_model.model.train()
    
    for fmri_tensor, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
        labels_one_hot = one_hot_encode(labels, num_classes=NUM_CLASSES)
        
        # ----- Get latent codes from the pretrained encoder -----
        with torch.no_grad():
            # We only need the latent representation (skip decoding)
            _, latent_3d, _, _ = autoencoder.encode(fmri_tensor)
        latent_3d = latent_3d.unsqueeze(1)  # Shape: [B, 1, D, H, W]
        
        # ----- Forward diffusion: add noise to latent code -----
        noisy_latent, t, true_noise = diffusion_model.forward_diffusion(latent_3d)
        
        # Predict noise using the diffusion model in latent space
        pred_noise = diffusion_model.model(noisy_latent, t, labels_one_hot)
        
        # Compute losses in the latent space
        mse_loss = mse_criterion(pred_noise, true_noise)
        l1_loss = l1_criterion(pred_noise, true_noise)
        ssim_loss = 1 - ssim(pred_noise, true_noise)
        loss = w_mse * mse_loss + w_l1 * l1_loss + w_ssim * ssim_loss
        
        diffusion_model.optim.zero_grad()
        loss.backward()
        diffusion_model.optim.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1} Training Loss: {train_loss:.6f}")
    
    # ----- Validation in Latent Space -----
    diffusion_model.model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for fmri_tensor, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            fmri_tensor, labels = fmri_tensor.to(DEVICE), labels.to(DEVICE)
            labels_one_hot = one_hot_encode(labels, num_classes=NUM_CLASSES)
            
            # Get latent code from autoencoder
            _, latent_3d, _, _ = autoencoder.encode(fmri_tensor)
            latent_3d = latent_3d.unsqueeze(1)
            
            # Add noise and compute predicted noise in latent space
            noisy_latent, t, true_noise = diffusion_model.forward_diffusion(latent_3d)
            pred_noise = diffusion_model.model(noisy_latent, t, labels_one_hot)
            
            mse_loss = mse_criterion(pred_noise, true_noise)
            l1_loss = l1_criterion(pred_noise, true_noise)
            ssim_loss = 1 - ssim(pred_noise, true_noise)
            loss = w_mse * mse_loss + w_l1 * l1_loss + w_ssim * ssim_loss
            
            val_loss += loss.item()
            
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1} Validation Loss: {val_loss:.6f}")
    
    # Logging and checkpointing
    with open(log_file, 'a') as f:
        f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f}\n")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        stopping_counter = 0
        torch.save(diffusion_model.model.state_dict(), DIFFUSION_CKPT_PATH)
        print(f"Checkpoint saved at epoch {epoch+1}")
    else:
        stopping_counter += 1
        if stopping_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# ----- Plot losses using matplotlib -----
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Latent-space Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(CHECKPOINT_DIR, 'loss_plot.png'))
plt.show()
