import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to the path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)
from models.autoencoder import Improved3DAutoencoder
from utils.dataset import FMRIDataModule
from config import *



# Load model
model = Improved3DAutoencoder().to(DEVICE)
state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()  # Evaluation mode: disables dropout and batchnorm

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

def test_fc_layer(autoencoder, test_loader):
    """
    Tests the FC layers of the autoencoder by measuring reconstruction error 
    of the flattened representation on real test data.

    Args:
        autoencoder: The loaded Improved3DAutoencoder model
        test_loader: DataLoader for the test dataset
    """
    total_loss = 0.0
    num_batches = 0
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        for fmri_tensor, _ in tqdm(test_loader, desc="Evaluating FC layers on Test Data"):
            fmri_tensor = fmri_tensor.to(DEVICE)
            
            # Pass through convolutional encoder to get flattened representation
            e1 = F.leaky_relu(autoencoder.enc_norm1(autoencoder.enc_conv1(fmri_tensor)), 0.2)
            e2 = F.leaky_relu(autoencoder.enc_norm2(autoencoder.enc_conv2(e1)), 0.2)
            e3 = F.leaky_relu(autoencoder.enc_norm3(autoencoder.enc_conv3(e2)), 0.2)
            flattened = e3.view(e3.size(0), -1)  # e.g. (batch_size, 128*12*14*12)
            
            # Pass through encoder FC layers
            encoded = F.leaky_relu(autoencoder.enc_fc1(flattened), 0.2)
            encoded = autoencoder.enc_dropout(encoded)  # Identity in eval mode
            z = autoencoder.enc_fc2(encoded)  # Latent vector: (batch_size, 512)
            
            # Pass through decoder FC layers
            d = F.leaky_relu(autoencoder.dec_fc1(z), 0.2)
            d = autoencoder.dec_dropout(d)  # Identity in eval mode
            reconstructed_flattened = F.leaky_relu(autoencoder.dec_fc2(d), 0.2)  # (batch_size, 128*12*14*12)
            
            # Compute reconstruction loss
            loss = loss_fn(flattened, reconstructed_flattened)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Average reconstruction error of FC layers (MSE): {avg_loss:.6f}")

# Run the test
test_fc_layer(model, test_loader)
