import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Szülő könyvtár hozzáadása az elérési úthoz
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)
from models.autoencoder import Improved3DAutoencoder
from utils.dataset import FMRIDataModule

# Konfiguráció
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
NUM_EPOCHS = 50  # Nagyobb szám, hogy az early stopping működjön
LEARNING_RATE = 1e-4
PATIENCE = 5  # Early stopping türelmi idő
CHECKPOINT_DIR = r'/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_New'
AUTOENCODER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'improved_autoencoder_best.pth')
TRAIN_CSV = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/train.csv'
VAL_CSV = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/val.csv'
DATA_DIR = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri'

# Modell betöltése
model = Improved3DAutoencoder().to(DEVICE)
model.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT, map_location=DEVICE))

# Befagyasztjuk a konvolúciós rétegeket
for name, param in model.named_parameters():
    if 'enc_conv' in name or 'enc_norm' in name or 'dec_conv' in name or 'dec_norm' in name or 'label_embedding' in name:
        param.requires_grad = False

# Optimalizáló csak az FC rétegekhez
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# Loss funkciók definiálása
mse_loss_fn = nn.MSELoss()
l1_loss_fn = nn.L1Loss()
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# DataModule inicializálása
data_module = FMRIDataModule(
    train_csv=TRAIN_CSV,
    val_csv=VAL_CSV,
    test_csv=VAL_CSV,  # Nem használjuk itt, de meg kell adni
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=20,
    prefetch_factor=4
)
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Early stopping változók
best_val_loss = float('inf')
patience_counter = 0

# Tanítás
model.train()
for epoch in range(NUM_EPOCHS):
    total_train_loss = 0.0
    for fmri_tensor, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}"):
        fmri_tensor = fmri_tensor.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Encode: konvolúciós rétegek (befagyasztva)
        with torch.no_grad():
            e1 = nn.functional.leaky_relu(model.enc_norm1(model.enc_conv1(fmri_tensor)), 0.2)
            e2 = nn.functional.leaky_relu(model.enc_norm2(model.enc_conv2(e1)), 0.2)
            e3 = nn.functional.leaky_relu(model.enc_norm3(model.enc_conv3(e2)), 0.2)
            flattened = e3.view(e3.size(0), -1)
        
        # FC rétegeken áthaladás
        encoded = nn.functional.leaky_relu(model.enc_fc1(flattened), 0.2)
        encoded = model.enc_dropout(encoded)
        z = model.enc_fc2(encoded)
        
        # Dekóder FC rétegeken áthaladás
        d = nn.functional.leaky_relu(model.dec_fc1(z), 0.2)
        d = model.dec_dropout(d)
        reconstructed_flattened = nn.functional.leaky_relu(model.dec_fc2(d), 0.2)
        
        # Kompozit loss számítása
        mse_loss = mse_loss_fn(reconstructed_flattened, flattened)
        l1_loss = l1_loss_fn(reconstructed_flattened, flattened)
        # SSIM kihagyása, mert flattened vektoron nem alkalmazható közvetlenül
        loss = 0.7 * mse_loss + 0.3 * l1_loss  # Ideiglenesen egyszerűsített loss
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Training Loss: {avg_train_loss:.6f}")
    
    # Validáció
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for fmri_tensor, _ in tqdm(val_loader, desc="Validating"):
            fmri_tensor = fmri_tensor.to(DEVICE)
            
            # Encode
            e1 = nn.functional.leaky_relu(model.enc_norm1(model.enc_conv1(fmri_tensor)), 0.2)
            e2 = nn.functional.leaky_relu(model.enc_norm2(model.enc_conv2(e1)), 0.2)
            e3 = nn.functional.leaky_relu(model.enc_norm3(model.enc_conv3(e2)), 0.2)
            flattened = e3.view(e3.size(0), -1)
            
            # FC rétegeken áthaladás
            encoded = nn.functional.leaky_relu(model.enc_fc1(flattened), 0.2)
            encoded = model.enc_dropout(encoded)
            z = model.enc_fc2(encoded)
            
            # Dekóder FC rétegeken áthaladás
            d = nn.functional.leaky_relu(model.dec_fc1(z), 0.2)
            d = model.dec_dropout(d)
            reconstructed_flattened = nn.functional.leaky_relu(model.dec_fc2(d), 0.2)
            
            # Kompozit loss számítása
            mse_loss = mse_loss_fn(reconstructed_flattened, flattened)
            l1_loss = l1_loss_fn(reconstructed_flattened, flattened)
            loss = 0.7 * mse_loss + 0.3 * l1_loss
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Validation Loss: {avg_val_loss:.6f}")
    
    # Modell mentése és early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "Autoencoder_best_FC.pth"))
        print(f"Modell mentve: Autoencoder_best_FC.pth (Validation Loss: {avg_val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping: Validation loss nem javult {PATIENCE} epochon keresztül.")
            break

print("Tanítás befejezve.")