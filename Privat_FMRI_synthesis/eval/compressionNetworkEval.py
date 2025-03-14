import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# Szülő könyvtár hozzáadása az elérési úthoz
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)
from models.autoencoder import Improved3DAutoencoder
from utils.dataset import FMRIDataModule

# Eszköz definiálása
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modell betöltése
model = Improved3DAutoencoder().to(device)
checkpoint_path = r"/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_New/improved_autoencoder_best.pth"
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()  # Értékelési mód: dropout és batchnorm kikapcsolva

# DataModule inicializálása
TEST_CSV = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config/test.csv'
DATA_DIR = r'/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri'
BATCH_SIZE = 16
NUM_WORKERS = 20
PREFETCH_FACTOR = 4

data_module = FMRIDataModule(
    train_csv=TEST_CSV,  # Itt a test.csv-t használjuk, mint a complexEval.py-ban
    val_csv=TEST_CSV,
    test_csv=TEST_CSV,
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR
)
data_module.setup()
test_loader = data_module.test_dataloader()

def test_fc_layer(autoencoder, test_loader):
    """
    Teszteli az autoencoder FC rétegeit a flattened reprezentáció rekonstrukciós hibájának mérésével valós tesztadatokon.
    
    Args:
        autoencoder: A betöltött Improved3DAutoencoder modell
        test_loader: DataLoader a tesztadatokhoz
    """
    total_loss = 0.0
    num_batches = 0
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        for fmri_tensor, _ in tqdm(test_loader, desc="Evaluating FC layers on Test Data"):
            fmri_tensor = fmri_tensor.to(device)
            
            # Konvolúciós enkóderen áthaladás a flattened reprezentációhoz
            e1 = F.leaky_relu(autoencoder.enc_norm1(autoencoder.enc_conv1(fmri_tensor)), 0.2)
            e2 = F.leaky_relu(autoencoder.enc_norm2(autoencoder.enc_conv2(e1)), 0.2)
            e3 = F.leaky_relu(autoencoder.enc_norm3(autoencoder.enc_conv3(e2)), 0.2)
            flattened = e3.view(e3.size(0), -1)  # Pl. (batch_size, 128*12*14*12)
            
            # Enkóder FC rétegeken áthaladás
            encoded = F.leaky_relu(autoencoder.enc_fc1(flattened), 0.2)
            encoded = autoencoder.enc_dropout(encoded)  # Eval módban identitásfüggvény
            z = autoencoder.enc_fc2(encoded)  # Látens vektor: (batch_size, 512)
            
            # Dekóder FC rétegeken áthaladás
            d = F.leaky_relu(autoencoder.dec_fc1(z), 0.2)
            d = autoencoder.dec_dropout(d)  # Eval módban identitásfüggvény
            reconstructed_flattened = F.leaky_relu(autoencoder.dec_fc2(d), 0.2)  # (batch_size, 128*12*14*12)
            
            # Rekonstrukciós hiba kiszámítása
            loss = loss_fn(flattened, reconstructed_flattened)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"FC rétegek átlagos rekonstrukciós hibája (MSE): {avg_loss:.6f}")

# Teszt futtatása
test_fc_layer(model, test_loader)