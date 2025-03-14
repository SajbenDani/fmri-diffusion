import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import nibabel as nib
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FMRI_Dataset(Dataset):
    LABEL_MAP = {
        "lh": 0,   # Left hemisphere
        "rh": 1,   # Right hemisphere
        "lf": 2,   # Left foot
        "rf": 3,   # Right foot
        "t": 4,    # Tongue
    }
    
    def __init__(self, csv_path, data_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the dataset CSV file (train.csv, val.csv, test.csv)
            data_dir (str): Root directory containing data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.data_info = pd.read_csv(csv_path, delimiter=';')
        self.transform = transform
        
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        file_id = str(row['FILE_ID'])
        label = row['LABEL']
        
        # Convert label using mapping
        if isinstance(label, str):
            if label in self.LABEL_MAP:
                label = self.LABEL_MAP[label]
            else:
                raise ValueError(f"Unknown label at index {idx}: {label}")
        else:
            label = int(label)
            
        start_time, end_time = row['START_TIME'], row['END_TIME']
        
        # Construct the file path
        fmri_path = os.path.join(self.data_dir, file_id, 'tfMRI_MOTOR_RL.nii.gz')
        
        # Load the fMRI file using memory mapping to reduce memory footprint
        fmri_img = nib.load(fmri_path, mmap=True).get_fdata()
        
        # Extract the relevant time frames
        fmri_slices = fmri_img[:, :, :, int(start_time):int(end_time)]
        
        
        # Convert to tensor and apply min-max normalization
        fmri_tensor = torch.tensor(fmri_slices, dtype=torch.float32)
        min_val = fmri_tensor.min()
        max_val = fmri_tensor.max()
        if min_val != max_val:
            fmri_tensor = (fmri_tensor - min_val) / (max_val - min_val)
        else:
            fmri_tensor = torch.zeros_like(fmri_tensor)
        
        # Apply any additional transformations if provided
        if self.transform:
            fmri_tensor = self.transform(fmri_tensor)
            
        # Average over the time dimension and adjust dimensions
        fmri_tensor = fmri_tensor.mean(dim=-1, keepdim=True)  # (D, H, W, 1)
        fmri_tensor = fmri_tensor.permute(3, 0, 1, 2)           # (1, D, H, W)
        
        return fmri_tensor, torch.tensor(label, dtype=torch.long)

class FMRIDataModule(LightningDataModule):
    def __init__(self, train_csv, val_csv, test_csv, data_dir,
                 batch_size=8, num_workers=16, prefetch_factor=4):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
    def setup(self, stage=None):
        logger.info("Setting up datasets...")
        self.train_dataset = FMRI_Dataset(self.train_csv, self.data_dir)
        self.val_dataset   = FMRI_Dataset(self.val_csv, self.data_dir)
        self.test_dataset  = FMRI_Dataset(self.test_csv, self.data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=max(2, self.num_workers // 2),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=max(2, self.num_workers // 2),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor
        )
