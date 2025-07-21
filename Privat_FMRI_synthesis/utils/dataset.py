import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import torch.serialization
from monai.data.meta_tensor import MetaTensor
from monai.utils.enums import MetaKeys

torch.serialization.add_safe_globals([MetaTensor])
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FMRIPatchDataset(Dataset):
    LABEL_MAP = {"lh": 0, "rh": 1, "lf": 2, "rf": 3, "t": 4}
    
    def __init__(self, csv_path, view='axial', transform=None):
        """
        Args:
            csv_path (str): Path to the preprocessed CSV file with patch information
            view (str): View orientation ('axial', 'coronal', or 'sagittal')
            transform (callable, optional): Optional transform to apply to the patches
        """
        self.data_info = pd.read_csv(csv_path)
        self.view = view
        self.transform = transform
        
        # Check if files exist
        missing = 0
        for i, row in enumerate(self.data_info.iterrows()):
            if not os.path.exists(row[1]['patch_path']):
                missing += 1
        
        if missing > 0:
            logger.warning(f"Found {missing} missing patch files out of {len(self.data_info)}")
    
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        patch_path = row['patch_path']
        label_str = row['label']
        
        # Handle different label formats
        if isinstance(label_str, str):
            label = self.LABEL_MAP.get(label_str, -1)
        else:
            label = int(label_str)
            
        # Load the pre-processed patch - using weights_only=False
        try:
            fmri_patch = torch.load(patch_path, weights_only=False)
        except Exception as e:
            logger.error(f"Error loading patch at {patch_path}: {e}")
            # Return a dummy patch and flag it as invalid
            return torch.zeros((1, 64, 64, 64)), torch.tensor(-1, dtype=torch.long)
        
        # Apply view permutation
        if self.view == 'axial':
            fmri_patch = fmri_patch.permute(0, 2, 1, 3)  # (C, H, D, W)
        elif self.view == 'coronal':
            fmri_patch = fmri_patch.permute(0, 3, 2, 1)  # (C, W, H, D)
        
        # Apply additional transforms if provided
        if self.transform:
            fmri_patch = self.transform(fmri_patch)
            
        return fmri_patch, torch.tensor(label, dtype=torch.long)


class FMRIDataModule:
    """Data module for the FMRI dataset using pre-processed patches"""
    def __init__(self, train_csv, val_csv, test_csv, 
                 batch_size=8, num_workers=8, prefetch_factor=2, view='axial'):
        """
        Args:
            train_csv: Path to the training patches CSV
            val_csv: Path to the validation patches CSV
            test_csv: Path to the test patches CSV
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            prefetch_factor: Number of batches to prefetch per worker
            view: View orientation ('axial', 'coronal', or 'sagittal')
        """
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.view = view
        
    def setup(self):
        """Setup the datasets"""
        logger.info("Setting up patch datasets...")
        self.train_dataset = FMRIPatchDataset(self.train_csv, self.view)
        self.val_dataset = FMRIPatchDataset(self.val_csv, self.view)
        self.test_dataset = FMRIPatchDataset(self.test_csv, self.view)
        
        logger.info(f"Train dataset: {len(self.train_dataset)} patches")
        logger.info(f"Validation dataset: {len(self.val_dataset)} patches")
        logger.info(f"Test dataset: {len(self.test_dataset)} patches")

    def train_dataloader(self):
        """Return the training dataloader"""
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True
        )

    def val_dataloader(self):
        """Return the validation dataloader"""
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers // 2,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
        )

    def test_dataloader(self):
        """Return the test dataloader"""
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers // 2,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
        )