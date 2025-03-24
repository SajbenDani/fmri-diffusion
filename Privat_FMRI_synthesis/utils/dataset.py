import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import nibabel as nib
import gc
import weakref
import psutil
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FMRI_Dataset(Dataset):
    LABEL_MAP = {
        "lh": 0,   # Left hemisphere
        "rh": 1,    # Right hemisphere
        "lf": 2,   # Left foot
        "rf": 3,    # Right foot
        "t": 4,    # Tongue
    }
    
    # Class-level cache to share between instances
    _file_cache = weakref.WeakValueDictionary()
    
    def __init__(self, csv_path, data_dir, transform=None, cache_size=50, memory_limit_gb=None):
        """
        Args:
            csv_path (str): Path to the dataset CSV file (train.csv, val.csv, test.csv)
            data_dir (str): Path to the root data directory containing the 'mri/' folder
            transform (callable, optional): Optional transform to be applied on a sample.
            cache_size (int): Maximum number of processed samples to cache in memory
            memory_limit_gb (float): Maximum memory usage in GB before cache clearing
        """
        self.data_dir = data_dir
        self.data_info = pd.read_csv(csv_path, delimiter=';')
        self.transform = transform
        
        # Sample cache (processed tensors)
        self.cache = {}
        self.cache_size = cache_size
        self.cache_keys_queue = []  # For LRU implementation
        
        # Memory management
        self.memory_limit_gb = memory_limit_gb
        
        # Analyze file sharing
        self._analyze_file_sharing()
        
    def _analyze_file_sharing(self):
        """Analyze how many samples share the same source files to optimize caching"""
        file_counts = self.data_info['FILE_ID'].value_counts()
        self.files_by_frequency = file_counts.index.tolist()
        logger.info(f"Dataset has {len(file_counts)} unique files across {len(self.data_info)} samples")
        logger.info(f"Most frequent file appears {file_counts.max()} times")

    def _check_memory_usage(self):
        """Check if memory usage exceeds limit and clear cache if needed"""
        if self.memory_limit_gb is not None:
            process = psutil.Process(os.getpid())
            memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
            
            if memory_gb > self.memory_limit_gb:
                logger.warning(f"Memory usage ({memory_gb:.2f} GB) exceeds limit ({self.memory_limit_gb} GB). Clearing cache.")
                self.cache.clear()
                self.cache_keys_queue = []
                gc.collect()
                return True
        return False

    def _add_to_cache(self, idx, item):
        """Add an item to the cache with LRU eviction policy"""
        # Check and manage memory first
        self._check_memory_usage()
        
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.cache_size:
            oldest_key = self.cache_keys_queue.pop(0)
            del self.cache[oldest_key]
            
        # Add new item to cache
        self.cache[idx] = item
        self.cache_keys_queue.append(idx)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # Check if item is in cache
        if idx in self.cache:
            # Move to most recently used
            self.cache_keys_queue.remove(idx)
            self.cache_keys_queue.append(idx)
            return self.cache[idx]
            
        row = self.data_info.iloc[idx]
        file_id = str(row['FILE_ID'])
        label = row['LABEL']
        
        # Convert label using the mapping
        if isinstance(label, str):
            if label in self.LABEL_MAP:
                label = self.LABEL_MAP[label]
            else:
                raise ValueError(f"Unknown label at index {idx}: {label}")
        else:
            label = int(label)
            
        start_time, end_time = row['START_TIME'], row['END_TIME']
        
        # Try to get file from class cache first
        fmri_path = os.path.join(self.data_dir, file_id, 'tfMRI_MOTOR_RL.nii.gz')
        cache_key = f"{file_id}"
        
        # Load the fMRI file (with memory mapping)
        if cache_key in self._file_cache:
            fmri_img = self._file_cache[cache_key]
        else:
            # Use memory mapping to reduce memory usage
            fmri_img = nib.load(fmri_path, mmap=True).get_fdata()
            self._file_cache[cache_key] = fmri_img
        
        # Extract relevant frames (this creates a view, not a copy when possible)
        fmri_slices = fmri_img[:, :, :, int(start_time):int(end_time)]
        
        # Convert to tensor
        fmri_tensor = torch.tensor(fmri_slices, dtype=torch.float32)
        
        # Min-Max Normalization
        min_val = fmri_tensor.min()
        max_val = fmri_tensor.max()
        fmri_tensor = (fmri_tensor - min_val) / (max_val - min_val)
        
        # Apply transformations if provided
        if self.transform:
            fmri_tensor = self.transform(fmri_tensor)
            
        # Average over time dimension
        fmri_tensor = fmri_tensor.mean(dim=-1, keepdim=True)  # (D, H, W, 1)
        fmri_tensor = fmri_tensor.permute(3, 0, 1, 2)  # (1, D, H, W)
        
        # Create the final item
        item = (fmri_tensor, torch.tensor(label, dtype=torch.long))
        
        # Add to cache
        self._add_to_cache(idx, item)
        
        return item


class FMRIDataModule(LightningDataModule):
    #num workers set to zero because of cpu setup.
    def __init__(self, train_csv, val_csv, test_csv, data_dir, batch_size=8, num_workers=0, 
                 cache_size=50, memory_limit_gb=60, prefetch_factor=2):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.memory_limit_gb = memory_limit_gb
        self.prefetch_factor = prefetch_factor
        
    def setup(self, stage=None):
        logger.info("Setting up datasets - this may take a moment...")
        
        # Setup datasets with memory management parameters
        self.train_dataset = FMRI_Dataset(
            self.train_csv, 
            self.data_dir, 
            cache_size=self.cache_size,
            memory_limit_gb=self.memory_limit_gb
        )
        
        self.val_dataset = FMRI_Dataset(
            self.val_csv, 
            self.data_dir,
            cache_size=int(self.cache_size/2),  # Use smaller cache for validation
            memory_limit_gb=self.memory_limit_gb
        )
        
        self.test_dataset = FMRI_Dataset(
            self.test_csv, 
            self.data_dir,
            cache_size=int(self.cache_size/2),  # Use smaller cache for test
            memory_limit_gb=self.memory_limit_gb
        )
        
        # Memory status after setup
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"Memory usage after dataset setup: {memory_info.rss / (1024*1024*1024):.2f} GB")
        
        # Force garbage collection
        gc.collect()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            drop_last=True  # Slightly more efficient
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=max(2, self.num_workers // 2),  # Use fewer workers for validation
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=max(2, self.num_workers // 2),  # Use fewer workers for test
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor
        )