"""
fMRI Dataset Management and Data Loading Infrastructure.

This module provides comprehensive data loading and management functionality for
functional Magnetic Resonance Imaging (fMRI) datasets used in the diffusion model
project. It implements efficient, memory-aware data loading strategies specifically
designed for 3D medical imaging data.

Key Features:
    - Patch-based data loading for memory efficiency
    - Multi-view support (axial, coronal, sagittal orientations)
    - Robust error handling and fallback mechanisms  
    - Configurable data loading with performance optimizations
    - Support for different fMRI task labels (motor tasks)

Dataset Organization:
    The system expects pre-processed fMRI data organized as patches stored in
    PyTorch tensor format. This approach provides several advantages:
    - Reduced memory footprint compared to full volume loading
    - Faster I/O operations with pre-processed data
    - Flexible sampling strategies across different brain regions

Motor Task Labels:
    The dataset supports 5 different motor task conditions:
    - lh (0): Left hand movement
    - rh (1): Right hand movement  
    - lf (2): Left foot movement
    - rf (3): Right foot movement
    - t (4): Tongue movement

Data Pipeline:
    CSV Files -> Patch Loading -> View Permutation -> Batch Formation -> Model Training

Usage Example:
    ```python
    # Initialize data module
    data_module = FMRIDataModule(
        train_csv="train_patches.csv",
        val_csv="val_patches.csv", 
        test_csv="test_patches.csv",
        batch_size=8,
        view='axial'
    )
    
    # Setup datasets
    data_module.setup()
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Training loop
    for batch in train_loader:
        patches, labels = batch
        # Process batch...
    ```

Performance Considerations:
    - Uses pin_memory for faster GPU transfer
    - Configurable number of workers for parallel loading
    - Persistent workers to avoid recreation overhead
    - Prefetch factor for loading optimization
    - Error handling prevents training interruption
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import torch.serialization
from monai.data.meta_tensor import MetaTensor
from monai.utils.enums import MetaKeys
from typing import Optional, Tuple, Union, Callable

# Configure PyTorch serialization for MONAI tensors
torch.serialization.add_safe_globals([MetaTensor])

# Setup logging for dataset operations
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CORE DATASET CLASS
# =============================================================================

class FMRIPatchDataset(Dataset):
    """
    PyTorch Dataset for Loading Pre-processed fMRI Patches.
    
    This dataset class handles loading of pre-processed fMRI patches stored as
    PyTorch tensors. It provides flexible view orientation support and robust
    error handling to ensure stable training even with occasional corrupted files.
    
    The patch-based approach is crucial for memory efficiency when working with
    high-resolution 3D fMRI data. Instead of loading entire brain volumes
    (which can be 200+ MB each), the system works with smaller patches that
    maintain spatial coherence while fitting comfortably in GPU memory.
    
    Key Features:
        - Multi-view orientation support for different analysis perspectives
        - Robust error handling with fallback mechanisms
        - Memory-efficient patch-based loading
        - Consistent label mapping across different datasets
        - Transform support for data augmentation
    
    Label Mapping:
        The dataset uses a standardized mapping for motor task labels:
        - "lh" -> 0: Left hand movement
        - "rh" -> 1: Right hand movement
        - "lf" -> 2: Left foot movement  
        - "rf" -> 3: Right foot movement
        - "t"  -> 4: Tongue movement
        
        This mapping ensures consistent numerical representation across
        different dataset splits and experimental conditions.
    
    View Orientations:
        - 'axial': Horizontal slices (default medical view)
        - 'coronal': Vertical front-to-back slices
        - 'sagittal': Vertical left-to-right slices
        
        Different views can reveal different patterns in brain activation
        and may be useful for different analysis purposes.
    
    Args:
        csv_path (str): Path to CSV file containing patch information
            Expected columns: 'patch_path', 'label'
        view (str): Anatomical view orientation ('axial', 'coronal', 'sagittal')
        transform (callable, optional): Optional transform to apply to patches
        
    CSV Format:
        The CSV file should contain at minimum:
        - patch_path: Full path to the .pt patch file
        - label: Motor task label (string or integer)
        
    Patch File Format:
        Each patch file should be a PyTorch tensor saved with torch.save()
        Expected shape: (channels, depth, height, width)
        Typical patch size: (1, 64, 64, 64) for 64³ voxel patches
    """
    
    # Class-level label mapping for consistent encoding across instances
    LABEL_MAP = {
        "lh": 0,   # Left hand motor task
        "rh": 1,   # Right hand motor task  
        "lf": 2,   # Left foot motor task
        "rf": 3,   # Right foot motor task
        "t": 4,    # Tongue motor task
    }
    
    def __init__(self, csv_path: str, view: str = 'axial', transform: Optional[Callable] = None):
        """
        Initialize the fMRI patch dataset.
        
        Args:
            csv_path: Path to CSV file with patch metadata
            view: Anatomical view orientation for patch interpretation
            transform: Optional callable for data augmentation/preprocessing
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If view orientation is not supported
        """
        # Validate inputs
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        valid_views = ['axial', 'coronal', 'sagittal']
        if view not in valid_views:
            raise ValueError(f"View must be one of {valid_views}, got: {view}")
        
        # Load dataset metadata
        self.data_info = pd.read_csv(csv_path)
        self.view = view
        self.transform = transform
        
        # Validate required columns
        required_columns = ['patch_path', 'label']
        missing_columns = [col for col in required_columns if col not in self.data_info.columns]
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {missing_columns}")
        
        # Check patch file availability and log statistics
        self._validate_patch_files()
        
        logger.info(f"Initialized FMRIPatchDataset with {len(self.data_info)} patches")
        logger.info(f"View orientation: {view}")
        logger.info(f"Transform: {'Yes' if transform else 'None'}")
    
    def _validate_patch_files(self) -> None:
        """
        Validate that patch files exist and log dataset statistics.
        
        This method performs a preliminary check of patch file availability
        to identify potential data loading issues early. It logs warnings
        for missing files but doesn't remove them from the dataset to
        maintain consistent indexing.
        """
        missing_files = 0
        total_size_mb = 0
        
        for idx, row in self.data_info.iterrows():
            patch_path = row['patch_path']
            if not os.path.exists(patch_path):
                missing_files += 1
                if missing_files <= 5:  # Log first few missing files
                    logger.warning(f"Missing patch file: {patch_path}")
            else:
                # Estimate file size for memory planning
                try:
                    file_size_mb = os.path.getsize(patch_path) / (1024 * 1024)
                    total_size_mb += file_size_mb
                except OSError:
                    pass  # Skip size calculation if file access fails
        
        if missing_files > 0:
            logger.warning(f"Found {missing_files} missing patch files out of {len(self.data_info)}")
            if missing_files > 5:
                logger.warning("... (showing first 5 missing files)")
        
        if total_size_mb > 0:
            logger.info(f"Estimated dataset size: {total_size_mb:.1f} MB")
    
    def __len__(self) -> int:
        """Return the total number of patches in the dataset."""
        return len(self.data_info)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single patch with its label.
        
        This method handles the core data loading functionality, including:
        - Loading pre-processed patch tensors from disk
        - Applying view-specific permutations for anatomical orientation
        - Converting labels to numerical format
        - Applying optional transforms
        - Providing fallback mechanisms for corrupted files
        
        Args:
            idx (int): Index of the patch to load
            
        Returns:
            tuple: (patch_tensor, label_tensor)
                - patch_tensor: 4D tensor (C, D, H, W) with patch data
                - label_tensor: Scalar tensor with encoded label
                
        Error Handling:
            If a patch file is corrupted or missing, the method returns:
            - Zero tensor with expected patch dimensions
            - Label of -1 to indicate invalid sample
            
            This prevents training interruption while flagging problematic data.
            
        View Permutations:
            Different anatomical views require different axis arrangements:
            - axial: (C, H, D, W) - horizontal slices
            - coronal: (C, W, H, D) - front-to-back slices
            - sagittal: original (C, D, H, W) - left-to-right slices
        """
        # Get patch metadata
        row = self.data_info.iloc[idx]
        patch_path = row['patch_path']
        label_str = row['label']
        
        # Convert label to numerical format
        if isinstance(label_str, str):
            # String label: use mapping dictionary
            label = self.LABEL_MAP.get(label_str, -1)
            if label == -1:
                logger.warning(f"Unknown label '{label_str}' at index {idx}, using -1")
        else:
            # Numerical label: convert to integer
            try:
                label = int(label_str)
            except (ValueError, TypeError):
                logger.warning(f"Invalid label '{label_str}' at index {idx}, using -1")
                label = -1
        
        # Load pre-processed patch tensor
        try:
            # Load with weights_only=False to support MONAI tensors
            fmri_patch = torch.load(patch_path, weights_only=False)
            
            # Validate tensor shape and properties
            if not isinstance(fmri_patch, torch.Tensor):
                raise ValueError(f"Loaded object is not a tensor: {type(fmri_patch)}")
                
            if fmri_patch.dim() != 4:
                raise ValueError(f"Expected 4D tensor, got {fmri_patch.dim()}D")
                
        except Exception as e:
            # Handle loading errors gracefully
            logger.error(f"Error loading patch at {patch_path}: {e}")
            
            # Return dummy tensor with standard patch dimensions
            # This maintains batch consistency while flagging the error
            dummy_patch = torch.zeros((1, 64, 64, 64), dtype=torch.float32)
            error_label = torch.tensor(-1, dtype=torch.long)
            return dummy_patch, error_label
        
        # Apply view-specific permutations for anatomical orientation
        if self.view == 'axial':
            # Axial view: horizontal slices through the brain
            fmri_patch = fmri_patch.permute(0, 2, 1, 3)  # (C, D, H, W) -> (C, H, D, W)
        elif self.view == 'coronal':
            # Coronal view: vertical slices from front to back
            fmri_patch = fmri_patch.permute(0, 3, 2, 1)  # (C, D, H, W) -> (C, W, H, D)
        # sagittal view uses original orientation (C, D, H, W) - no permutation needed
        
        # Apply optional data transforms (augmentation, normalization, etc.)
        if self.transform is not None:
            try:
                fmri_patch = self.transform(fmri_patch)
            except Exception as e:
                logger.warning(f"Transform failed for patch at {patch_path}: {e}")
                # Continue with original patch if transform fails
        
        # Return patch and label as tensors
        return fmri_patch, torch.tensor(label, dtype=torch.long)


# =============================================================================
# DATA MODULE FOR MULTI-SPLIT MANAGEMENT
# =============================================================================

class FMRIDataModule:
    """
    Data Module for Managing Train/Validation/Test Splits of fMRI Patch Data.
    
    This class provides a high-level interface for managing the complete data
    pipeline, from dataset initialization to data loader creation. It follows
    the PyTorch Lightning DataModule pattern while remaining framework-agnostic.
    
    Key Features:
        - Centralized management of train/val/test splits
        - Optimized data loader configuration for performance
        - Memory-efficient loading strategies
        - Consistent batch sizing across splits
        - GPU-optimized data transfer settings
    
    Performance Optimizations:
        - pin_memory: Enables faster GPU transfer by using page-locked memory
        - persistent_workers: Avoids worker process recreation between epochs
        - prefetch_factor: Controls how many batches are loaded ahead of time
        - Variable worker counts: Uses fewer workers for validation/test (less I/O intensive)
        - drop_last: Ensures consistent batch sizes during training
    
    Usage Pattern:
        ```python
        # Initialize with dataset paths
        data_module = FMRIDataModule(
            train_csv="train_patches.csv",
            val_csv="val_patches.csv",
            test_csv="test_patches.csv"
        )
        
        # Setup datasets (loads metadata, validates files)
        data_module.setup()
        
        # Get optimized data loaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        ```
    
    Args:
        train_csv (str): Path to training patches CSV file
        val_csv (str): Path to validation patches CSV file  
        test_csv (str): Path to test patches CSV file
        batch_size (int): Batch size for all data loaders (default: 8)
        num_workers (int): Number of worker processes for data loading (default: 8)
        prefetch_factor (int): Batches to prefetch per worker (default: 2)
        view (str): Anatomical view orientation (default: 'axial')
    """
    
    def __init__(self, train_csv: str, val_csv: str, test_csv: str,
                 batch_size: int = 8, num_workers: int = 8, 
                 prefetch_factor: int = 2, view: str = 'axial'):
        """
        Initialize the fMRI data module with dataset configurations.
        
        Args:
            train_csv: Path to training dataset CSV
            val_csv: Path to validation dataset CSV
            test_csv: Path to test dataset CSV
            batch_size: Batch size for training (affects memory usage)
            num_workers: Parallel workers for data loading (affects CPU usage)
            prefetch_factor: Batches loaded ahead per worker (affects memory/latency)
            view: Anatomical view orientation for all datasets
        """
        # Store dataset paths
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        
        # Store data loading configuration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.view = view
        
        # Dataset objects (initialized in setup())
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        logger.info(f"Initialized FMRIDataModule:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  View: {view}")
        
    def setup(self) -> None:
        """
        Setup all datasets by loading metadata and validating files.
        
        This method initializes the actual dataset objects for each split.
        It should be called before accessing any data loaders. The method
        performs validation and logs dataset statistics for monitoring.
        
        Raises:
            FileNotFoundError: If any CSV files are missing
            ValueError: If datasets have inconsistent configurations
        """
        logger.info("Setting up fMRI patch datasets...")
        
        # Initialize dataset objects for each split
        self.train_dataset = FMRIPatchDataset(self.train_csv, self.view)
        self.val_dataset = FMRIPatchDataset(self.val_csv, self.view)
        self.test_dataset = FMRIPatchDataset(self.test_csv, self.view)
        
        # Log dataset statistics
        logger.info(f"Dataset setup complete:")
        logger.info(f"  Training patches: {len(self.train_dataset):,}")
        logger.info(f"  Validation patches: {len(self.val_dataset):,}")
        logger.info(f"  Test patches: {len(self.test_dataset):,}")
        
        # Calculate total dataset size for memory planning
        total_patches = len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)
        logger.info(f"  Total patches: {total_patches:,}")

    def train_dataloader(self) -> DataLoader:
        """
        Create optimized data loader for training data.
        
        Training data loader is configured for maximum throughput and
        includes optimizations like shuffling, drop_last for consistent
        batches, and full worker utilization.
        
        Returns:
            DataLoader: Configured training data loader
            
        Configuration:
            - shuffle=True: Randomize sample order each epoch
            - drop_last=True: Ensure consistent batch sizes
            - Full worker count for maximum I/O parallelism
            - pin_memory for faster GPU transfer
            - persistent_workers to avoid recreation overhead
        """
        if self.train_dataset is None:
            raise RuntimeError("Must call setup() before accessing data loaders")
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Randomize order for better training
            num_workers=self.num_workers,
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True  # Consistent batch sizes
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create optimized data loader for validation data.
        
        Validation data loader is configured for deterministic evaluation
        with reduced resource usage compared to training loader.
        
        Returns:
            DataLoader: Configured validation data loader
            
        Configuration:
            - shuffle=False: Deterministic evaluation order
            - Reduced worker count (validation is less I/O intensive)
            - pin_memory for GPU transfer efficiency
            - No drop_last (want to evaluate all samples)
        """
        if self.val_dataset is None:
            raise RuntimeError("Must call setup() before accessing data loaders")
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Deterministic evaluation
            num_workers=self.num_workers // 2,  # Reduced workers for validation
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
            # No drop_last - want to evaluate all validation samples
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create optimized data loader for test data.
        
        Test data loader is configured similarly to validation but for
        final evaluation on held-out test data.
        
        Returns:
            DataLoader: Configured test data loader
            
        Configuration:
            - shuffle=False: Deterministic test evaluation
            - Reduced worker count for efficiency
            - pin_memory for GPU compatibility
            - No drop_last to ensure all test samples are evaluated
        """
        if self.test_dataset is None:
            raise RuntimeError("Must call setup() before accessing data loaders")
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Deterministic test evaluation
            num_workers=self.num_workers // 2,  # Reduced workers for testing
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
            # No drop_last - want to evaluate all test samples
        )