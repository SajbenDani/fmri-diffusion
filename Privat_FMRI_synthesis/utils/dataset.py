import os
import torch
import nibabel as nib
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class FMRI_Dataset(Dataset):
    LABEL_MAP = {
    "lh": 0,   # Left hemisphere
    "rh": 1,    # Right hemisphere
    "lf": 2,   # Left foot
    "rf": 3,    # Right foot
    "t": 4,    # Tongue
    }

    def __init__(self, csv_path, data_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the dataset CSV file (train.csv, val.csv, test.csv)
            data_dir (str): Path to the root data directory containing the 'mri/' folder
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.data_info = pd.read_csv(csv_path, delimiter=';')  # Assuming ; seperated
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    
    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        file_id = str(row['FILE_ID'])  # Convert int64 to string
        label = row['LABEL']
        
        # Convert label using the mapping
        if isinstance(label, str):
            if label in self.LABEL_MAP:
                label = self.LABEL_MAP[label]
            else:
                raise ValueError(f"Unknown label at index {idx}: {label}")  # Debugging
        else:
            label = int(label)  # If it's already a number, convert to int

        start_time, end_time = row['START_TIME'], row['END_TIME']
        
        # Load the fMRI file
        fmri_path = os.path.join(self.data_dir, 'mri', file_id, 'tfMRI_MOTOR_RL.nii.gz')
        fmri_img = nib.load(fmri_path).get_fdata()
        
        # Extract the relevant frames based on START_TIME and END_TIME
        fmri_slices = fmri_img[:, :, :, int(start_time):int(end_time)]
        fmri_tensor = torch.tensor(fmri_slices, dtype=torch.float32)

        # **Apply Min-Max Normalization**
        fmri_tensor = (fmri_tensor - fmri_tensor.min()) / (fmri_tensor.max() - fmri_tensor.min())
        
        # Apply transformations if provided
        if self.transform:
            fmri_tensor = self.transform(fmri_tensor)

        # this below is an alternative if we do not want to average the time dimension
        # Add a channel dimension and permute dimensions correctly
        # fmri_tensor = fmri_tensor.unsqueeze(0)  # Adds channel dim: (1, D, H, W, T)
        # fmri_tensor = fmri_tensor.permute(0, 4, 1, 2, 3)  # Reorder to (1, T, D, H, W)
        # Ensure the tensor has the correct shape [batch_size * time, channels, depth, height, width]
        # fmri_tensor = fmri_tensor.reshape(-1, 1, *fmri_tensor.shape[2:])  # Reshape to (T, 1, D, H, W)

        # You can average over the time dimension:
        fmri_tensor = fmri_tensor.mean(dim=-1, keepdim=True)  # (D, H, W, 1)
        fmri_tensor = fmri_tensor.permute(3, 0, 1, 2)  # (1, D, H, W)

        return fmri_tensor, torch.tensor(label, dtype=torch.long)
    
# Example usage
def get_dataloader(csv_file, data_dir, batch_size=8, shuffle=True):
    dataset = FMRI_Dataset(csv_file, data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
