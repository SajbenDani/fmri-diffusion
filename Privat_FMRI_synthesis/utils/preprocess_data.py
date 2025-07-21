import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from monai.transforms import RandCropByPosNegLabeld

# --- Configuration ---
PARENT_DIR = Path(__file__).parent.parent
DATA_DIR = "/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri"
CONFIG_DIR = "/home/jovyan/work/ssd0/USERS/siposlevente/data/new_format_config"
OUTPUT_DIR = PARENT_DIR / "data_preprocessed" # New directory for patches

PATCH_SIZE = (64, 64, 64)
PATCHES_PER_VOLUME = 10  # Extract 10 random patches from each source volume
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Use all but one CPU core

def process_row(args):
    """Process a single row from the dataframe - for parallel processing"""
    row, split_output_dir, patch_size = args
    try:
        file_id = str(row['FILE_ID'])
        label = row['LABEL']
        start_time, end_time = int(row['START_TIME']), int(row['END_TIME'])
        
        # Load file and check if it exists
        base_path = os.path.join(DATA_DIR, file_id, 'tfMRI_MOTOR_RL')
        fmri_path = None
        for ext in ['.nii.gz', '.nii']:
            potential_path = base_path + ext
            if os.path.exists(potential_path):
                fmri_path = potential_path
                break
        
        if fmri_path is None:
            return None
            
        # Load and preprocess the data
        fmri_data = nib.load(fmri_path).get_fdata(dtype=np.float32)
        
        # Extract time slices and average
        fmri_3d_volume = np.mean(fmri_data[..., start_time:end_time], axis=-1, keepdims=True)
        fmri_4d_tensor = torch.from_numpy(fmri_3d_volume).permute(3, 0, 1, 2)
        
        # Min-max normalization
        min_val, max_val = fmri_4d_tensor.min(), fmri_4d_tensor.max()
        if max_val > min_val:
            fmri_4d_tensor = (fmri_4d_tensor - min_val) / (max_val - min_val)
        
        # Generate multiple patches
        cropper = RandCropByPosNegLabeld(
            keys=["image"], label_key="image", spatial_size=patch_size,
            pos=1.0, neg=0.0, num_samples=PATCHES_PER_VOLUME, allow_smaller=True
        )
        
        data_dict = {"image": fmri_4d_tensor}
        cropped_list = cropper(data_dict)
        
        records = []
        for i, patch_dict in enumerate(cropped_list):
            patch_tensor = patch_dict["image"]
            
            # Define a unique filename for the patch
            patch_filename = f"{file_id}_{label}_{start_time}_{end_time}_{i}.pt"
            patch_save_path = os.path.join(split_output_dir, patch_filename)
            
            # Save the small patch tensor
            torch.save(patch_tensor, patch_save_path)
            
            # Add a record to our new CSV
            records.append({"patch_path": patch_save_path, "label": label})
        
        return records
        
    except Exception as e:
        print(f"Error processing {row['FILE_ID']}: {e}")
        return None

def preprocess_and_save():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        print(f"--- Processing {split} split ---")
        
        input_csv_path = os.path.join(CONFIG_DIR, f"{split}.csv")
        df = pd.read_csv(input_csv_path, delimiter=';')
        
        # Create output directories
        split_output_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Prepare arguments for parallel processing
        args_list = [(row, split_output_dir, PATCH_SIZE) for _, row in df.iterrows()]
        
        # Process rows in parallel
        all_records = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for records in tqdm(executor.map(process_row, args_list), total=len(args_list)):
                if records:
                    all_records.extend(records)
        
        # Save the new CSV for this split
        new_df = pd.DataFrame(all_records)
        new_df.to_csv(os.path.join(OUTPUT_DIR, f"{split}_patches.csv"), index=False)
        print(f"Finished processing {split} split. Saved {len(new_df)} patches.")

if __name__ == "__main__":
    print(f"Starting preprocessing with {NUM_WORKERS} workers")
    preprocess_and_save()
    print("Preprocessing complete!")