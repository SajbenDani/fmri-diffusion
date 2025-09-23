"""
Main entry point for the fMRI Diffusion Model Project.

This module serves as the central orchestrator for the fMRI-based Latent Diffusion Model
(LDM) framework. It provides a simple interface to switch between different operational
modes of the system, including training and inference.

The project implements a two-stage deep learning approach for fMRI super-resolution:
1. Autoencoder Training: Learn a compact latent representation of fMRI brain volumes
2. Diffusion Training: Train a diffusion model in the learned latent space
3. Inference: Generate high-quality synthetic fMRI data using the trained models

Supported Operations:
    - train_autoencoder: Train the 3D autoencoder for fMRI latent space learning
    - train_diffusion: Train the diffusion model for generation in latent space
    - predict: Generate synthetic fMRI volumes using trained models

Usage:
    1. Set the MODE variable to desired operation
    2. Run: python main.py
    
    Alternatively, run modules directly:
    - python -m training.train_autoencoder
    - python -m training.train_diffusion  
    - python -m models.predict

Architecture Notes:
    The system processes 3D fMRI volumes (typically 91x109x91 voxels) representing
    brain activation patterns for different motor tasks (left/right hand, left/right foot, tongue).
    The autoencoder compresses these to a latent space for efficient diffusion modeling.

Requirements:
    - CUDA-capable GPU recommended for training
    - Sufficient RAM for 3D volume processing (>16GB recommended)
    - Pre-processed fMRI datasets in NIfTI format (.nii.gz)
"""

import subprocess
import sys
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Operation mode selector - modify this to change the system behavior
# Valid options: 'train_autoencoder', 'train_diffusion', 'predict'
MODE = "predict"

# =============================================================================
# EXECUTION LOGIC
# =============================================================================

def main():
    """
    Main execution function that routes to the appropriate training/inference module.
    
    This function acts as a dispatcher, launching the correct subprocess based on
    the selected MODE. Each mode corresponds to a different stage of the pipeline:
    
    - train_autoencoder: Stage 1 - Learn latent representations of fMRI data
    - train_diffusion: Stage 2 - Train diffusion model in latent space  
    - predict: Inference - Generate synthetic fMRI volumes
    
    Returns:
        int: Exit code from the subprocess (0 for success, non-zero for error)
        
    Raises:
        SystemExit: If an invalid mode is selected or subprocess fails
    """
    print(f"🧠 fMRI Diffusion Model - Starting in '{MODE}' mode")
    print("=" * 60)
    
    try:
        if MODE == "train_diffusion":
            print("📚 Starting diffusion model training...")
            print("This will train the diffusion model in the autoencoder's latent space.")
            result = subprocess.run(["python", "-m", "training.train_diffusion"], check=True)
            
        elif MODE == "train_autoencoder":
            print("🏗️  Starting autoencoder training...")
            print("This will learn a compact latent representation of fMRI volumes.")
            result = subprocess.run(["python", "-m", "training.train_autoencoder"], check=True)
            
        elif MODE == "predict":
            print("🎯 Starting inference/prediction...")
            print("This will generate synthetic fMRI volumes using trained models.")
            result = subprocess.run(["python", "-m", "models.predict"], check=True)
            
        else:
            print(f"❌ Invalid mode selected: '{MODE}'")
            print("Valid options: 'train_diffusion', 'train_autoencoder', 'predict'")
            print("\nTo change mode, edit the MODE variable in main.py")
            sys.exit(1)
            
        print(f"✅ '{MODE}' completed successfully!")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running '{MODE}': {e}")
        print(f"Exit code: {e.returncode}")
        sys.exit(e.returncode)
        
    except KeyboardInterrupt:
        print(f"\n⚠️  '{MODE}' interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"❌ Unexpected error in '{MODE}': {e}")
        sys.exit(1)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
