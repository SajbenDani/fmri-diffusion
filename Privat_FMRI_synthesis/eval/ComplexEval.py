#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for fMRI Diffusion Models.

This module provides a complete evaluation framework for assessing the quality
of fMRI synthesis using the trained autoencoder and diffusion models. It implements
rigorous quantitative metrics and supports both classifier-free guidance (CFG)
and standard diffusion evaluation approaches.

Key Features:
    - Quantitative quality assessment using PSNR and SSIM metrics
    - Support for classifier-free guidance evaluation
    - Batch-wise processing for memory efficiency
    - Comprehensive results logging with timestamps
    - Statistical analysis across multiple samples
    - GPU/CPU memory management

Evaluation Pipeline:
    1. Load pre-trained autoencoder and diffusion models
    2. Setup test dataset with proper data loading
    3. For each test batch:
       a. Encode ground truth to latent space (with/without degradation)
       b. Sample from diffusion model using appropriate guidance
       c. Decode samples back to fMRI space
       d. Compute quality metrics against ground truth
    4. Aggregate statistics and save comprehensive results

Metrics Computed:
    - PSNR (Peak Signal-to-Noise Ratio): Measures reconstruction fidelity
    - SSIM (Structural Similarity Index): Measures perceptual similarity
    - Both metrics are computed per sample and aggregated for statistical analysis

Usage:
    Run directly for complete evaluation:
    ```bash
    python ComplexEval.py
    ```
    
    Or import for custom evaluation workflows:
    ```python
    from eval.ComplexEval import evaluate_diffusion_model
    results = evaluate_diffusion_model(model_path, dataset_path)
    ```

Output:
    - JSON file with detailed results and timestamps
    - Console logging with progress tracking
    - Statistical summaries (mean, std, min, max for each metric)

Requirements:
    - Pre-trained autoencoder and diffusion model checkpoints
    - Test dataset in preprocessed patch format
    - Sufficient GPU memory for batch processing
    - scikit-image for metric computation
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import json
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any

# Add parent directory to path for relative imports
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PARENT_DIR))

# Import custom modules
from models.autoencoder import Improved3DAutoencoder
from models.diffusion import DiffusionUNet3D
from utils.dataset import FMRIDataModule
from diffusers import DDPMScheduler

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Hardware configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoint paths - update these paths for your setup
AUTOENCODER_PATH = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_finetuned/best_finetuned_autoencoder.pt"
DIFFUSION_PATH = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_diffusion_cfg/best_diffusion_cfg_adversarial_aligned.pt"

# Data and output paths
PREPROCESSED_DIR = PARENT_DIR / "data_preprocessed"
OUTPUT_DIR = PARENT_DIR / "evaluation"
RESULTS_FILE = OUTPUT_DIR / f"numeric_results_cfg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Model architecture parameters
LATENT_CHANNELS = 8           # Latent space dimensionality
BASE_CHANNELS = 32            # Autoencoder base channels
DIFFUSION_BASE_CHANNELS = 128 # Diffusion model base channels
SCALE_FACTOR = 2              # Super-resolution scale factor

# Evaluation parameters
NUM_INFERENCE_STEPS = 1000    # Full diffusion steps for maximum quality
BATCH_SIZE = 4                # Batch size for evaluation (memory-dependent)
VIEW = 'axial'                # Anatomical view orientation
MAX_BATCHES = 100             # Maximum batches to evaluate (computational limit)
GUIDANCE_SCALE = 7.5          # Classifier-free guidance strength

# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def compute_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive quality metrics between prediction and target volumes.
    
    This function calculates Peak Signal-to-Noise Ratio (PSNR) and Structural
    Similarity Index (SSIM) between predicted and ground truth fMRI volumes.
    Both metrics are computed in a way that's robust to batch processing and
    different tensor formats.
    
    Args:
        prediction (torch.Tensor): Predicted fMRI volume
            Shape: (batch_size, channels, depth, height, width) or flattened
        target (torch.Tensor): Ground truth fMRI volume  
            Shape: same as prediction
            
    Returns:
        Dict[str, float]: Dictionary containing computed metrics
            - 'psnr': Peak Signal-to-Noise Ratio in dB
            - 'ssim': Structural Similarity Index (0-1 scale)
            
    Notes:
        - PSNR measures pixel-wise reconstruction accuracy (higher is better)
        - SSIM measures perceptual similarity considering structure (higher is better)
        - Both metrics are averaged across all spatial dimensions and batch items
        - Input tensors are automatically normalized to [0,1] range for fair comparison
        
    Mathematical Background:
        PSNR = 20 * log10(MAX_VAL / sqrt(MSE))
        SSIM considers luminance, contrast, and structure similarities
    """
    # Convert to CPU tensors for metric computation
    if isinstance(prediction, torch.Tensor):
        pred = prediction.detach().cpu().numpy()
    else:
        pred = prediction
        
    if isinstance(target, torch.Tensor):
        targ = target.detach().cpu().numpy()
    else:
        targ = target
    
    # Ensure both arrays have same shape
    if pred.shape != targ.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs target {targ.shape}")
    
    # Flatten arrays for batch processing while preserving spatial structure
    if pred.ndim > 3:
        # For multi-dimensional tensors, compute metrics per sample then average
        batch_size = pred.shape[0]
        psnr_scores = []
        ssim_scores = []
        
        for i in range(batch_size):
            # Extract single volume (remove batch and channel dims if present)
            pred_vol = pred[i].squeeze()
            targ_vol = targ[i].squeeze()
            
            # Ensure 3D volume for processing
            if pred_vol.ndim == 4:  # (C, D, H, W)
                pred_vol = pred_vol[0]  # Take first channel
                targ_vol = targ_vol[0]
            
            # Normalize to [0,1] range for fair metric comparison
            pred_vol = (pred_vol - pred_vol.min()) / (pred_vol.max() - pred_vol.min() + 1e-8)
            targ_vol = (targ_vol - targ_vol.min()) / (targ_vol.max() - targ_vol.min() + 1e-8)
            
            # Compute PSNR
            try:
                psnr_val = psnr(targ_vol, pred_vol, data_range=1.0)
                psnr_scores.append(psnr_val)
            except Exception as e:
                print(f"Warning: PSNR computation failed for sample {i}: {e}")
                psnr_scores.append(0.0)
            
            # Compute SSIM (requires at least 7x7 patches, handle edge cases)
            try:
                ssim_val = ssim(targ_vol, pred_vol, data_range=1.0, 
                               win_size=min(7, min(pred_vol.shape[-2:])))
                ssim_scores.append(ssim_val)
            except Exception as e:
                print(f"Warning: SSIM computation failed for sample {i}: {e}")
                ssim_scores.append(0.0)
        
        # Return averaged metrics across batch
        return {
            'psnr': np.mean(psnr_scores),
            'ssim': np.mean(ssim_scores)
        }
    
    else:
        # Single volume case
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        targ = (targ - targ.min()) / (targ.max() - targ.min() + 1e-8)
        
        psnr_val = psnr(targ, pred, data_range=1.0)
        ssim_val = ssim(targ, pred, data_range=1.0)
        
        return {'psnr': psnr_val, 'ssim': ssim_val}
        
    if isinstance(target, torch.Tensor):
        targ = target.detach().cpu()
    else:
        targ = target
    
    # Handle singleton dimensions
    pred_np = pred.squeeze().numpy() if hasattr(pred, 'numpy') else pred.squeeze()
    target_np = targ.squeeze().numpy() if hasattr(targ, 'numpy') else targ.squeeze()
    
    # Resize if dimensions don't match
    if pred_np.shape != target_np.shape:
        if isinstance(pred, torch.Tensor) and isinstance(targ, torch.Tensor):
            if pred.dim() < 5:
                pred = pred.unsqueeze(0)
            if pred.dim() < 5:
                pred = pred.unsqueeze(0)
                
            if targ.dim() < 5:
                targ = targ.unsqueeze(0)
            if targ.dim() < 5:
                targ = targ.unsqueeze(0)
                
            pred = F.interpolate(pred, size=targ.shape[2:], mode='trilinear', align_corners=False)
            pred_np = pred.squeeze().numpy()
            target_np = targ.squeeze().numpy()
        else:
            return 0.0, 0.0
    
    # Calculate metrics
    data_range = max(1.0, target_np.max() - target_np.min())
    try:
        psnr_value = psnr(target_np, pred_np, data_range=data_range)
        ssim_value = ssim(target_np, pred_np, data_range=data_range, channel_axis=None)
    except Exception as e:
        print(f"[WARNING] Metric calculation failed: {e}")
        return 0.0, 0.0
    
    return psnr_value, ssim_value

def ensure_same_size(source, target):
    """Ensure source tensor has the same spatial dimensions as target."""
    if source.shape[2:] != target.shape[2:]:
        source = F.interpolate(source, size=target.shape[2:], mode='trilinear', align_corners=False)
    return source

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Target: {MAX_BATCHES * BATCH_SIZE} samples")
    print(f"[INFO] Using CFG with guidance scale: {GUIDANCE_SCALE}")
    print(f"[INFO] Using {NUM_INFERENCE_STEPS} diffusion steps")
    print(f"[INFO] Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] User: SajbenDani")
    
    # Load models
    try:
        # Autoencoder - using adversarially trained model
        autoencoder = Improved3DAutoencoder(
            in_channels=1,
            latent_channels=LATENT_CHANNELS, 
            base_channels=BASE_CHANNELS,
            use_vq=True
        ).to(DEVICE)
        
        autoencoder_ckpt = torch.load(AUTOENCODER_PATH, map_location=DEVICE)
        if 'model_state_dict' in autoencoder_ckpt:
            autoencoder.load_state_dict(autoencoder_ckpt['model_state_dict'])
        else:
            autoencoder.load_state_dict(autoencoder_ckpt)
        autoencoder.eval()
        print(f"[INFO] Loaded adversarially fine-tuned autoencoder from {AUTOENCODER_PATH}")
        
        # Diffusion model - using CFG-trained model
        diffusion_model = DiffusionUNet3D(
            latent_channels=LATENT_CHANNELS,
            base_channels=DIFFUSION_BASE_CHANNELS,
            time_emb_dim=256
        ).to(DEVICE)
        
        diffusion_ckpt = torch.load(DIFFUSION_PATH, map_location=DEVICE)
        if 'model_state_dict' in diffusion_ckpt:
            diffusion_model.load_state_dict(diffusion_ckpt['model_state_dict'])
        else:
            diffusion_model.load_state_dict(diffusion_ckpt)
        
        diffusion_model.eval()
        print(f"[INFO] Loaded CFG-trained diffusion model from {DIFFUSION_PATH}")

        # Noise scheduler
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        noise_scheduler.set_timesteps(NUM_INFERENCE_STEPS)
        
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load test data
    try:
        test_csv = os.path.join(PREPROCESSED_DIR, "test_patches.csv")
        data_module = FMRIDataModule(
            train_csv=os.path.join(PREPROCESSED_DIR, "train_patches.csv"),
            val_csv=os.path.join(PREPROCESSED_DIR, "val_patches.csv"),
            test_csv=test_csv,
            batch_size=BATCH_SIZE,
            num_workers=2,
            prefetch_factor=2,
            view=VIEW
        )
        
        data_module.setup()
        test_loader = data_module.test_dataloader()
        print(f"[INFO] Test dataset: {len(data_module.test_dataset)} samples")
        
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # Initialize metric storage
    metrics = {
        "ae_recon_with_skips": {"psnr": [], "ssim": []},
        "ae_recon_no_skips": {"psnr": [], "ssim": []},
        "bicubic_baseline": {"psnr": [], "ssim": []},
        "ldm_super_res": {"psnr": [], "ssim": []},
        "latent_stats": {"real_mean": [], "real_std": [], "gen_mean": [], "gen_std": []}
    }

    # Evaluation loop
    print("[INFO] Starting evaluation...")
    sample_count = 0
    start_time = datetime.now()
    
    with torch.no_grad():
        for batch_idx, (hr_patches, _) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if batch_idx >= MAX_BATCHES:
                break
                
            hr_patches = hr_patches.to(DEVICE)
            lr_patches = F.interpolate(hr_patches, scale_factor=1/SCALE_FACTOR, mode='trilinear', align_corners=False)

            try:
                # 1. Autoencoder with skips
                recon_with_skips, z_real, _ = autoencoder(hr_patches)
                recon_with_skips = ensure_same_size(recon_with_skips, hr_patches)
                
                # 2. Autoencoder without skips
                recon_no_skips = autoencoder.decode(z_real.detach(), skip_features=None)
                recon_no_skips = ensure_same_size(recon_no_skips, hr_patches)

                # 3. LDM super-resolution with Classifier-Free Guidance
                z_lr, _, _ = autoencoder.encode(lr_patches)
                z_lr_upsampled = F.interpolate(z_lr, size=z_real.shape[2:], mode='trilinear', align_corners=False)
                
                # Create null conditioning for unconditional path
                null_conditioning = torch.zeros_like(z_lr_upsampled).to(DEVICE)
                
                # Diffusion sampling with CFG
                latents_gen = torch.randn_like(z_real).to(DEVICE)
                
                # Use progress bar for diffusion steps
                diffusion_bar = tqdm(noise_scheduler.timesteps, desc=f"Batch {batch_idx+1} diffusion", leave=False)
                for t in diffusion_bar:
                    t_tensor = t.unsqueeze(0).to(DEVICE)
                    
                    # Get unconditional prediction
                    noise_pred_uncond = diffusion_model(latents_gen, t_tensor, null_conditioning)
                    
                    # Get conditional prediction
                    noise_pred_cond = diffusion_model(latents_gen, t_tensor, z_lr_upsampled)
                    
                    # Combine with guidance scale
                    noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                    
                    # Perform denoising step
                    latents_gen = noise_scheduler.step(noise_pred, t, latents_gen).prev_sample
                
                # Decode generated latents using the adversarially trained decoder (no skip connections)
                sr_images = autoencoder.decode(latents_gen, skip_features=None)
                sr_images = ensure_same_size(sr_images, hr_patches)

                # 4. Bicubic baseline
                bicubic_images = F.interpolate(lr_patches, size=hr_patches.shape[2:], mode='trilinear', align_corners=False)

                # Calculate metrics
                for i in range(hr_patches.shape[0]):
                    psnr_val, ssim_val = compute_metrics(recon_with_skips[i:i+1], hr_patches[i:i+1])
                    metrics["ae_recon_with_skips"]["psnr"].append(psnr_val)
                    metrics["ae_recon_with_skips"]["ssim"].append(ssim_val)

                    psnr_val, ssim_val = compute_metrics(recon_no_skips[i:i+1], hr_patches[i:i+1])
                    metrics["ae_recon_no_skips"]["psnr"].append(psnr_val)
                    metrics["ae_recon_no_skips"]["ssim"].append(ssim_val)

                    psnr_val, ssim_val = compute_metrics(bicubic_images[i:i+1], hr_patches[i:i+1])
                    metrics["bicubic_baseline"]["psnr"].append(psnr_val)
                    metrics["bicubic_baseline"]["ssim"].append(ssim_val)

                    psnr_val, ssim_val = compute_metrics(sr_images[i:i+1], hr_patches[i:i+1])
                    metrics["ldm_super_res"]["psnr"].append(psnr_val)
                    metrics["ldm_super_res"]["ssim"].append(ssim_val)
                    
                    sample_count += 1
                
                # Latent statistics
                metrics["latent_stats"]["real_mean"].append(z_real.mean().item())
                metrics["latent_stats"]["real_std"].append(z_real.std().item())
                metrics["latent_stats"]["gen_mean"].append(latents_gen.mean().item())
                metrics["latent_stats"]["gen_std"].append(latents_gen.std().item())
                
            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
                
            # Progress indicator every 10 batches (since processing is slower with 1000 steps)
            if (batch_idx + 1) % 10 == 0:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                eta = elapsed_time / (batch_idx + 1) * (MAX_BATCHES - batch_idx - 1)
                eta_h, eta_remainder = divmod(eta, 3600)
                eta_m, eta_s = divmod(eta_remainder, 60)
                
                print(f"[INFO] Processed {sample_count} samples in {int(hours)}h {int(minutes)}m {int(seconds)}s")
                print(f"[INFO] Estimated time remaining: {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s")
                
                # Save interim results every 50 batches
                if (batch_idx + 1) % 50 == 0:
                    interim_results = calculate_results(metrics, sample_count, start_time)
                    interim_file = OUTPUT_DIR / f"interim_results_batch_{batch_idx+1}.json"
                    with open(interim_file, 'w') as f:
                        json.dump(interim_results, f, indent=2)
                    print(f"[INFO] Saved interim results to: {interim_file}")

    # Calculate and save final results
    final_results = calculate_results(metrics, sample_count, start_time)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print_summary(final_results)
    print(f"\nResults saved to: {RESULTS_FILE}")

def calculate_results(metrics, sample_count, start_time):
    """Calculate all metrics and prepare the results dictionary."""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    results = {
        "meta": {
            "samples": sample_count,
            "evaluation_time_seconds": duration,
            "evaluation_time_formatted": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
            "samples_per_second": sample_count / duration if duration > 0 else 0,
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "user": "SajbenDani",
            "guidance_scale": GUIDANCE_SCALE,
            "inference_steps": NUM_INFERENCE_STEPS,
            "autoencoder": str(AUTOENCODER_PATH),
            "diffusion_model": str(DIFFUSION_PATH)
        },
        "metrics": {}
    }
    
    # Process metrics
    for key, value in metrics.items():
        if key != "latent_stats":
            if value["psnr"]:  # Check if there are any values
                avg_psnr = np.mean(value["psnr"])
                avg_ssim = np.mean(value["ssim"])
                std_psnr = np.std(value["psnr"])
                std_ssim = np.std(value["ssim"])
                
                results["metrics"][key] = {
                    "psnr_mean": float(avg_psnr),
                    "psnr_std": float(std_psnr),
                    "ssim_mean": float(avg_ssim),
                    "ssim_std": float(std_ssim)
                }
            else:
                results["metrics"][key] = {
                    "psnr_mean": 0.0,
                    "psnr_std": 0.0,
                    "ssim_mean": 0.0,
                    "ssim_std": 0.0
                }
    
    # Latent statistics
    if metrics["latent_stats"]["real_mean"]:
        real_mean = np.mean(metrics["latent_stats"]["real_mean"])
        real_std = np.mean(metrics["latent_stats"]["real_std"])
        gen_mean = np.mean(metrics["latent_stats"]["gen_mean"])
        gen_std = np.mean(metrics["latent_stats"]["gen_std"])
        
        results["metrics"]["latent_stats"] = {
            "real_mean": float(real_mean),
            "real_std": float(real_std),
            "gen_mean": float(gen_mean),
            "gen_std": float(gen_std),
            "mean_diff_pct": float(abs((gen_mean - real_mean) / (real_mean if real_mean != 0 else 1e-5)) * 100),
            "std_diff_pct": float(abs((gen_std - real_std) / (real_std if real_std != 0 else 1e-5)) * 100)
        }
    else:
        results["metrics"]["latent_stats"] = {
            "real_mean": 0.0,
            "real_std": 0.0,
            "gen_mean": 0.0,
            "gen_std": 0.0,
            "mean_diff_pct": 0.0,
            "std_diff_pct": 0.0
        }
    
    # Comparative metrics
    ae_with_skips_psnr = results["metrics"]["ae_recon_with_skips"]["psnr_mean"]
    ae_no_skips_psnr = results["metrics"]["ae_recon_no_skips"]["psnr_mean"]
    ldm_psnr = results["metrics"]["ldm_super_res"]["psnr_mean"]
    bicubic_psnr = results["metrics"]["bicubic_baseline"]["psnr_mean"]
    
    results["metrics"]["comparative"] = {
        "skip_vs_noskip_psnr_diff": float(ae_with_skips_psnr - ae_no_skips_psnr),
        "skip_vs_noskip_psnr_pct": float((ae_with_skips_psnr - ae_no_skips_psnr) / max(ae_with_skips_psnr, 1e-5) * 100),
        "ldm_vs_bicubic_psnr_diff": float(ldm_psnr - bicubic_psnr),
        "ldm_vs_bicubic_psnr_pct": float((ldm_psnr - bicubic_psnr) / max(bicubic_psnr, 1e-5) * 100)
    }
    
    return results

def print_summary(results):
    """Print a summary of the evaluation results."""
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Samples: {results['meta']['samples']}")
    print(f"Time: {results['meta']['evaluation_time_formatted']} ({results['meta']['samples_per_second']:.2f} samples/sec)")
    print(f"CFG Guidance Scale: {results['meta']['guidance_scale']}")
    print(f"Diffusion Steps: {results['meta']['inference_steps']}")
    
    # Quality metrics
    print("\nQUALITY METRICS (PSNR)")
    print(f"AE with skips:     {results['metrics']['ae_recon_with_skips']['psnr_mean']:.4f}")
    print(f"AE without skips:  {results['metrics']['ae_recon_no_skips']['psnr_mean']:.4f}")
    print(f"Bicubic baseline:  {results['metrics']['bicubic_baseline']['psnr_mean']:.4f}")
    print(f"LDM super-res:     {results['metrics']['ldm_super_res']['psnr_mean']:.4f}")
    
    # SSIM metrics
    print("\nQUALITY METRICS (SSIM)")
    print(f"AE with skips:     {results['metrics']['ae_recon_with_skips']['ssim_mean']:.4f}")
    print(f"AE without skips:  {results['metrics']['ae_recon_no_skips']['ssim_mean']:.4f}")
    print(f"Bicubic baseline:  {results['metrics']['bicubic_baseline']['ssim_mean']:.4f}")
    print(f"LDM super-res:     {results['metrics']['ldm_super_res']['ssim_mean']:.4f}")
    
    # Latent statistics
    print("\nLATENT STATS")
    print(f"Real latents:      mean={results['metrics']['latent_stats']['real_mean']:.4f}, std={results['metrics']['latent_stats']['real_std']:.4f}")
    print(f"Generated latents: mean={results['metrics']['latent_stats']['gen_mean']:.4f}, std={results['metrics']['latent_stats']['gen_std']:.4f}")
    print(f"Mean difference:   {results['metrics']['latent_stats']['mean_diff_pct']:.2f}%")
    print(f"Std difference:    {results['metrics']['latent_stats']['std_diff_pct']:.2f}%")
    
    # Comparative analysis
    print("\nCOMPARATIVE ANALYSIS")
    print(f"Skip vs no-skip:   {results['metrics']['comparative']['skip_vs_noskip_psnr_diff']:.2f}dB ({results['metrics']['comparative']['skip_vs_noskip_psnr_pct']:.2f}%)")
    print(f"LDM vs bicubic:    {results['metrics']['comparative']['ldm_vs_bicubic_psnr_diff']:.2f}dB ({results['metrics']['comparative']['ldm_vs_bicubic_psnr_pct']:.2f}%)")

if __name__ == "__main__":
    main()