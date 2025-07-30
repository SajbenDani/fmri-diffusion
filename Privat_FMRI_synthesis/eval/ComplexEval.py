#!/usr/bin/env python3
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

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PARENT_DIR))

# Import models and dataset
from models.autoencoder import Improved3DAutoencoder
from models.diffusion import DiffusionUNet3D
from utils.dataset import FMRIDataModule
from diffusers import DDPMScheduler

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTOENCODER_PATH = PARENT_DIR / "checkpoints_adversarial" / "best_finetuned_adversarial_autoencoder.pt"
#DIFFUSION_PATH = PARENT_DIR / "checkpoints_diffusion" / "best_diffusion.pt"
DIFFUSION_PATH = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_diffusion_cfg/best_diffusion_cfg.pt"
PREPROCESSED_DIR = PARENT_DIR / "data_preprocessed"
OUTPUT_DIR = PARENT_DIR / "evaluation"
RESULTS_FILE = OUTPUT_DIR / f"numeric_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

LATENT_CHANNELS = 8
BASE_CHANNELS = 32
DIFFUSION_BASE_CHANNELS = 128
SCALE_FACTOR = 2
NUM_INFERENCE_STEPS = 250
BATCH_SIZE = 4
VIEW = 'axial'
MAX_BATCHES = 750  # ~3000 samples

def compute_metrics(prediction, target):
    """Compute PSNR and SSIM between prediction and target."""
    # Convert to CPU tensors
    if isinstance(prediction, torch.Tensor):
        pred = prediction.detach().cpu()
    else:
        pred = prediction
        
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
    except Exception:
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
    
    # Load models
    try:
        # Autoencoder
        autoencoder = Improved3DAutoencoder(
            in_channels=1,
            latent_channels=LATENT_CHANNELS, 
            base_channels=BASE_CHANNELS,
            use_vq=True
        ).to(DEVICE)
        
        autoencoder_ckpt = torch.load(AUTOENCODER_PATH, map_location=DEVICE)
        autoencoder.load_state_dict(autoencoder_ckpt['model_state_dict'])
        autoencoder.eval()
        
        # Diffusion model
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

        # Noise scheduler
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        noise_scheduler.set_timesteps(NUM_INFERENCE_STEPS)
        
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
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

                # 3. LDM super-resolution
                z_lr, _, _ = autoencoder.encode(lr_patches)
                z_lr_upsampled = F.interpolate(z_lr, size=z_real.shape[2:], mode='trilinear', align_corners=False)
                
                # Diffusion sampling
                latents_gen = torch.randn_like(z_real).to(DEVICE)
                for t in noise_scheduler.timesteps:
                    t_tensor = t.unsqueeze(0).to(DEVICE)
                    noise_pred = diffusion_model(latents_gen, t_tensor, z_lr_upsampled)
                    latents_gen = noise_scheduler.step(noise_pred, t, latents_gen).prev_sample
                
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
                continue
                
            # Progress indicator every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"[INFO] Processed {sample_count} samples")

    # Calculate final metrics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results = {
        "meta": {
            "samples": sample_count,
            "evaluation_time_seconds": duration,
            "samples_per_second": sample_count / duration,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": "SajbenDani"
        },
        "metrics": {}
    }
    
    # Process metrics
    for key, value in metrics.items():
        if key != "latent_stats":
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
    
    # Latent statistics
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
    
    # Comparative metrics
    ae_with_skips_psnr = results["metrics"]["ae_recon_with_skips"]["psnr_mean"]
    ae_no_skips_psnr = results["metrics"]["ae_recon_no_skips"]["psnr_mean"]
    ldm_psnr = results["metrics"]["ldm_super_res"]["psnr_mean"]
    bicubic_psnr = results["metrics"]["bicubic_baseline"]["psnr_mean"]
    
    results["metrics"]["comparative"] = {
        "skip_vs_noskip_psnr_diff": float(ae_with_skips_psnr - ae_no_skips_psnr),
        "skip_vs_noskip_psnr_pct": float((ae_with_skips_psnr - ae_no_skips_psnr) / ae_with_skips_psnr * 100),
        "ldm_vs_bicubic_psnr_diff": float(ldm_psnr - bicubic_psnr),
        "ldm_vs_bicubic_psnr_pct": float((ldm_psnr - bicubic_psnr) / bicubic_psnr * 100)
    }
    
    # Save results to JSON file
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Samples: {sample_count}")
    print(f"Time: {duration:.2f} seconds ({sample_count/duration:.2f} samples/sec)")
    print("\nQUALITY METRICS (PSNR)")
    print(f"AE with skips:     {ae_with_skips_psnr:.4f}")
    print(f"AE without skips:  {ae_no_skips_psnr:.4f}")
    print(f"Bicubic baseline:  {bicubic_psnr:.4f}")
    print(f"LDM super-res:     {ldm_psnr:.4f}")
    print("\nLATENT STATS")
    print(f"Real latents:      mean={real_mean:.4f}, std={real_std:.4f}")
    print(f"Generated latents: mean={gen_mean:.4f}, std={gen_std:.4f}")
    print(f"Mean difference:   {results['metrics']['latent_stats']['mean_diff_pct']:.2f}%")
    print(f"Std difference:    {results['metrics']['latent_stats']['std_diff_pct']:.2f}%")
    print("\nCOMPARATIVE ANALYSIS")
    print(f"Skip vs no-skip:   {results['metrics']['comparative']['skip_vs_noskip_psnr_diff']:.2f}dB ({results['metrics']['comparative']['skip_vs_noskip_psnr_pct']:.2f}%)")
    print(f"LDM vs bicubic:    {results['metrics']['comparative']['ldm_vs_bicubic_psnr_diff']:.2f}dB ({results['metrics']['comparative']['ldm_vs_bicubic_psnr_pct']:.2f}%)")
    print(f"\nResults saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    main()