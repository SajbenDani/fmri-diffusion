#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from diffusers import DDPMScheduler
from monai.inferers import sliding_window_inference

# --- Setup Paths ---
PARENT_DIR = Path(__file__).parent.parent
sys.path.append(str(PARENT_DIR))

# Import the models
from models.autoencoder import Improved3DAutoencoder
from models.diffusion import DiffusionUNet3D

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTOENCODER_PATH = PARENT_DIR / "checkpoints_finetuned" / "best_finetuned_autoencoder.pt"
DIFFUSION_PATH = PARENT_DIR / "checkpoints_diffusion" / "best_diffusion.pt"
OUTPUT_DIR = PARENT_DIR / "logs"
ORIGINAL_DATA_DIR = Path("/home/jovyan/work/ssd0/USERS/siposlevente/data/fmri")

# Model parameters
LATENT_CHANNELS = 8
BASE_CHANNELS = 32
SCALE_FACTOR = 2
NUM_INFERENCE_STEPS = 500
ROI_SIZE_HR = (64, 64, 64) 
GUIDANCE_SCALE = 0

# Define custom colormap
colors = [(0, 0, 0), (0, 1, 0.5)]
cm = LinearSegmentedColormap.from_list('custom_green', colors, N=256)

def plot_slices(data_np, output_path, title):
    """Plot 2x4 slices of 3D data. Assumes input shape is (H, W, D)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    depth = data_np.shape[2]
    slice_indices = np.linspace(0, depth - 1, 8, dtype=int)
    
    for i, slice_idx in enumerate(slice_indices):
        axes[i].imshow(data_np[:, :, slice_idx], cmap=cm, vmin=0, vmax=1)
        axes[i].set_title(f"Slice {slice_idx}")
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved visualization to: {output_path}")

def analyze_checkpoint_and_model(checkpoint_path, model):
    """
    Analyze both checkpoint and model to determine compatibility and loading strategy.
    Returns: (model_is_cfg, checkpoint_is_cfg, loading_strategy)
    """
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Analyze checkpoint - look for first conv layer
    checkpoint_input_channels = None
    checkpoint_first_conv_key = None
    for key, tensor in state_dict.items():
        if 'conv' in key.lower() and 'weight' in key and tensor.dim() == 5:  # 3D conv weight
            checkpoint_input_channels = tensor.shape[1]  # Input channels
            checkpoint_first_conv_key = key
            break
    
    # Analyze model - look for first conv layer
    model_input_channels = None
    model_first_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            model_input_channels = module.in_channels
            model_first_conv = module
            break
    
    print(f"Checkpoint first conv key: {checkpoint_first_conv_key}")
    print(f"Checkpoint expects {checkpoint_input_channels} input channels")
    print(f"Model expects {model_input_channels} input channels")
    
    # Determine if each is CFG-style
    single_latent_channels = LATENT_CHANNELS
    checkpoint_is_cfg = checkpoint_input_channels == single_latent_channels * 2
    model_is_cfg = model_input_channels == single_latent_channels * 2
    
    print(f"Checkpoint is CFG-style: {checkpoint_is_cfg}")
    print(f"Model is CFG-style: {model_is_cfg}")
    
    # Determine loading strategy
    if model_is_cfg and checkpoint_is_cfg:
        strategy = "direct"  # Direct loading
    elif not model_is_cfg and not checkpoint_is_cfg:
        strategy = "direct"  # Direct loading
    elif model_is_cfg and not checkpoint_is_cfg:
        strategy = "adapt_to_cfg"  # Need to adapt non-CFG checkpoint to CFG model
    elif not model_is_cfg and checkpoint_is_cfg:
        strategy = "adapt_to_non_cfg"  # Need to adapt CFG checkpoint to non-CFG model
    else:
        strategy = "unknown"
    
    print(f"Loading strategy: {strategy}")
    return model_is_cfg, checkpoint_is_cfg, strategy

def adapt_checkpoint_for_cfg_model(state_dict, original_channels, target_channels):
    """
    Adapt a non-CFG checkpoint to work with a CFG model by duplicating/splitting weights.
    """
    print(f"Adapting checkpoint: {original_channels} -> {target_channels} channels")
    
    adapted_state_dict = {}
    for key, tensor in state_dict.items():
        if 'conv' in key.lower() and 'weight' in key and tensor.dim() == 5:
            # This is a 3D conv weight tensor
            if tensor.shape[1] == original_channels and key.endswith('.weight'):
                # First conv layer - need to adapt input channels
                if target_channels == original_channels * 2:
                    # Duplicate the weights for both channels (noisy + conditioning)
                    print(f"Duplicating weights for {key}: {tensor.shape} -> {(tensor.shape[0], target_channels, *tensor.shape[2:])}")
                    duplicated_tensor = torch.cat([tensor, tensor], dim=1)
                    adapted_state_dict[key] = duplicated_tensor
                else:
                    adapted_state_dict[key] = tensor
            else:
                adapted_state_dict[key] = tensor
        else:
            adapted_state_dict[key] = tensor
    
    return adapted_state_dict

def load_models():
    print("Loading models...")
    autoencoder = Improved3DAutoencoder(in_channels=1, latent_channels=LATENT_CHANNELS, base_channels=BASE_CHANNELS, use_vq=True).to(DEVICE)
    checkpoint = torch.load(AUTOENCODER_PATH, map_location=DEVICE)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()
    print(f"Loaded adversarial autoencoder from: {AUTOENCODER_PATH}")

    # Create diffusion model
    diffusion_model = DiffusionUNet3D(latent_channels=LATENT_CHANNELS, base_channels=128, time_emb_dim=256).to(DEVICE)
    
    # Analyze checkpoint and model compatibility
    model_is_cfg, checkpoint_is_cfg, strategy = analyze_checkpoint_and_model(DIFFUSION_PATH, diffusion_model)
    
    # Load checkpoint with appropriate strategy
    diffusion_ckpt = torch.load(DIFFUSION_PATH, map_location=DEVICE)
    state_dict = diffusion_ckpt['model_state_dict'] if 'model_state_dict' in diffusion_ckpt else diffusion_ckpt
    
    if strategy == "direct":
        print("Using direct loading")
        missing_keys, unexpected_keys = diffusion_model.load_state_dict(state_dict, strict=False)
        
    elif strategy == "adapt_to_cfg":
        print("Adapting non-CFG checkpoint to CFG model")
        adapted_state_dict = adapt_checkpoint_for_cfg_model(
            state_dict, 
            LATENT_CHANNELS,  # original (non-CFG)
            LATENT_CHANNELS * 2  # target (CFG)
        )
        missing_keys, unexpected_keys = diffusion_model.load_state_dict(adapted_state_dict, strict=False)
        
    elif strategy == "adapt_to_non_cfg":
        print("Warning: CFG checkpoint -> non-CFG model not implemented. Using partial loading.")
        missing_keys, unexpected_keys = diffusion_model.load_state_dict(state_dict, strict=False)
        
    else:
        print("Unknown loading strategy, using flexible loading")
        missing_keys, unexpected_keys = diffusion_model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys[:3]}...")
    if unexpected_keys:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
    
    diffusion_model.eval()
    print(f"Loaded diffusion model from: {DIFFUSION_PATH}")

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    print("Models loaded successfully.")
    
    return autoencoder, diffusion_model, noise_scheduler, model_is_cfg

class RobustDiffusionSR:
    """Robust super-resolution pipeline that adapts to model/checkpoint mismatches."""
    def __init__(self, autoencoder, diffusion_model, noise_scheduler, num_steps=1000, guidance_scale=7.5, model_is_cfg=True):
        self.autoencoder = autoencoder
        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler.set_timesteps(num_steps)
        self.guidance_scale = guidance_scale
        self.model_is_cfg = model_is_cfg
        
        # Adjust guidance scale based on model type
        if not self.model_is_cfg and guidance_scale > 0:
            print(f"Model is non-CFG, setting guidance_scale to 0 (was {guidance_scale})")
            self.guidance_scale = 0
        elif self.model_is_cfg and guidance_scale == 0:
            print(f"Model is CFG but guidance_scale is 0 - using unconditional generation")
        
        print(f"Using guidance scale: {self.guidance_scale}")
        print(f"Using {num_steps} diffusion steps")
        print(f"Model type: {'CFG' if self.model_is_cfg else 'Standard'}")
    
    @torch.no_grad()
    def __call__(self, lr_patch):
        z_lr, _, _ = self.autoencoder.encode(lr_patch)
        hr_latent_shape = (
            lr_patch.shape[2] * SCALE_FACTOR // 8,
            lr_patch.shape[3] * SCALE_FACTOR // 8,
            lr_patch.shape[4] * SCALE_FACTOR // 8
        )
        z_lr_upsampled = F.interpolate(z_lr, size=hr_latent_shape, mode='trilinear')
        latents = torch.randn((lr_patch.shape[0], LATENT_CHANNELS, *hr_latent_shape), device=DEVICE)
        
        for t in tqdm(self.noise_scheduler.timesteps, desc="Generating SR Patch", leave=False):
            t_tensor = torch.tensor([t], device=DEVICE)
            
            try:
                if self.model_is_cfg:
                    # CFG model
                    if self.guidance_scale > 0:
                        # CFG with guidance
                        null_conditioning = torch.zeros_like(z_lr_upsampled)
                        
                        # Get unconditional prediction
                        noise_pred_uncond = self.diffusion_model(latents, t_tensor, null_conditioning)
                        
                        # Get conditional prediction
                        noise_pred_cond = self.diffusion_model(latents, t_tensor, z_lr_upsampled)
                        
                        # Combine with guidance scale
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        # CFG model but no guidance
                        null_conditioning = torch.zeros_like(z_lr_upsampled)
                        noise_pred = self.diffusion_model(latents, t_tensor, null_conditioning)
                else:
                    # Non-CFG model
                    noise_pred = self.diffusion_model(latents, t_tensor, z_lr_upsampled)
                    
            except Exception as e:
                print(f"Error in diffusion forward pass: {e}")
                print("Trying fallback approaches...")
                
                # Fallback 1: Try without conditioning
                try:
                    noise_pred = self.diffusion_model(latents, t_tensor)
                    print("Fallback 1 successful: no conditioning")
                except:
                    # Fallback 2: Try with zero conditioning
                    try:
                        zero_cond = torch.zeros_like(z_lr_upsampled)
                        noise_pred = self.diffusion_model(latents, t_tensor, zero_cond)
                        print("Fallback 2 successful: zero conditioning")
                    except Exception as e2:
                        print(f"All fallbacks failed: {e2}")
                        raise e
            
            # Perform denoising step
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        sr_patch = self.autoencoder.decode(latents, skip_features=None)
        return sr_patch

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    autoencoder, diffusion_model, noise_scheduler, model_is_cfg = load_models()
    if autoencoder is None:
        return

    # Load original HR file
    sample_base_path = ORIGINAL_DATA_DIR / '100307' / 'tfMRI_MOTOR_RL'
    hr_file_path = None
    for ext in ['.nii', '.nii.gz']:
        potential_path = sample_base_path.with_suffix(ext)
        if potential_path.exists():
            hr_file_path = potential_path
            break
    if not hr_file_path:
        print(f"Error: Original fMRI file not found at {sample_base_path}")
        return

    print(f"Loading original HR file: {hr_file_path}")
    nii_img = nib.load(hr_file_path)
    hr_data = nii_img.get_fdata(dtype=np.float32)
    hr_data = np.mean(hr_data, axis=3) if hr_data.ndim == 4 else hr_data
    hr_data = (hr_data - hr_data.min()) / (hr_data.max() - hr_data.min())
    print(f"Original full volume shape (W, H, D): {hr_data.shape}")

    # Convert to tensor with shape (B, C, W, H, D)
    hr_tensor = torch.from_numpy(hr_data.copy()).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    print("Creating synthetic low-resolution volume...")
    lr_tensor = F.interpolate(hr_tensor, scale_factor=1/SCALE_FACTOR, mode='trilinear', align_corners=False)
    
    # Initialize the pipeline
    sr_pipeline = RobustDiffusionSR(
        autoencoder, 
        diffusion_model, 
        noise_scheduler, 
        NUM_INFERENCE_STEPS, 
        GUIDANCE_SCALE,
        model_is_cfg
    )
    
    # Extract a central portion to process
    print("Processing central portion of the volume...")
    center = [dim // 2 for dim in hr_data.shape]
    region_size = min(96, min(hr_data.shape))
    
    half_size = region_size // 2
    hr_region = hr_tensor[:, :, 
                          max(0, center[0] - half_size):min(hr_data.shape[0], center[0] + half_size),
                          max(0, center[1] - half_size):min(hr_data.shape[1], center[1] + half_size),
                          max(0, center[2] - half_size):min(hr_data.shape[2], center[2] + half_size)]
    
    lr_region = F.interpolate(hr_region, scale_factor=1/SCALE_FACTOR, mode='trilinear', align_corners=False)
    roi_size_lr = tuple(s // SCALE_FACTOR for s in ROI_SIZE_HR)
    
    print(f"Running sliding window inference on region with LR patch size: {roi_size_lr}...")
    try:
        sr_region = sliding_window_inference(
            inputs=lr_region,
            roi_size=roi_size_lr,
            sw_batch_size=1,
            predictor=sr_pipeline,
            overlap=0.5,
            mode="gaussian"
        )
        
        # Prepare images for plotting
        hr_np = hr_region.squeeze().cpu().numpy()
        lr_np = F.interpolate(lr_region, size=hr_region.shape[2:], mode='trilinear').squeeze().cpu().numpy()
        sr_np = sr_region.squeeze().cpu().numpy()
        
    except Exception as e:
        print(f"Error during sliding window inference: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save visualizations
    basename = hr_file_path.name.split('.')[0]
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_type = "CFG" if sr_pipeline.model_is_cfg else "Standard"
    guidance_str = f"_guidance_{sr_pipeline.guidance_scale}" if sr_pipeline.model_is_cfg else ""
    
    plot_slices(hr_np, OUTPUT_DIR / f"{basename}_{timestamp}_1_original_HR.png", 
                "Original High-Resolution")
    plot_slices(lr_np, OUTPUT_DIR / f"{basename}_{timestamp}_2_input_LR.png", 
                "Low-Resolution Input (Upscaled)")
    plot_slices(sr_np, OUTPUT_DIR / f"{basename}_{timestamp}_3_superresolved_{model_type}{guidance_str}.png", 
                f"Super-Resolved with {model_type} Model (Guidance: {sr_pipeline.guidance_scale})")
    
    print("\n=== PREDICTION COMPLETE ===")
    print("Generated three images for visual comparison:")
    print(f"1. Original HR: {OUTPUT_DIR}/{basename}_{timestamp}_1_original_HR.png")
    print(f"2. Low-res:     {OUTPUT_DIR}/{basename}_{timestamp}_2_input_LR.png")
    print(f"3. Super-res:   {OUTPUT_DIR}/{basename}_{timestamp}_3_superresolved_{model_type}{guidance_str}.png")
    print(f"Model type: {model_type}")
    print(f"Guidance scale used: {sr_pipeline.guidance_scale}")
    print(f"Current Date and Time (UTC): 2025-08-11 14:47:29")
    print(f"Current User's Login: SajbenDani")

if __name__ == "__main__":
    main()