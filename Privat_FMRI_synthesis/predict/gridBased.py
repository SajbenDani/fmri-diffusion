import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

# Get the parent directory of the current script (training/)
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add parent directory to sys.path
sys.path.append(PARENT_DIR)

# Import the necessary models
from models.diffusion import UNet3DDiffusion, LatentDiffusion
from models.autoencoder import Improved3DAutoencoder
from models.skipPredictor import SkipPredictor
import datetime

class ImprovedFMRIPredictor:
    def __init__(self, latent_dims=(8, 8, 8), num_classes=5, device='cuda', 
                 input_shape=(91, 109, 91), diffusion_steps=100):
        """
        Initialize the FMRI predictor.

        Args:
            latent_dims: Dimensions of the latent space (default: (8, 8, 8))
            num_classes: Number of classes for conditioning (default: 5)
            device: Device to run the models on (default: 'cuda')
            input_shape: Expected shape of the input image (D, H, W) (default: (91, 109, 91))
            diffusion_steps: Number of steps for the diffusion sampling (default: 100)
        """
        self.latent_dims = latent_dims
        self.num_classes = num_classes
        self.device = device
        self.input_shape = input_shape
        self.diffusion_steps = diffusion_steps

        # Initialize models as None; they will be loaded later
        self.autoencoder = None
        self.diffusion = None
        self.skip_predictor = None

        # Define directories (update these paths according to your local setup)
        self.models_dir = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/models"
        self.checkpoints_dir = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/checkpoints_New"
        self.logs_dir = "/home/jovyan/work/ssd0/USERS/sajbendaniel/fmri-diffusion/Privat_FMRI_synthesis/logs"

        # Define label categories for better visualization
        self.LABELS = {
            0: "Left_Hemisphere",
            1: "Right_Hemisphere",
            2: "Left_Foot",
            3: "Right_Foot",
            4: "Tongue"
        }

    def load_models(self):
        """Load the pretrained autoencoder, diffusion, and skip predictor models."""
        print("Loading models...")

        # Load the Improved3DAutoencoder
        self.autoencoder = Improved3DAutoencoder(
            latent_dims=self.latent_dims,
            num_classes=self.num_classes
        ).to(self.device)
        autoencoder_path = os.path.join(self.checkpoints_dir, "finetuned_autoencoder_best.pth")
        self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device))
        self.autoencoder.eval()

        # Load the LatentDiffusion model
        self.diffusion = LatentDiffusion(
            latent_shape=self.latent_dims,
            num_classes=self.num_classes,
            device=self.device
        )
        diffusion_path = os.path.join(self.checkpoints_dir, "latent_diffusion.pth")
        self.diffusion.model.load_state_dict(torch.load(diffusion_path, map_location=self.device))
        self.diffusion.model.eval()

        # Load the SkipPredictor
        self.skip_predictor = SkipPredictor(latent_dims=self.latent_dims).to(self.device)
        skip_predictor_path = os.path.join(self.checkpoints_dir, "skip_predictor_best.pth")
        self.skip_predictor.load_state_dict(torch.load(skip_predictor_path, map_location=self.device))
        self.skip_predictor.eval()

        print("Models loaded successfully!")

    def generate_one_hot(self, label_idx, batch_size=1):
        """Generate one-hot encoded labels for the specified class index."""
        one_hot = torch.zeros(batch_size, self.num_classes, device=self.device)
        one_hot[:, label_idx] = 1.0
        return one_hot

    def save_slice_image(self, volume, path, slice_dim=1, use_mip=False, grid=False):
        """
        Save a 2D slice or a grid of slices from a 3D volume as an image.

        Args:
            volume: 3D volume tensor (shape: [B, C, D, H, W] or [D, H, W])
            path: Path to save the image
            slice_dim: Dimension to project along (0, 1, or 2; default: 1)
            use_mip: Whether to use Maximum Intensity Projection (default: False)
            grid: Whether to plot a 2x4 grid of 8 slices (default: False)
        """
        if volume.ndim > 3:
            volume = volume.squeeze(0).squeeze(0)

        if grid:
            # Plot 8 slices in a 2x4 grid along D dimension
            D = volume.shape[0]
            slice_indices = np.linspace(0, D-1, 8, dtype=int)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            for i, ax in enumerate(axes.flat):
                slice_idx = slice_indices[i]
                slice_data = volume[slice_idx, :, :].detach().cpu().numpy()
                # Normalize
                if np.all(np.isnan(slice_data)) or np.all(np.isinf(slice_data)) or slice_data.size == 0:
                    ax.imshow(np.zeros_like(slice_data), cmap='Greens')
                    ax.set_title(f"Slice {slice_idx} - Invalid Data")
                else:
                    if np.min(slice_data) == np.max(slice_data):
                        norm = plt.Normalize(np.min(slice_data), np.min(slice_data) + 1)
                    else:
                        p_low, p_high = np.percentile(slice_data[~np.isnan(slice_data)], [2, 98])
                        if p_low == p_high:
                            p_high = p_low + 1
                        norm = plt.Normalize(p_low, p_high)
                    ax.imshow(slice_data, cmap='Greens', norm=norm, aspect='auto')
                    ax.set_title(f"Slice {slice_idx}")
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        else:
            # Plot a single slice or MIP
            if use_mip:
                if slice_dim == 0:
                    projection = torch.max(volume, dim=0)[0]
                elif slice_dim == 1:
                    projection = torch.max(volume, dim=1)[0]
                else:
                    projection = torch.max(volume, dim=2)[0]
                slice_data = projection
                title = f"MIP along dimension {slice_dim}"
            else:
                slice_idx = volume.shape[slice_dim] // 2
                if slice_dim == 0:
                    slice_data = volume[slice_idx, :, :]
                elif slice_dim == 1:
                    slice_data = volume[:, slice_idx, :]
                else:
                    slice_data = volume[:, :, slice_idx]
                title = f"Slice {slice_idx} along dimension {slice_dim}"
            slice_np = slice_data.detach().cpu().numpy()
            plt.figure(figsize=(10, 8))
            if np.all(np.isnan(slice_np)) or np.all(np.isinf(slice_np)) or slice_np.size == 0:
                plt.imshow(np.zeros_like(slice_np), cmap='Greens')
                plt.title(f"{title} - Invalid Data")
            else:
                if np.min(slice_np) == np.max(slice_np):
                    norm = plt.Normalize(np.min(slice_np), np.min(slice_np) + 1)
                else:
                    p_low, p_high = np.percentile(slice_np[~np.isnan(slice_np)], [2, 98])
                    if p_low == p_high:
                        p_high = p_low + 1
                    norm = plt.Normalize(p_low, p_high)
                im = plt.imshow(slice_np, cmap='Greens', norm=norm)
                plt.colorbar(im)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()

    def predict(self, label_indices=[0, 1, 2, 3, 4], batch_size=1, seed=42):
        """
        Generate fMRI images for specified labels using the diffusion model with SkipPredictor.

        Args:
            label_indices: List of label indices to generate images for
            batch_size: Batch size for generation
            seed: Random seed for reproducibility
        """
        # Set seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Load models if not already loaded
        if self.autoencoder is None or self.diffusion is None or self.skip_predictor is None:
            self.load_models()

        # Set the original size for the autoencoder
        self.autoencoder.original_size = self.input_shape

        # Define target shapes for skip connections
        target_shape_e1 = (46, 55, 46)
        target_shape_e2 = (23, 28, 23)

        # Create output directories
        os.makedirs(self.logs_dir, exist_ok=True)

        for label_idx in label_indices:
            label_name = self.LABELS.get(label_idx, f"label_{label_idx}")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            label_dir = os.path.join(self.logs_dir, f"{label_name}_{timestamp}")
            os.makedirs(label_dir, exist_ok=True)
            print(f"Generating images for {label_name} in {label_dir}...")

            # Generate one-hot label
            one_hot_label = self.generate_one_hot(label_idx, batch_size)

            # Generate initial noise in latent space
            torch.manual_seed(seed + label_idx if seed is not None else label_idx)
            initial_noise = torch.randn(batch_size, 1, *self.latent_dims, device=self.device)

            # Decode and save initial noise
            with torch.no_grad():
                pred_e1, pred_e2 = self.skip_predictor(initial_noise, target_shape_e1, target_shape_e2)
                initial_noise_flat = initial_noise.view(batch_size, -1)
                initial_decoded = self.autoencoder.decode(initial_noise_flat, pred_e1, pred_e2)
            self.save_slice_image(
                initial_decoded,
                os.path.join(label_dir, "initial_noise.png"),
                use_mip=True
            )

            # Run diffusion sampling
            self.diffusion.model.eval()
            with torch.no_grad():
                time_steps = torch.linspace(self.diffusion.timesteps - 1, 0, self.diffusion_steps, device=self.device)
                x = initial_noise.clone()

                for i, t in enumerate(time_steps):
                    t_batch = torch.ones(batch_size, device=self.device) * t
                    pred_noise = self.diffusion.model(x, t_batch, one_hot_label)
                    alpha = self.diffusion.noise_schedule(t_batch).view(-1, 1, 1, 1, 1)
                    alpha_next = self.diffusion.noise_schedule(
                        torch.max(t_batch - 1, torch.zeros_like(t_batch))
                    ).view(-1, 1, 1, 1, 1)
                    x = (x - (1 - alpha) * pred_noise) / torch.sqrt(alpha)
                    if i < self.diffusion_steps - 1:
                        noise = torch.randn_like(x, device=self.device)
                        sigma = torch.sqrt((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next))
                        x = torch.sqrt(alpha_next) * x + sigma * noise

            # Decode and save final image with grid
            final_latent = x
            with torch.no_grad():
                pred_e1, pred_e2 = self.skip_predictor(final_latent, target_shape_e1, target_shape_e2)
                final_flat = final_latent.view(batch_size, -1)
                final_decoded = self.autoencoder.decode(final_flat, pred_e1, pred_e2)
            self.save_slice_image(
                final_decoded,
                os.path.join(label_dir, "final_image.png"),
                grid=True
            )

            print(f"Generated images for {label_name} saved to {label_dir}")
            print(f"    - Initial noise: initial_noise.png")
            print(f"    - Final image: final_image.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    predictor = ImprovedFMRIPredictor(
        latent_dims=(8, 8, 8),
        num_classes=5,
        device=device,
        input_shape=(91, 109, 91),
        diffusion_steps=100
    )
    predictor.predict(label_indices=[0, 1, 2, 3, 4], batch_size=1, seed=42)
    print("Generation completed successfully!")