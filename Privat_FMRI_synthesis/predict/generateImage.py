import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
import imageio
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
import io
from PIL import Image

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
        
        # Initialize brain mask as None; will be created later
        self.brain_mask = None

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
        
        # Create a simple brain contour/mask
        self.create_brain_mask()

    def create_brain_mask(self):
        """Create a simple ellipsoid brain mask for contour visualization."""
        D, H, W = self.input_shape
        
        # Create coordinate grids
        z, y, x = np.ogrid[:D, :H, :W]
        
        # Center points
        center_z, center_y, center_x = D//2, H//2, W//2
        
        # Create an ellipsoid mask
        # Adjust these parameters to change the shape of the "brain"
        z_radius, y_radius, x_radius = D//2 - 5, H//2 - 5, W//2 - 5
        
        # Ellipsoid equation
        distance = ((z - center_z)**2 / (z_radius**2) + 
                   (y - center_y)**2 / (y_radius**2) + 
                   (x - center_x)**2 / (x_radius**2))
        
        # Create mask (1 inside brain, 0 outside)
        mask = (distance <= 1.0).astype(np.float32)
        
        # Convert to torch tensor
        self.brain_mask = torch.from_numpy(mask).to(self.device)
        print("Brain mask created successfully!")

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

    def save_slice_image(self, volume, path, slice_dim=1, slice_idx=None, use_mip=True):
        """
        Save a 2D slice from a 3D volume as an image with enhanced visualization.
        Can use Maximum Intensity Projection (MIP) instead of a single slice.
        Now includes brain contour.

        Args:
            volume: 3D volume tensor (shape: [B, C, D, H, W] or [D, H, W])
            path: Path to save the image
            slice_dim: Dimension to project along (0, 1, or 2; default: 1)
            slice_idx: Index of the slice (if None, uses middle slice) - used only if use_mip=False
            use_mip: Whether to use Maximum Intensity Projection (default: True)
        """
        # Remove batch and channel dimensions if present
        if volume.ndim > 3:
            volume = volume.squeeze(0).squeeze(0)

        # Apply Maximum Intensity Projection if requested
        if use_mip:
            # Project along the specified dimension
            if slice_dim == 0:
                # MIP along D dimension (axis 0)
                projection = torch.max(volume, dim=0)[0]
                # Project brain mask for contour
                brain_contour = torch.max(self.brain_mask, dim=0)[0]
            elif slice_dim == 1:
                # MIP along H dimension (axis 1)
                projection = torch.max(volume, dim=1)[0]
                # Project brain mask for contour
                brain_contour = torch.max(self.brain_mask, dim=1)[0]
            else:
                # MIP along W dimension (axis 2)
                projection = torch.max(volume, dim=2)[0]
                # Project brain mask for contour
                brain_contour = torch.max(self.brain_mask, dim=2)[0]

            slice_data = projection
        else:
            # Extract a single slice based on slice_dim (original behavior)
            if slice_idx is None:
                slice_idx = volume.shape[slice_dim] // 2

            if slice_dim == 0:
                slice_data = volume[slice_idx, :, :]
                brain_contour = self.brain_mask[slice_idx, :, :]
            elif slice_dim == 1:
                slice_data = volume[:, slice_idx, :]
                brain_contour = self.brain_mask[:, slice_idx, :]
            else:
                slice_data = volume[:, :, slice_idx]
                brain_contour = self.brain_mask[:, :, slice_idx]

        # Convert to numpy
        slice_np = slice_data.detach().cpu().numpy()
        brain_contour_np = brain_contour.detach().cpu().numpy()

        # Create a figure with enhanced visualization
        plt.figure(figsize=(10, 8))

        # Check if the data contains valid values before normalization
        if np.all(np.isnan(slice_np)) or np.all(np.isinf(slice_np)) or slice_np.size == 0:
            # Handle the case with invalid data
            plt.imshow(np.zeros_like(slice_np), cmap='Greens')
            plt.title("Warning: Invalid data in slice")
        else:
            # Ensure we have a range of values for normalization
            if np.min(slice_np) == np.max(slice_np):
                # If all values are the same, use a simple normalization
                norm = plt.Normalize(np.min(slice_np), np.min(slice_np) + 1)
            else:
                # Use percentile-based normalization when we have a range of values
                p_low, p_high = np.percentile(slice_np[~np.isnan(slice_np)], [2, 98])
                # Ensure p_low != p_high to avoid division by zero
                if p_low == p_high:
                    p_high = p_low + 1
                norm = plt.Normalize(p_low, p_high)

            # Plot the data with the normalization
            im = plt.imshow(slice_np, cmap='Greens', norm=norm)
            plt.colorbar(im)  # Pass the image to colorbar to ensure proper mapping

        # Add brain contour
        plt.contour(brain_contour_np, levels=[0.5], colors='red', linewidths=1.5)

        if use_mip:
            plt.title(f"Maximum Intensity Projection along dimension {slice_dim}")
        else:
            plt.title(f"Slice {slice_idx} along dimension {slice_dim}")

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def predict(self, label_indices=[0, 1, 2, 3, 4], batch_size=1, seed=42, total_duration=12):
        """
        Generate fMRI images for specified labels using the diffusion model with SkipPredictor.

        Args:
            label_indices: List of label indices to generate images for
            batch_size: Batch size for generation
            seed: Random seed for reproducibility
            total_duration: Desired GIF duration in seconds (default: 12)
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
                slice_dim=1,
                use_mip=True  # Use MIP for initial noise
            )

            # Prepare for collecting frames for GIF (all steps, in memory)
            intermediate_frames = []
            frame_indices = list(range(self.diffusion_steps))  # Collect all steps

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

                    # Generate frame in memory
                    # Inside the predict method, replace the GIF generation code:

                    # Generate frame in memory
                    if i in frame_indices:
                        pred_e1, pred_e2 = self.skip_predictor(x, target_shape_e1, target_shape_e2)
                        x_flat = x.view(batch_size, -1)
                        decoded = self.autoencoder.decode(x_flat, pred_e1, pred_e2)

                        # Create matplotlib figure in memory with fixed size
                        fig = plt.figure(figsize=(10, 8))  # Fixed figure size
                        ax = fig.add_subplot(111)

                        # Apply MIP along dimension 1 (H)
                        slice_data = decoded.squeeze(0).squeeze(0)
                        # Max projection along dimension 1 (H)
                        mip_slice = torch.max(slice_data, dim=1)[0].detach().cpu().numpy()
                        brain_contour = torch.max(self.brain_mask, dim=1)[0].detach().cpu().numpy()

                        # Check for valid data
                        if np.all(np.isnan(mip_slice)) or np.all(np.isinf(mip_slice)) or mip_slice.size == 0:
                            # Handle invalid data
                            im = ax.imshow(np.zeros_like(mip_slice), cmap='Greens')
                            ax.set_title(f"Step {i}/{len(time_steps)-1} - MIP (Invalid Data)")
                        else:
                            # Check if all values are the same
                            if np.min(mip_slice) == np.max(mip_slice):
                                norm = plt.Normalize(np.min(mip_slice), np.min(mip_slice) + 1)
                            else:
                                p_low, p_high = np.percentile(mip_slice[~np.isnan(mip_slice)], [2, 98])
                                # Ensure p_low != p_high
                                if p_low == p_high:
                                    p_high = p_low + 1
                                norm = plt.Normalize(p_low, p_high)

                            im = ax.imshow(mip_slice, cmap='Greens', norm=norm)
                            ax.set_title(f"Step {i}/{len(time_steps)-1} - MIP")

                        # Add brain contour
                        ax.contour(brain_contour, levels=[0.5], colors='red', linewidths=1.5)

                        fig.colorbar(im, ax=ax)  # Pass the image object to colorbar

                        # Convert figure to image array with consistent size
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        frame = np.array(Image.open(buf))
                        intermediate_frames.append(frame)
                        buf.close()
                        plt.close(fig)

            # Decode and save final image
            final_latent = x
            with torch.no_grad():
                pred_e1, pred_e2 = self.skip_predictor(final_latent, target_shape_e1, target_shape_e2)
                final_flat = final_latent.view(batch_size, -1)
                final_decoded = self.autoencoder.decode(final_flat, pred_e1, pred_e2)
            self.save_slice_image(
                final_decoded,
                os.path.join(label_dir, "final_image.png"),
                slice_dim=1,
                use_mip=True  # Use MIP for final image
            )

            # Create and save GIF from in-memory frames
            gif_path = os.path.join(label_dir, "diffusion_process.gif")
            if intermediate_frames:
                duration_per_frame = total_duration / len(intermediate_frames)
                print(f"Creating GIF with {len(intermediate_frames)} frames, each {duration_per_frame:.4f} seconds")
                imageio.mimsave(gif_path, intermediate_frames, duration=duration_per_frame)
                print(f"GIF saved to {gif_path}")

            print(f"Generated images for {label_name} saved to {label_dir}")
            print(f"    - Initial noise: initial_noise.png")
            print(f"    - Final image: final_image.png")
            print(f"    - Process GIF: diffusion_process.gif")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    predictor = ImprovedFMRIPredictor(
        latent_dims=(8, 8, 8),
        num_classes=5,
        device=device,
        input_shape=(91, 109, 91),
        diffusion_steps=100  # Csökkentve 500-ról 100-ra
    )
    predictor.predict(label_indices=[0, 1, 2, 3, 4], batch_size=1, seed=42, total_duration=12)
    print("Generation completed successfully!")