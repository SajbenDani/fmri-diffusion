import torch
import torch.nn as nn
import os
from diffusers import UNet2DModel
from config import DEVICE, DIFFUSION_CHECKPOINT

# Our autoencoder produces a latent vector of size 256.
# We reshape that vector into a (1, 16, 16) "image" because 16 x 16 = 256.
latent_dim = 1024  # Növelt latens tér
spatial_dim = int(latent_dim ** 0.5)  # Most 32x32

# Cosine noise scheduling function
def cosine_noise_schedule(t, T=1000):
    return torch.cos((t / T + 0.008) / 1.008 * torch.pi / 2) ** 2

# Multiplicative FiLM-based label embedding
class FiLMLabelEmbedding(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super().__init__()
        self.embedding = nn.Linear(num_classes, latent_dim)
        self.modulation = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()  # Sigmoid ensures scaling factor between 0 and 1
        )

    def forward(self, labels, latents):
        label_emb = self.embedding(labels)  # [B, latent_dim]
        scale = self.modulation(label_emb)  # [B, latent_dim]
        scale = scale.view(-1, 1, spatial_dim, spatial_dim)  # Reshape for modulation
        return latents * scale  # Multiplicative FiLM-style conditioning

# Conditional Diffusion Model with deeper UNet and FiLM embedding
class ConditionalDiffusion(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.label_embed = FiLMLabelEmbedding(num_classes, spatial_dim * spatial_dim)  # Use FiLM embedding
        
        # Deeper and wider UNet2DModel
        self.diffusion_model = UNet2DModel(
            sample_size=spatial_dim,  # 16x16
            in_channels=1,  # Only image input now (label modulation is multiplicative)
            out_channels=1,
            layers_per_block=3,  # Increased from 2 to 3
            block_out_channels=(128, 256, 512, 1024),  # Added an extra 1024 block
            down_block_types=(
                "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"
            ),
            norm_num_groups=32,  # GroupNorm for stability
        )

    def forward(self, x, timesteps, labels):
        batch_size = x.shape[0]

        # Apply FiLM-based label embedding
        x = self.label_embed(labels, x)  # Multiplicative modulation

        # Compute cosine noise scaling
        alpha_t = cosine_noise_schedule(timesteps)
        alpha_t = alpha_t.view(batch_size, 1, 1, 1)  # Reshape for broadcasting

        # Apply noise scheduling
        x = x * alpha_t  # Adjusting input noise level

        # Forward pass through diffusion model
        output = self.diffusion_model(x, timesteps)

        # Return the sample from the output
        if hasattr(output, 'sample'):
            return output.sample
        return output

# Initialize model
diffusion_model = ConditionalDiffusion().to(DEVICE)

# Load checkpoint if available
if os.path.exists(DIFFUSION_CHECKPOINT):
    diffusion_model.load_state_dict(torch.load(DIFFUSION_CHECKPOINT, map_location=DEVICE))
    print("Loaded diffusion checkpoint")