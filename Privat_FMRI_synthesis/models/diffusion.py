import torch
import torch.nn as nn
import os
from diffusers import UNet2DModel
from config import DEVICE, DIFFUSION_CHECKPOINT

# Our autoencoder produces a latent vector of size 256.
# We reshape that vector into a (1, 16, 16) "image" because 16 x 16 = 256.
latent_dim = 256
spatial_dim = int(latent_dim ** 0.5)  # 16

# Create a simple conditional diffusion model that works with the diffusers library
class ConditionalDiffusion(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, spatial_dim * spatial_dim)
        
        # The base diffusion model
        self.diffusion_model = UNet2DModel(
            sample_size=spatial_dim,  # 16x16
            in_channels=2,  # 1 for image, 1 for label conditioning
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 256, 512),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
            # Add group norm for better stability
            norm_num_groups=32,
        )

    def forward(self, x, timesteps, labels=None):
        batch_size = x.shape[0]
        
        # Process the label embeddings
        if labels is not None:
            label_emb = self.label_embed(labels)
            label_emb = label_emb.view(batch_size, 1, spatial_dim, spatial_dim)
            
            # Concatenate the label embedding along the channel dimension
            x = torch.cat([x, label_emb], dim=1)  # [batch_size, 2, 16, 16]
        else:
            # If no labels provided, concatenate zeros
            zeros = torch.zeros_like(x)
            x = torch.cat([x, zeros], dim=1)  # [batch_size, 2, 16, 16]
        
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