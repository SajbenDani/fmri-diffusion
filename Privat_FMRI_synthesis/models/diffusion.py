from diffusers import UNet2DModel
from config import DEVICE

# Our autoencoder produces a latent vector of size 256.
# We reshape that vector into a (1, 16, 16) "image" because 16 x 16 = 256.
latent_dim = 256
spatial_dim = int(latent_dim ** 0.5)  # 16

# Specify down and up block types explicitly so their length matches block_out_channels (3 blocks)
diffusion_model = UNet2DModel(
    sample_size=spatial_dim,  # This is now 16
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 256, 512),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
).to(DEVICE)
