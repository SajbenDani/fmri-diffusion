# models/autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ResidualBlock3D and DownBlock3D remain the same ---
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.skip(identity)
        return F.silu(out)

class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block = ResidualBlock3D(in_channels, out_channels, stride=2)
    def forward(self, x):
        return self.res_block(x)

# --- THE KEY MODIFICATION IS HERE ---
class UpBlock3D(nn.Module):
    """Upsampling block with a switchable path for skip connections."""
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        # Path for when skip connection IS provided
        self.res_block_with_skip = ResidualBlock3D(in_channels + skip_channels, out_channels)
        
        # Path for when skip connection is NOT provided (for generation)
        self.res_block_no_skip = ResidualBlock3D(in_channels, out_channels)
        
        self.has_skip_connection_input = skip_channels > 0

    def forward(self, x, skip=None):
        x = self.upsample(x)
        
        if self.has_skip_connection_input and skip is not None:
            # This is the path used during autoencoder training and reconstruction
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            return self.res_block_with_skip(x)
        else:
            # This is the path used during generation
            return self.res_block_no_skip(x)

# --- VectorQuantizer remains the same ---
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=8, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    def forward(self, z):
        batch_size, emb_dim, d, h, w = z.shape
        z_flattened = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, emb_dim)
        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight ** 2, dim=1) - \
                   2 * torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(min_encoding_indices).view(batch_size, d, h, w, emb_dim)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        z_q = z + (z_q - z).detach()
        return z_q, vq_loss, min_encoding_indices

class Improved3DAutoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=8, base_channels=32, use_vq=True, num_vq_embeddings=512):
        super().__init__()
        self.initial_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc1 = ResidualBlock3D(base_channels, base_channels)
        self.enc2 = DownBlock3D(base_channels, base_channels*2)
        self.enc3 = DownBlock3D(base_channels*2, base_channels*4)
        self.enc4 = DownBlock3D(base_channels*4, base_channels*8)
        self.bottleneck = ResidualBlock3D(base_channels*8, latent_channels)
        self.use_vq = use_vq
        if use_vq:
            self.vq = VectorQuantizer(num_embeddings=num_vq_embeddings, embedding_dim=latent_channels)
        
        self.dec4 = ResidualBlock3D(latent_channels, base_channels*8)
        self.dec3 = UpBlock3D(base_channels*8, base_channels*4, skip_channels=base_channels*8)
        self.dec2 = UpBlock3D(base_channels*4, base_channels*2, skip_channels=base_channels*4)
        self.dec1 = UpBlock3D(base_channels*2, base_channels, skip_channels=base_channels*2)
        self.final_upsample = UpBlock3D(base_channels, base_channels, skip_channels=base_channels)
        self.final = nn.Sequential(
            ResidualBlock3D(base_channels, base_channels),
            nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.initial_conv(x)
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        z = self.bottleneck(enc4_out)
        
        if self.use_vq:
            z_q, vq_loss, _ = self.vq(z)
            return z_q, vq_loss, [enc1_out, enc2_out, enc3_out, enc4_out]
        else:
            return z, None, [enc1_out, enc2_out, enc3_out, enc4_out]
    
    def decode(self, z, skip_features=None):
        """Decode from latent. Can now handle skip_features being None."""
        dec4_out = self.dec4(z)
        
        # This logic remains the same, but the UpBlocks will now handle None correctly
        if skip_features is not None and len(skip_features) == 4:
            enc1_out, enc2_out, enc3_out, enc4_out = skip_features
            dec3_out = self.dec3(dec4_out, enc4_out)
            dec2_out = self.dec2(dec3_out, enc3_out)
            dec1_out = self.dec1(dec2_out, enc2_out)
            final_dec = self.final_upsample(dec1_out, enc1_out)
        else: # Generation path
            dec3_out = self.dec3(dec4_out, None)
            dec2_out = self.dec2(dec3_out, None)
            dec1_out = self.dec1(dec2_out, None)
            final_dec = self.final_upsample(dec1_out, None)
        
        out = self.final(final_dec)
        return out
    
    def forward(self, x):
        if self.use_vq:
            z, vq_loss, skip_features = self.encode(x)
        else:
            z, _, skip_features = self.encode(x)
            vq_loss = None
        
        recon = self.decode(z, skip_features)
        
        if self.use_vq:
            return recon, z, vq_loss
        else:
            return recon, z