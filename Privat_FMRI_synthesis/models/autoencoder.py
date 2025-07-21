import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """3D Residual block with improved normalization and skip connections"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # First convolution path
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.act1 = nn.SiLU()
        
        # Second convolution path
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        
        # Residual connection (with projection if needed)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
            )
    
    def forward(self, x):
        # Main path
        identity = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        # Add skip connection
        out += self.skip(identity)
        return F.silu(out)


class DownBlock3D(nn.Module):
    """Downsampling block with residual connections"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block = ResidualBlock3D(in_channels, out_channels, stride=2)
    
    def forward(self, x):
        return self.res_block(x)


class UpBlock3D(nn.Module):
    """Upsampling block with residual connections and skip input handling"""
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        # Account for skip connection input
        self.has_skip = skip_channels > 0
        total_in_channels = in_channels + skip_channels if self.has_skip else in_channels
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        # Residual block after combining with skip connection
        self.res_block = ResidualBlock3D(total_in_channels, out_channels)
    
    def forward(self, x, skip=None):
        # Upsample first
        x = self.upsample(x)
        
        # Concatenate with skip connection if available
        if self.has_skip and skip is not None:
            # Make sure spatial dimensions match exactly
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            
        # Apply residual block
        return self.res_block(x)


class VectorQuantizer(nn.Module):
    """Vector Quantizer for VQ-VAE"""
    def __init__(self, num_embeddings=512, embedding_dim=8, commitment_cost=0.25):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Create embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z):
        # Reshape z -> (batch, embedding_dim, d, h, w) to (batch*d*h*w, embedding_dim)
        batch_size, emb_dim, d, h, w = z.shape
        z_flattened = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flattened = z_flattened.view(-1, emb_dim)
        
        # Calculate distances from embeddings
        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight ** 2, dim=1) - \
                   2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        # Find nearest embedding indices
        min_encoding_indices = torch.argmin(distances, dim=1)
        
        # Get quantized latent vectors
        z_q = self.embedding(min_encoding_indices).view(batch_size, d, h, w, emb_dim)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        
        # Compute loss
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()  # Pass gradient to encoder
        
        return z_q, vq_loss, min_encoding_indices


class Improved3DAutoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=8, base_channels=32, use_vq=True, num_vq_embeddings=512):
        """
        Improved 3D autoencoder with residual connections, VQ-VAE, and deeper architecture
        
        Args:
            in_channels: Number of input channels (1 for grayscale fMRI)
            latent_channels: Number of channels in the latent space
            base_channels: Base number of channels (will be multiplied in deeper layers)
            use_vq: Whether to use Vector Quantization in the latent space
            num_vq_embeddings: Number of embeddings in the VQ codebook
        """
        super().__init__()
        
        # Initial projection to base_channels
        self.initial_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder blocks
        self.enc1 = ResidualBlock3D(base_channels, base_channels)
        self.enc2 = DownBlock3D(base_channels, base_channels*2)  # 1/2
        self.enc3 = DownBlock3D(base_channels*2, base_channels*4)  # 1/4
        self.enc4 = DownBlock3D(base_channels*4, base_channels*8)  # 1/8
        
        # Bottleneck
        self.bottleneck = ResidualBlock3D(base_channels*8, latent_channels)
        
        # Vector Quantizer
        self.use_vq = use_vq
        if use_vq:
            self.vq = VectorQuantizer(
                num_embeddings=num_vq_embeddings,
                embedding_dim=latent_channels,
                commitment_cost=0.25
            )
        
        # Decoder blocks
        self.dec4 = ResidualBlock3D(latent_channels, base_channels*8)
        self.dec3 = UpBlock3D(base_channels*8, base_channels*4, skip_channels=base_channels*8)  # 1/4
        self.dec2 = UpBlock3D(base_channels*4, base_channels*2, skip_channels=base_channels*4)  # 1/2
        self.dec1 = UpBlock3D(base_channels*2, base_channels, skip_channels=base_channels*2)  # 1/1
        
        # Additional upsampling to get back to original resolution (if needed)
        self.final_upsample = UpBlock3D(base_channels, base_channels, skip_channels=base_channels)
        
        # Final convolution
        self.final = nn.Sequential(
            ResidualBlock3D(base_channels, base_channels),
            nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent representation"""
        # Initial projection
        x = self.initial_conv(x)
        
        # Store intermediate features for skip connections
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        
        # Bottleneck
        z = self.bottleneck(enc4_out)
        
        if self.use_vq:
            z_q, vq_loss, _ = self.vq(z)
            return z_q, vq_loss, [enc1_out, enc2_out, enc3_out, enc4_out]
        else:
            return z, None, [enc1_out, enc2_out, enc3_out, enc4_out]
    
    def decode(self, z, skip_features=None):
        """Decode from latent representation"""
        # Initial decoding
        dec4_out = self.dec4(z)
        
        # Use skip connections if available
        if skip_features is not None:
            enc1_out, enc2_out, enc3_out, enc4_out = skip_features
            
            dec3_out = self.dec3(dec4_out, enc4_out)
            dec2_out = self.dec2(dec3_out, enc3_out)
            dec1_out = self.dec1(dec2_out, enc2_out)
            
            # Additional upsampling to match input dimensions
            final_dec = self.final_upsample(dec1_out, enc1_out)
        else:
            dec3_out = self.dec3(dec4_out, None)
            dec2_out = self.dec2(dec3_out, None)
            dec1_out = self.dec1(dec2_out, None)
            
            # Additional upsampling without skip connection
            final_dec = self.final_upsample(dec1_out, None)
        
        # Final convolution
        out = self.final(final_dec)
        return out
    
    def forward(self, x):
        # Encode
        if self.use_vq:
            z, vq_loss, skip_features = self.encode(x)
        else:
            z, _, skip_features = self.encode(x)
            vq_loss = None
        
        # Decode
        recon = self.decode(z, skip_features)
        
        if self.use_vq:
            return recon, z, vq_loss
        else:
            return recon, z


class SuperResolution3DAutoencoder(nn.Module):
    """Improved super-resolution model with latent conditioning"""
    def __init__(self, in_channels=1, latent_channels=8, base_channels=32, 
                 scale_factor=2, use_vq=True, num_vq_embeddings=512):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Base autoencoder
        self.autoencoder = Improved3DAutoencoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            use_vq=use_vq,
            num_vq_embeddings=num_vq_embeddings
        )
        
        # Enhanced super-resolution module that uses both latent and low-res reconstruction
        self.sr_module = nn.Module()
        
        # Feature extraction from low-res reconstruction
        self.sr_module.extract = ResidualBlock3D(in_channels, base_channels)
        
        # Feature extraction from latent representation
        self.sr_module.latent_process = nn.Sequential(
            nn.Conv3d(latent_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, base_channels), num_channels=base_channels),
            nn.SiLU(),
            nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False)
        )
        
        # Combined upsampling module
        self.sr_module.upsample = nn.Sequential(
            ResidualBlock3D(base_channels*2, base_channels*4),
            nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False),
            ResidualBlock3D(base_channels*4, base_channels*2),
            ResidualBlock3D(base_channels*2, base_channels),
            nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # First get autoencoder output
        if self.autoencoder.use_vq:
            recon, latent, vq_loss = self.autoencoder(x)
        else:
            recon, latent = self.autoencoder(x)
            vq_loss = None
        
        # Process low-res reconstruction
        recon_features = self.sr_module.extract(recon)
        
        # Process latent representation
        latent_features = self.sr_module.latent_process(latent)
        
        # Combine features and generate super-resolution output
        combined = torch.cat([recon_features, latent_features], dim=1)
        sr_output = self.sr_module.upsample(combined)
        
        # Return super-resolution output, low-res reconstruction, and latent
        if self.autoencoder.use_vq:
            return sr_output, recon, latent, vq_loss
        else:
            return sr_output, recon, latent