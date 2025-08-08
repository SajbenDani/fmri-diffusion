#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding3D(nn.Module):
    """Adds 3D positional encoding to latent representations to help preserve spatial information."""
    def __init__(self, channels, dropout=0.0, max_len=32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding for 3D volumes - simplified version
        pe = torch.zeros(1, channels, max_len, max_len, max_len)
        
        # Calculate position indices for each dimension
        d_pos = torch.arange(0, max_len).float()
        h_pos = torch.arange(0, max_len).float()
        w_pos = torch.arange(0, max_len).float()
        
        # Calculate division term for frequency
        div_term = torch.exp(torch.arange(0, channels, 6).float() * (-math.log(10000.0) / channels))
        
        # Create positional encoding for each spatial dimension
        for i in range(0, channels, 6):
            if i < channels:
                # Depth dimension
                pe_d = torch.zeros(max_len, max_len, max_len)
                for d in range(max_len):
                    pe_d[d, :, :] = torch.sin(d_pos[d] * div_term[i//6])
                pe[0, i, :, :, :] = pe_d
                
                if i+1 < channels:
                    pe_d = torch.zeros(max_len, max_len, max_len)
                    for d in range(max_len):
                        pe_d[d, :, :] = torch.cos(d_pos[d] * div_term[i//6])
                    pe[0, i+1, :, :, :] = pe_d
            
            if i+2 < channels:
                # Height dimension
                pe_h = torch.zeros(max_len, max_len, max_len)
                for h in range(max_len):
                    pe_h[:, h, :] = torch.sin(h_pos[h] * div_term[i//6])
                pe[0, i+2, :, :, :] = pe_h
                
                if i+3 < channels:
                    pe_h = torch.zeros(max_len, max_len, max_len)
                    for h in range(max_len):
                        pe_h[:, h, :] = torch.cos(h_pos[h] * div_term[i//6])
                    pe[0, i+3, :, :, :] = pe_h
            
            if i+4 < channels:
                # Width dimension
                pe_w = torch.zeros(max_len, max_len, max_len)
                for w in range(max_len):
                    pe_w[:, :, w] = torch.sin(w_pos[w] * div_term[i//6])
                pe[0, i+4, :, :, :] = pe_w
                
                if i+5 < channels:
                    pe_w = torch.zeros(max_len, max_len, max_len)
                    for w in range(max_len):
                        pe_w[:, :, w] = torch.cos(w_pos[w] * div_term[i//6])
                    pe[0, i+5, :, :, :] = pe_w
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to the input tensor."""
        d, h, w = x.size(2), x.size(3), x.size(4)
        # Slice the positional encoding to match the input dimensions
        positional_encoding = self.pe[:, :, :d, :h, :w]
        x = x + positional_encoding
        return self.dropout(x)

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

class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.res_block = ResidualBlock3D(in_channels, out_channels)
        
    def forward(self, x):
        x = self.upsample(x)
        return self.res_block(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=16, commitment_cost=0.25):
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
        z_q = z + (z_q - z).detach()  # Straight-through estimator
        return z_q, vq_loss, min_encoding_indices

class NoSkipAutoencoder(nn.Module):
    """3D Autoencoder without skip connections, with larger latent space and positional encoding."""
    def __init__(self, in_channels=1, latent_channels=16, base_channels=32, 
                 use_vq=True, num_vq_embeddings=1024, use_positional_encoding=True):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.use_vq = use_vq
        self.use_positional_encoding = use_positional_encoding
        
        # Encoder
        self.initial_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc1 = ResidualBlock3D(base_channels, base_channels)
        self.enc2 = DownBlock3D(base_channels, base_channels*2)
        self.enc3 = DownBlock3D(base_channels*2, base_channels*4)
        self.enc4 = DownBlock3D(base_channels*4, base_channels*8)
        
        # Enhanced bottleneck with more channels
        self.bottleneck = nn.Sequential(
            ResidualBlock3D(base_channels*8, base_channels*8),
            ResidualBlock3D(base_channels*8, latent_channels)
        )
        
        # Vector Quantization
        if use_vq:
            self.vq = VectorQuantizer(num_embeddings=num_vq_embeddings, embedding_dim=latent_channels)
        
        # Positional Encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding3D(latent_channels, dropout=0.0, max_len=32)
        
        # Decoder (without skip connections)
        self.dec4 = ResidualBlock3D(latent_channels, base_channels*8)
        self.dec3 = UpBlock3D(base_channels*8, base_channels*4)
        self.dec2 = UpBlock3D(base_channels*4, base_channels*2)
        self.dec1 = UpBlock3D(base_channels*2, base_channels)
        self.final_upsample = UpBlock3D(base_channels, base_channels)
        
        # Final convolutional layers
        self.final = nn.Sequential(
            ResidualBlock3D(base_channels, base_channels),
            nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent representation."""
        x = self.initial_conv(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        z = self.bottleneck(x)
        
        if self.use_vq:
            z_q, vq_loss, _ = self.vq(z)
            
            # Apply positional encoding after VQ if enabled
            if self.use_positional_encoding:
                z_q = self.pos_encoding(z_q)
                
            return z_q, vq_loss
        else:
            # Apply positional encoding if enabled
            if self.use_positional_encoding:
                z = self.pos_encoding(z)
                
            return z, None
    
    def decode(self, z):
        """Decode from latent representation without skip connections."""
        x = self.dec4(z)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.final_upsample(x)
        x = self.final(x)
        return x
    
    def forward(self, x):
        """Forward pass through the autoencoder."""
        if self.use_vq:
            z, vq_loss = self.encode(x)
            recon = self.decode(z)
            return recon, z, vq_loss
        else:
            z, _ = self.encode(x)
            recon = self.decode(z)
            return recon, z