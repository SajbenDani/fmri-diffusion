import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock3D(nn.Module):
    """3D Residual block with time embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_channels), out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None
    
    def forward(self, x, t_emb=None):
        identity = self.skip(x)
        
        x = F.silu(self.norm1(self.conv1(x)))
        
        # Add time embedding
        if self.time_mlp and t_emb is not None:
            time_cond = self.time_mlp(t_emb)
            x = x + time_cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        x = self.norm2(self.conv2(x))
        return F.silu(x + identity)

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Positional Embedding for time steps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionUNet3D(nn.Module):
    """3D U-Net for Latent Diffusion"""
    def __init__(self, latent_channels=8, base_channels=128, time_emb_dim=256):
        super().__init__()
        
        # Conditioning input: z_hr_noisy + z_lr_upsampled
        in_channels = latent_channels * 2 
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # --- Encoder ---
        self.init_conv = ResidualBlock3D(in_channels, base_channels, time_emb_dim)
        self.down1 = nn.Sequential(
            ResidualBlock3D(base_channels, base_channels, time_emb_dim),
            nn.Conv3d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        )
        self.down2 = nn.Sequential(
            ResidualBlock3D(base_channels, base_channels*2, time_emb_dim),
            nn.Conv3d(base_channels*2, base_channels*2, kernel_size=4, stride=2, padding=1)
        )
        
        # --- Bottleneck ---
        self.mid = nn.Sequential(
            ResidualBlock3D(base_channels*2, base_channels*4, time_emb_dim),
            ResidualBlock3D(base_channels*4, base_channels*2, time_emb_dim)
        )
        
        # --- Decoder ---
        # Separate convolution and residual blocks for proper skip connections
        self.up1_conv = nn.ConvTranspose3d(base_channels*2, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.up1_block = ResidualBlock3D(base_channels*2 + base_channels*2, base_channels, time_emb_dim)
        
        self.up2_conv = nn.ConvTranspose3d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.up2_block = ResidualBlock3D(base_channels + base_channels, base_channels, time_emb_dim)
        
        # Final blocks
        self.final_res = ResidualBlock3D(base_channels + base_channels, base_channels, time_emb_dim)
        self.final_conv = nn.Conv3d(base_channels, latent_channels, kernel_size=1)
    
    def forward(self, z_hr_noisy, t, z_lr_upsampled):
        # 1. Concatenate inputs
        x = torch.cat([z_hr_noisy, z_lr_upsampled], dim=1)
        
        # 2. Get time embedding
        t_emb = self.time_mlp(t)
        
        # 3. Downsampling path with shape tracking
        s1 = self.init_conv(x, t_emb)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        
        # 4. Bottleneck
        x = self.mid(s3)
        
        # 5. Upsampling path with explicit size matching for skip connections
        # First upsampling block
        x = self.up1_conv(x)
        
        # Handle potential size mismatch with interpolation
        if x.shape[2:] != s3.shape[2:]:
            x = F.interpolate(x, size=s3.shape[2:], mode='trilinear', align_corners=False)
            
        x = torch.cat([x, s3], dim=1)
        x = self.up1_block(x, t_emb)
        
        # Second upsampling block
        x = self.up2_conv(x)
        
        # Handle potential size mismatch with interpolation
        if x.shape[2:] != s2.shape[2:]:
            x = F.interpolate(x, size=s2.shape[2:], mode='trilinear', align_corners=False)
            
        x = torch.cat([x, s2], dim=1)
        x = self.up2_block(x, t_emb)
        
        # 6. Final projection with last skip connection
        # Handle potential size mismatch with interpolation
        if x.shape[2:] != s1.shape[2:]:
            x = F.interpolate(x, size=s1.shape[2:], mode='trilinear', align_corners=False)
            
        x = torch.cat([x, s1], dim=1)
        x = self.final_res(x, t_emb)
        x = self.final_conv(x)
        
        return x