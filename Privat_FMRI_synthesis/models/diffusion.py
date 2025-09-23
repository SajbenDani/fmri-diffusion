"""
3D Diffusion Models for fMRI Latent Space Generation.

This module implements sophisticated 3D diffusion models specifically designed for
generating high-quality functional Magnetic Resonance Imaging (fMRI) data in latent space.
The models work in conjunction with the autoencoder to enable efficient and realistic
brain activation pattern synthesis.

The diffusion approach follows the Denoising Diffusion Probabilistic Models (DDPM) paradigm,
adapted for 3D medical imaging data. The key innovation is operating in the compressed
latent space learned by the autoencoder, rather than directly on high-dimensional fMRI volumes.

Key Features:
    - 3D U-Net architecture with attention mechanisms for spatial coherence
    - Time-conditional generation with sinusoidal position embeddings
    - Residual connections and group normalization for stable training
    - Multi-scale feature processing through encoder-decoder structure
    - Designed for latent space operation (efficient memory usage)

Architecture Components:
    - ResidualBlock3D: Time-conditional residual blocks with normalization
    - SinusoidalPosEmb: Positional encoding for diffusion timesteps
    - DiffusionUNet3D: Complete U-Net for noise prediction in latent space

Theoretical Foundation:
    The diffusion process gradually adds Gaussian noise to clean latent codes over T timesteps:
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
    
    The model learns to reverse this process by predicting the noise:
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

Usage Pipeline:
    1. Encode fMRI data to latent space using autoencoder
    2. Train diffusion model to generate latent codes
    3. Sample new latent codes using trained diffusion model
    4. Decode latent codes back to fMRI space using autoencoder decoder

Example:
    ```python
    # Initialize diffusion model for latent space with 8 channels
    diffusion = DiffusionUNet3D(latent_channels=8, base_channels=128)
    
    # During training: predict noise for denoising
    noise_pred = diffusion(noisy_latent_hr, timesteps, latent_lr_upsampled)
    
    # During inference: iteratively denoise random noise
    sample = torch.randn(1, 8, 11, 13, 11)  # Random latent noise
    for t in reversed(range(num_timesteps)):
        noise_pred = diffusion(sample, t, condition)
        sample = scheduler.step(noise_pred, t, sample).prev_sample
    ```

Performance Considerations:
    - Operates on compressed latent space (~8 channels vs 1 input channel)
    - Memory efficient compared to pixel-space diffusion
    - Faster training and inference due to reduced spatial dimensions
    - Maintains spatial coherence through 3D convolutions and attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# =============================================================================
# BUILDING BLOCKS FOR 3D DIFFUSION
# =============================================================================

class ResidualBlock3D(nn.Module):
    """
    3D Residual Block with Time Embedding for Diffusion Models.
    
    This block is specifically designed for diffusion models, incorporating
    time step information into the feature processing. The time embedding
    allows the model to adapt its behavior based on the current noise level
    in the diffusion process.
    
    Architecture:
        Input + Time -> Conv3D -> GroupNorm -> SiLU -> Conv3D -> GroupNorm -> + -> SiLU -> Output
                 \                                                              /
                  \-----------> Skip Connection + Time Conditioning ----------/
    
    Time Conditioning Strategy:
        The time embedding is processed through an MLP and added as a bias term
        to the feature maps after the first convolution. This allows the network
        to modulate its processing based on the current diffusion timestep.
    
    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels
        time_emb_dim (int, optional): Dimension of time embedding. If None, no time conditioning
        
    Input Shapes:
        x: (batch_size, in_channels, depth, height, width)
        t_emb: (batch_size, time_emb_dim) if time conditioning enabled
        
    Output Shape:
        (batch_size, out_channels, depth, height, width)
        
    Design Choices:
        - GroupNorm for normalization (better with small batches than BatchNorm)
        - SiLU activation for smooth gradients (important for diffusion training)
        - Additive time conditioning (simple but effective)
        - Skip connection with proper channel matching
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: Optional[int] = None):
        super().__init__()
        
        # Main convolutional path
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_channels), out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        
        # Skip connection handling for channel dimension changes
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Time embedding processing (if enabled)
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None
    
    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional time conditioning.
        
        Args:
            x (torch.Tensor): Input feature tensor
            t_emb (torch.Tensor, optional): Time embedding tensor
            
        Returns:
            torch.Tensor: Output feature tensor with residual connection
        """
        identity = self.skip(x)
        
        # First convolution and normalization
        x = F.silu(self.norm1(self.conv1(x)))
        
        # Add time conditioning if available
        if self.time_mlp is not None and t_emb is not None:
            # Process time embedding and add as bias
            time_cond = self.time_mlp(t_emb)
            # Reshape to broadcast across spatial dimensions: (B, C) -> (B, C, 1, 1, 1)
            x = x + time_cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Second convolution and normalization
        x = self.norm2(self.conv2(x))
        
        # Add skip connection and final activation
        return F.silu(x + identity)


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Positional Embedding for Diffusion Timesteps.
    
    This module generates rich positional encodings for diffusion timesteps using
    sinusoidal functions of different frequencies. The encoding allows the model
    to understand the current position in the diffusion process and adapt its
    denoising behavior accordingly.
    
    The encoding is inspired by the Transformer positional encoding but adapted
    for the temporal dimension of diffusion processes. Different frequency components
    help the model distinguish between different noise levels and apply appropriate
    denoising strategies.
    
    Mathematical Formulation:
        For position t and dimension i:
        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
        where d is the embedding dimension
    
    Args:
        dim (int): Dimension of the output embedding
        
    Input Shape:
        (batch_size,) - timestep values
        
    Output Shape:
        (batch_size, dim) - positional embeddings
        
    Properties:
        - Deterministic: same timestep always produces same embedding
        - Smooth: nearby timesteps have similar embeddings
        - Rich: captures both low and high frequency temporal patterns
        - Learnable: can be combined with learned transformations
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal positional embeddings for given timesteps.
        
        Args:
            t (torch.Tensor): Timestep values of shape (batch_size,)
            
        Returns:
            torch.Tensor: Positional embeddings of shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Create frequency scaling factors
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Apply frequencies to timesteps
        emb = t[:, None] * emb[None, :]
        
        # Combine sine and cosine components
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# =============================================================================
# MAIN DIFFUSION U-NET ARCHITECTURE
# =============================================================================

class DiffusionUNet3D(nn.Module):
    """
    3D U-Net for Latent Diffusion Model - Advanced Architecture for fMRI Generation.
    
    This is the core architecture for the diffusion model, implementing a sophisticated
    3D U-Net that operates in the latent space learned by the autoencoder. The model
    is specifically designed for super-resolution tasks, taking both noisy high-resolution
    latent codes and upsampled low-resolution conditions as input.
    
    Key Architectural Features:
        - Dual-input design: noisy HR latent + LR condition
        - Time-conditional processing throughout the network
        - Multi-scale feature extraction with skip connections
        - Group normalization for stable training with small batches
        - Residual connections for deep network training stability
        
    Super-Resolution Conditioning:
        The model implements a conditioning strategy where low-resolution fMRI data
        is upsampled and concatenated with the noisy high-resolution latent code.
        This provides spatial guidance for the denoising process while maintaining
        the flexibility of the diffusion framework.
    
    Architecture Overview:
        Input: [z_hr_noisy, z_lr_upsampled] -> Concat -> Encoder -> Bottleneck -> Decoder -> Noise Prediction
        
        Encoder Path:
            - Initial convolution and residual processing
            - Progressive downsampling (2 stages)
            - Feature channel expansion at each stage
            
        Bottleneck:
            - High-capacity feature processing
            - Maximum compression point
            
        Decoder Path:
            - Progressive upsampling with skip connections
            - Feature channel reduction
            - Skip connections from corresponding encoder stages
    
    Args:
        latent_channels (int): Number of channels in the latent space (default: 8)
        base_channels (int): Base number of feature channels, scales throughout network (default: 128)
        time_emb_dim (int): Dimension of time embeddings (default: 256)
        
    Input Shapes:
        z_hr_noisy: (batch_size, latent_channels, depth, height, width) - Noisy HR latent
        t: (batch_size,) - Timestep values
        z_lr_upsampled: (batch_size, latent_channels, depth, height, width) - Upsampled LR condition
        
    Output Shape:
        (batch_size, latent_channels, depth, height, width) - Predicted noise
        
    Training Strategy:
        During training, the model learns to predict the noise that was added to clean
        latent codes. The loss is typically L2 between predicted and actual noise:
        L = ||ε - ε_θ(z_t, t, c)||²
        where ε is actual noise, ε_θ is predicted noise, z_t is noisy latent, t is timestep, c is condition
    
    Usage Example:
        ```python
        model = DiffusionUNet3D(latent_channels=8, base_channels=128)
        
        # During training
        noise_pred = model(noisy_hr_latent, timesteps, lr_condition)
        loss = F.mse_loss(noise_pred, actual_noise)
        
        # During inference (with scheduler)
        for t in scheduler.timesteps:
            noise_pred = model(current_sample, t, condition)
            current_sample = scheduler.step(noise_pred, t, current_sample).prev_sample
        ```
    """
    
    def __init__(self, latent_channels: int = 8, base_channels: int = 128, time_emb_dim: int = 256):
        super().__init__()
        
        # Store configuration
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        
        # Input processing: concatenated noisy HR + upsampled LR
        in_channels = latent_channels * 2
        
        # =============================================================================
        # TIME EMBEDDING NETWORK
        # =============================================================================
        
        # Process timestep information into rich embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),        # Convert timestep to sinusoidal encoding
            nn.Linear(time_emb_dim, time_emb_dim), # Learn from positional encoding
            nn.SiLU()                              # Smooth activation
        )
        
        # =============================================================================
        # ENCODER PATH (DOWNSAMPLING)
        # =============================================================================
        
        # Initial feature extraction from concatenated inputs
        self.init_conv = ResidualBlock3D(in_channels, base_channels, time_emb_dim)
        
        # Progressive downsampling with feature expansion
        # Stage 1: Maintain spatial resolution, establish features
        self.down1 = nn.Sequential(
            ResidualBlock3D(base_channels, base_channels, time_emb_dim),
            nn.Conv3d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)  # /2 spatial
        )
        
        # Stage 2: Further compression, double features
        self.down2 = nn.Sequential(
            ResidualBlock3D(base_channels, base_channels*2, time_emb_dim),
            nn.Conv3d(base_channels*2, base_channels*2, kernel_size=4, stride=2, padding=1)  # /4 spatial
        )
        
        # =============================================================================
        # BOTTLENECK (MAXIMUM COMPRESSION)
        # =============================================================================
        
        # High-capacity processing at maximum compression
        self.mid = nn.Sequential(
            ResidualBlock3D(base_channels*2, base_channels*4, time_emb_dim),  # Expand features
            ResidualBlock3D(base_channels*4, base_channels*2, time_emb_dim)   # Contract back
        )
        
        # =============================================================================
        # DECODER PATH (UPSAMPLING WITH SKIP CONNECTIONS)
        # =============================================================================
        
        # Progressive upsampling with skip connections from encoder
        # Each stage consists of: transpose conv -> concatenate skip -> residual block
        
        # Stage 1: First upsampling
        self.up1_conv = nn.ConvTranspose3d(base_channels*2, base_channels*2, 
                                          kernel_size=4, stride=2, padding=1)
        self.up1_block = ResidualBlock3D(base_channels*2 + base_channels*2,  # Input + skip
                                        base_channels, time_emb_dim)
        
        # Stage 2: Second upsampling  
        self.up2_conv = nn.ConvTranspose3d(base_channels, base_channels,
                                          kernel_size=4, stride=2, padding=1)
        self.up2_block = ResidualBlock3D(base_channels + base_channels,  # Input + skip
                                        base_channels, time_emb_dim)
        
        # =============================================================================
        # OUTPUT PROCESSING
        # =============================================================================
        
        # Final processing with last skip connection
        self.final_res = ResidualBlock3D(base_channels + base_channels,  # Input + skip
                                        base_channels, time_emb_dim)
        
        # Output projection to noise prediction
        self.final_conv = nn.Conv3d(base_channels, latent_channels, kernel_size=1)
    
    def forward(self, z_hr_noisy: torch.Tensor, t: torch.Tensor, 
                z_lr_upsampled: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for noise prediction in latent diffusion.
        
        This method implements the complete forward pass of the U-Net, processing
        the concatenated inputs through the encoder-decoder structure with time
        conditioning and skip connections.
        
        Args:
            z_hr_noisy (torch.Tensor): Noisy high-resolution latent code 
                Shape: (batch_size, latent_channels, depth, height, width)
            t (torch.Tensor): Diffusion timesteps
                Shape: (batch_size,)
            z_lr_upsampled (torch.Tensor): Upsampled low-resolution condition
                Shape: (batch_size, latent_channels, depth, height, width)
                
        Returns:
            torch.Tensor: Predicted noise tensor
                Shape: (batch_size, latent_channels, depth, height, width)
                
        Processing Flow:
            1. Concatenate inputs along channel dimension
            2. Generate time embeddings
            3. Forward pass through encoder with feature storage
            4. Process through bottleneck
            5. Forward pass through decoder with skip connections
            6. Output noise prediction
        
        Skip Connection Strategy:
            The decoder uses skip connections from corresponding encoder stages:
            - up2_block receives features from down1 (s2)
            - up1_block receives features from down2 (s3)  
            - final_res receives features from init_conv (s1)
            
            This preserves both local details and global context for accurate
            noise prediction across multiple scales.
        """
        # =============================================================================
        # INPUT PROCESSING AND TIME EMBEDDING
        # =============================================================================
        
        # Concatenate noisy HR latent with upsampled LR condition
        # This provides both the signal to denoise and the conditioning information
        x = torch.cat([z_hr_noisy, z_lr_upsampled], dim=1)  # (B, 2*latent_channels, D, H, W)
        
        # Generate rich time embeddings for current diffusion timestep
        t_emb = self.time_mlp(t)  # (B, time_emb_dim)
        
        # =============================================================================
        # ENCODER PATH WITH SKIP FEATURE STORAGE
        # =============================================================================
        
        # Initial feature extraction - stored for final skip connection
        s1 = self.init_conv(x, t_emb)  # (B, base_channels, D, H, W) 
        
        # Progressive downsampling with feature storage for skip connections
        s2 = self.down1[0](s1, t_emb)                    # ResidualBlock processing
        s2 = self.down1[1](s2)                           # Spatial downsampling -> (B, base_channels, D/2, H/2, W/2)
        
        s3 = self.down2[0](s2, t_emb)                    # ResidualBlock processing  
        s3 = self.down2[1](s3)                           # Spatial downsampling -> (B, base_channels*2, D/4, H/4, W/4)
        
        # =============================================================================
        # BOTTLENECK PROCESSING
        # =============================================================================
        
        # Maximum compression processing with high-capacity features
        x = self.mid[0](s3, t_emb)  # Expand to base_channels*4
        x = self.mid[1](x, t_emb)   # Contract back to base_channels*2
        
        # =============================================================================
        # DECODER PATH WITH SKIP CONNECTIONS
        # =============================================================================
        
        # First upsampling stage
        x = self.up1_conv(x)  # Transpose convolution for spatial upsampling
        
        # Handle potential spatial mismatches with skip connection
        if x.shape[2:] != s3.shape[2:]:
            x = F.interpolate(x, size=s3.shape[2:], mode='trilinear', align_corners=False)
            
        # Concatenate with skip connection and process
        x = torch.cat([x, s3], dim=1)  # Combine upsampled features with encoder features
        x = self.up1_block(x, t_emb)   # Process combined features
        
        # Second upsampling stage
        x = self.up2_conv(x)  # Transpose convolution for spatial upsampling
        
        # Handle potential spatial mismatches with skip connection
        if x.shape[2:] != s2.shape[2:]:
            x = F.interpolate(x, size=s2.shape[2:], mode='trilinear', align_corners=False)
            
        # Concatenate with skip connection and process
        x = torch.cat([x, s2], dim=1)  # Combine upsampled features with encoder features
        x = self.up2_block(x, t_emb)   # Process combined features
        
        # =============================================================================
        # FINAL PROCESSING AND OUTPUT
        # =============================================================================
        
        # Final upsampling and skip connection with initial features
        if x.shape[2:] != s1.shape[2:]:
            x = F.interpolate(x, size=s1.shape[2:], mode='trilinear', align_corners=False)
            
        # Final feature combination and processing
        x = torch.cat([x, s1], dim=1)    # Combine with initial features
        x = self.final_res(x, t_emb)     # Final residual processing
        
        # Project to noise prediction
        x = self.final_conv(x)  # (B, latent_channels, D, H, W)
        
        return x