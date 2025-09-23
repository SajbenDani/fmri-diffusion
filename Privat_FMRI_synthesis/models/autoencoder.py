"""
3D Autoencoder Models for fMRI Latent Space Learning.

This module implements sophisticated 3D autoencoder architectures specifically designed
for functional Magnetic Resonance Imaging (fMRI) data processing. The autoencoders
learn compact latent representations of brain activation patterns, enabling efficient
downstream processing with diffusion models.

Key Features:
    - 3D Convolutional architectures optimized for brain volume data
    - Residual connections for stable training and gradient flow
    - Skip connections for preserving fine-grained spatial information
    - Vector Quantization (VQ) support for discrete latent representations
    - Flexible encode/decode paths supporting both reconstruction and generation

Architecture Components:
    - ResidualBlock3D: Building block with residual connections and normalization
    - DownBlock3D/UpBlock3D: Spatial downsampling/upsampling with feature learning
    - VectorQuantizer: Optional discrete latent space quantization
    - Improved3DAutoencoder: Complete autoencoder with skip connections

Input/Output Specifications:
    - Input: 3D fMRI volumes (typically 91x109x91 voxels, single channel)
    - Output: Reconstructed volumes with same spatial dimensions
    - Latent: Compressed representation (configurable dimensions)

Usage Examples:
    ```python
    # Standard autoencoder
    model = Improved3DAutoencoder(in_channels=1, latent_channels=8)
    
    # With vector quantization
    model = Improved3DAutoencoder(use_vq=True, num_vq_embeddings=512)
    
    # Training mode (with skip connections)
    recon, latent, vq_loss = model(fmri_volume)
    
    # Generation mode (without skip connections)
    generated = model.decode(sampled_latent, skip_features=None)
    ```

Paper References:
    - Based on VQ-VAE and modern autoencoder architectures
    - Adapted for 3D medical imaging and fMRI-specific requirements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ResidualBlock3D(nn.Module):
    """
    3D Residual Block with Group Normalization and SiLU activation.
    
    This is the fundamental building block of the autoencoder, providing stable
    gradient flow through residual connections and effective feature learning
    through 3D convolutions. The block is specifically designed for medical
    imaging data where spatial coherence is crucial.
    
    Architecture:
        Input -> Conv3D -> GroupNorm -> SiLU -> Conv3D -> GroupNorm -> + -> SiLU -> Output
                    \                                                  /
                     \-----------> Skip Connection -----------------/
    
    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels  
        kernel_size (int): Size of the 3D convolution kernels (default: 3)
        stride (int): Stride for the first convolution (default: 1)
        padding (int): Padding for convolutions (default: 1)
        
    Input Shape:
        (batch_size, in_channels, depth, height, width)
        
    Output Shape:
        (batch_size, out_channels, depth//stride, height//stride, width//stride)
        
    Design Choices:
        - GroupNorm instead of BatchNorm for better performance with small batches
        - SiLU activation for smooth gradients and better training dynamics
        - 1x1 skip connection when channel dimensions change
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        
        # First convolution path
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.act1 = nn.SiLU()
        
        # Second convolution path  
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        
        # Skip connection - handle channel/spatial dimension changes
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, D, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, D', H', W')
        """
        identity = x
        
        # Main path
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        # Add skip connection and final activation
        out += self.skip(identity)
        return F.silu(out)


class DownBlock3D(nn.Module):
    """
    3D Downsampling Block for encoder path.
    
    Reduces spatial dimensions by factor of 2 while increasing feature channels.
    This is used in the encoder to progressively abstract spatial information
    into higher-level feature representations.
    
    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels
        
    Input Shape:
        (batch_size, in_channels, depth, height, width)
        
    Output Shape: 
        (batch_size, out_channels, depth//2, height//2, width//2)
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.res_block = ResidualBlock3D(in_channels, out_channels, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through downsampling block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Downsampled tensor with increased channels
        """
        return self.res_block(x)

class UpBlock3D(nn.Module):
    """
    3D Upsampling Block with Switchable Skip Connections.
    
    This innovative upsampling block supports two operational modes:
    1. Reconstruction mode: Uses skip connections for high-fidelity reconstruction
    2. Generation mode: Operates without skip connections for novel sample generation
    
    This dual functionality is crucial for the autoencoder to serve both as a 
    reconstruction model (during training) and as a decoder for diffusion-generated
    latent codes (during inference).
    
    Architecture:
        Input -> Upsample -> [Concat with Skip] -> ResidualBlock -> Output
    
    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels
        skip_channels (int): Number of skip connection channels (0 = no skip support)
        
    Input Shape:
        x: (batch_size, in_channels, depth, height, width)
        skip: (batch_size, skip_channels, depth*2, height*2, width*2) or None
        
    Output Shape:
        (batch_size, out_channels, depth*2, height*2, width*2)
        
    Design Philosophy:
        The skip connection flexibility allows the same architecture to be used
        for both autoencoder training (with skip connections for precise reconstruction)
        and diffusion model decoding (without skip connections for creative generation).
    """
    
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        
        # Trilinear upsampling for smooth spatial interpolation
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        # Dual path architecture for skip connection flexibility
        self.res_block_with_skip = ResidualBlock3D(in_channels + skip_channels, out_channels)
        self.res_block_no_skip = ResidualBlock3D(in_channels, out_channels)
        
        # Track if this block was configured to expect skip connections
        self.has_skip_connection_input = skip_channels > 0

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional skip connection.
        
        The method automatically determines whether to use skip connections based on:
        1. Whether the block was configured to accept them (skip_channels > 0)
        2. Whether a skip tensor is actually provided
        
        Args:
            x (torch.Tensor): Input tensor to upsample
            skip (torch.Tensor, optional): Skip connection from encoder path
            
        Returns:
            torch.Tensor: Upsampled and processed tensor
            
        Note:
            Spatial dimension mismatches between x and skip are automatically
            resolved using trilinear interpolation to ensure proper concatenation.
        """
        # Upsample input to double spatial dimensions
        x = self.upsample(x)
        
        if self.has_skip_connection_input and skip is not None:
            # Reconstruction path: use skip connections for detailed reconstruction
            
            # Handle potential spatial dimension mismatches
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            # Concatenate along channel dimension and process
            x = torch.cat([x, skip], dim=1)
            return self.res_block_with_skip(x)
        else:
            # Generation path: pure upsampling without skip connections
            return self.res_block_no_skip(x)


# =============================================================================
# VECTOR QUANTIZATION
# =============================================================================

class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for discrete latent representations.
    
    This module implements the Vector Quantization technique from VQ-VAE,
    which maps continuous latent vectors to discrete codes from a learned
    codebook. This can be beneficial for:
    - More structured latent spaces
    - Better downstream generation quality
    - Interpretable latent representations
    
    The quantization process:
    1. Find nearest codebook vector for each spatial location
    2. Replace continuous values with discrete codes
    3. Use straight-through estimator for backpropagation
    4. Update codebook through commitment and codebook losses
    
    Args:
        num_embeddings (int): Size of the discrete codebook (default: 512)
        embedding_dim (int): Dimension of each codebook vector (default: 8)
        commitment_cost (float): Weight for commitment loss (default: 0.25)
        
    Input Shape:
        (batch_size, embedding_dim, depth, height, width)
        
    Output Shape:
        quantized: (batch_size, embedding_dim, depth, height, width)
        vq_loss: scalar tensor
        encoding_indices: (batch_size * depth * height * width,)
        
    Loss Components:
        - Codebook loss: ||sg[z] - e||^2 (update codebook toward encoder outputs)
        - Commitment loss: ||z - sg[e]||^2 (encourage encoder to commit to codes)
        where sg[] denotes stop-gradient operation
    """
    
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 8, 
                 commitment_cost: float = 0.25):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Learnable codebook of discrete vectors
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Initialize codebook with uniform distribution
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous latent vectors to discrete codebook entries.
        
        Args:
            z (torch.Tensor): Continuous latent tensor from encoder
            
        Returns:
            tuple: (quantized_z, vq_loss, encoding_indices)
                - quantized_z: Discrete version of input
                - vq_loss: Combined codebook and commitment loss
                - encoding_indices: Indices of selected codebook entries
        """
        batch_size, emb_dim, d, h, w = z.shape
        
        # Flatten spatial dimensions for distance computation
        z_flattened = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, emb_dim)
        
        # Compute L2 distances to all codebook vectors
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight ** 2, dim=1) - \
                   2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        # Find nearest codebook entries
        min_encoding_indices = torch.argmin(distances, dim=1)
        
        # Look up quantized vectors and reshape back to spatial dimensions
        z_q = self.embedding(min_encoding_indices).view(batch_size, d, h, w, emb_dim)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        
        # Compute VQ losses
        commitment_loss = F.mse_loss(z_q.detach(), z)  # Encourage encoder commitment
        codebook_loss = F.mse_loss(z_q, z.detach())    # Update codebook
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: forward uses quantized, backward uses continuous
        z_q = z + (z_q - z).detach()
        
        return z_q, vq_loss, min_encoding_indices


# =============================================================================
# COMPLETE AUTOENCODER ARCHITECTURE  
# =============================================================================

class Improved3DAutoencoder(nn.Module):
    """
    Advanced 3D Autoencoder for fMRI Data with Skip Connections and Optional Vector Quantization.
    
    This is the main autoencoder architecture designed specifically for functional MRI data.
    It combines several advanced techniques to achieve high-quality reconstruction and 
    meaningful latent representations:
    
    Key Features:
        - 3D U-Net architecture with skip connections for spatial detail preservation
        - Optional vector quantization for discrete latent spaces
        - Flexible encoder-decoder separation for generation tasks
        - Group normalization for stable training with small batches
        - Residual connections for deep network training stability
    
    Architecture Overview:
        Encoder: Input -> Conv -> ResBlock -> 3x DownBlock -> Bottleneck -> [VQ] -> Latent
        Decoder: Latent -> ResBlock -> 3x UpBlock (with skips) -> FinalUp -> Conv -> Output
    
    The encoder progressively downsamples the input fMRI volume while increasing
    the number of feature channels. The decoder reverses this process, using skip
    connections from the encoder to preserve fine spatial details.
    
    Args:
        in_channels (int): Number of input channels (typically 1 for grayscale fMRI)
        latent_channels (int): Number of channels in the latent bottleneck (default: 8)
        base_channels (int): Base number of feature channels, scales throughout network (default: 32)
        use_vq (bool): Whether to apply vector quantization to latent space (default: True)
        num_vq_embeddings (int): Size of VQ codebook if VQ is enabled (default: 512)
        
    Input Shape:
        (batch_size, in_channels, depth, height, width)
        Typical fMRI: (batch_size, 1, 91, 109, 91)
        
    Output Shapes:
        During training (with skip connections):
            recon: (batch_size, in_channels, depth, height, width) - reconstructed volume
            latent: (batch_size, latent_channels, D', H', W') - compressed representation
            vq_loss: scalar tensor (if VQ enabled) or None
            
        During generation (without skip connections):
            Same as training, but latent comes from external source (e.g., diffusion model)
    
    Usage Examples:
        ```python
        # Standard autoencoder with VQ
        model = Improved3DAutoencoder(in_channels=1, latent_channels=8, use_vq=True)
        
        # Training: full forward pass with reconstruction
        recon, latent, vq_loss = model(fmri_batch)
        
        # Encoding only
        latent, vq_loss, skip_features = model.encode(fmri_batch)
        
        # Decoding with skip connections (reconstruction)
        recon = model.decode(latent, skip_features)
        
        # Decoding without skip connections (generation)
        generated = model.decode(sampled_latent, skip_features=None)
        ```
    
    Design Rationale:
        The architecture balances several competing objectives:
        1. Compression: Reduce spatial dimensions for efficient downstream processing
        2. Reconstruction: Preserve important spatial details through skip connections
        3. Generation: Support novel sample creation without requiring encoder features
        4. Stability: Use normalization and residual connections for robust training
    """
    
    def __init__(self, in_channels: int = 1, latent_channels: int = 8, 
                 base_channels: int = 32, use_vq: bool = True, 
                 num_vq_embeddings: int = 512):
        super().__init__()
        
        # Store configuration
        self.use_vq = use_vq
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        
        # =============================================================================
        # ENCODER ARCHITECTURE
        # =============================================================================
        
        # Initial convolution: map input to base feature channels
        self.initial_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Progressive downsampling with feature expansion
        # Each stage roughly halves spatial dimensions and doubles channels
        self.enc1 = ResidualBlock3D(base_channels, base_channels)              # No downsampling
        self.enc2 = DownBlock3D(base_channels, base_channels*2)                # /2 spatial, *2 channels  
        self.enc3 = DownBlock3D(base_channels*2, base_channels*4)              # /4 spatial, *4 channels
        self.enc4 = DownBlock3D(base_channels*4, base_channels*8)              # /8 spatial, *8 channels
        
        # Bottleneck: compress to latent representation
        self.bottleneck = ResidualBlock3D(base_channels*8, latent_channels)
        
        # Optional vector quantization layer
        if use_vq:
            self.vq = VectorQuantizer(num_embeddings=num_vq_embeddings, 
                                    embedding_dim=latent_channels)
        
        # =============================================================================
        # DECODER ARCHITECTURE
        # =============================================================================
        
        # Initial latent processing
        self.dec4 = ResidualBlock3D(latent_channels, base_channels*8)
        
        # Progressive upsampling with skip connections
        # Skip channels match the corresponding encoder stage output channels
        self.dec3 = UpBlock3D(base_channels*8, base_channels*4, skip_channels=base_channels*8)
        self.dec2 = UpBlock3D(base_channels*4, base_channels*2, skip_channels=base_channels*4)  
        self.dec1 = UpBlock3D(base_channels*2, base_channels, skip_channels=base_channels*2)
        
        # Final upsampling to match input resolution
        self.final_upsample = UpBlock3D(base_channels, base_channels, skip_channels=base_channels)
        
        # Output projection and activation
        self.final = nn.Sequential(
            ResidualBlock3D(base_channels, base_channels),  # Final feature refinement
            nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1),  # Map to output channels
            nn.Sigmoid()  # Ensure output values in [0,1] range (suitable for normalized fMRI)
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """
        Encode input volume to latent representation.
        
        This method processes the input through all encoder stages, optionally
        applies vector quantization, and returns both the latent representation
        and intermediate features needed for skip connections during decoding.
        
        Args:
            x (torch.Tensor): Input fMRI volume of shape (B, C, D, H, W)
            
        Returns:
            tuple: (latent, vq_loss, skip_features)
                - latent: Compressed representation (B, latent_channels, D', H', W')
                - vq_loss: Vector quantization loss (if VQ enabled) or None  
                - skip_features: List of encoder features for skip connections
                  [enc1_out, enc2_out, enc3_out, enc4_out]
        
        Shape Evolution:
            Input (B, 1, 91, 109, 91) ->
            initial_conv (B, 32, 91, 109, 91) ->
            enc1 (B, 32, 91, 109, 91) ->
            enc2 (B, 64, 45, 54, 45) ->  
            enc3 (B, 128, 22, 27, 22) ->
            enc4 (B, 256, 11, 13, 11) ->
            bottleneck (B, 8, 11, 13, 11)
        """
        # Initial feature extraction
        x = self.initial_conv(x)
        
        # Progressive encoding with feature storage for skip connections
        enc1_out = self.enc1(x)      # Store for final skip connection
        enc2_out = self.enc2(enc1_out)  # Store for dec1 skip connection
        enc3_out = self.enc3(enc2_out)  # Store for dec2 skip connection  
        enc4_out = self.enc4(enc3_out)  # Store for dec3 skip connection
        
        # Compress to latent space
        z = self.bottleneck(enc4_out)
        
        # Optional vector quantization
        if self.use_vq:
            z_q, vq_loss, _ = self.vq(z)
            return z_q, vq_loss, [enc1_out, enc2_out, enc3_out, enc4_out]
        else:
            return z, None, [enc1_out, enc2_out, enc3_out, enc4_out]
    
    def decode(self, z: torch.Tensor, skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Decode latent representation back to full resolution volume.
        
        This method supports two modes of operation:
        1. Reconstruction mode: Uses skip connections for high-fidelity reconstruction
        2. Generation mode: Operates without skip connections for novel synthesis
        
        The dual functionality is essential for using the same decoder both during
        autoencoder training (with skip connections) and during diffusion model
        inference (without skip connections).
        
        Args:
            z (torch.Tensor): Latent representation to decode (B, latent_channels, D', H', W')
            skip_features (List[torch.Tensor], optional): Encoder features for skip connections
                Expected as [enc1_out, enc2_out, enc3_out, enc4_out] from encode()
                If None, decoder operates in generation mode without skip connections
                
        Returns:
            torch.Tensor: Reconstructed/generated volume (B, in_channels, D, H, W)
            
        Shape Evolution (reconstruction mode):
            Latent (B, 8, 11, 13, 11) ->
            dec4 (B, 256, 11, 13, 11) ->
            dec3 + skip4 (B, 128, 22, 27, 22) ->
            dec2 + skip3 (B, 64, 45, 54, 45) ->
            dec1 + skip2 (B, 32, 91, 109, 91) ->
            final_up + skip1 (B, 32, 91, 109, 91) ->
            final (B, 1, 91, 109, 91)
            
        Note:
            Skip connection handling includes automatic spatial alignment through
            interpolation when necessary, ensuring robust operation across different
            input sizes or numerical precision variations.
        """
        # Initial latent processing
        dec4_out = self.dec4(z)
        
        if skip_features is not None and len(skip_features) == 4:
            # Reconstruction path: use skip connections for detailed reconstruction
            enc1_out, enc2_out, enc3_out, enc4_out = skip_features
            
            # Progressive upsampling with skip connections
            dec3_out = self.dec3(dec4_out, enc4_out)  # Use encoder stage 4 features
            dec2_out = self.dec2(dec3_out, enc3_out)  # Use encoder stage 3 features  
            dec1_out = self.dec1(dec2_out, enc2_out)  # Use encoder stage 2 features
            final_dec = self.final_upsample(dec1_out, enc1_out)  # Use encoder stage 1 features
            
        else:
            # Generation path: pure upsampling without skip connections
            # This path is used when decoding latent codes from diffusion models
            dec3_out = self.dec3(dec4_out, None)
            dec2_out = self.dec2(dec3_out, None)  
            dec1_out = self.dec1(dec2_out, None)
            final_dec = self.final_upsample(dec1_out, None)
        
        # Final output processing and activation
        out = self.final(final_dec)
        return out
    
    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                               Tuple[torch.Tensor, torch.Tensor]]:
        """
        Complete forward pass: encode -> decode with skip connections.
        
        This method performs the standard autoencoder forward pass, encoding the
        input to a latent representation and then decoding it back using skip
        connections for optimal reconstruction quality.
        
        Args:
            x (torch.Tensor): Input fMRI volume (B, in_channels, D, H, W)
            
        Returns:
            tuple: (reconstruction, latent, vq_loss) if VQ enabled, else (reconstruction, latent)
                - reconstruction: Decoded volume with same shape as input
                - latent: Compressed latent representation  
                - vq_loss: Vector quantization loss (only if use_vq=True)
                
        Example:
            ```python
            model = Improved3DAutoencoder(use_vq=True)
            fmri_batch = torch.randn(4, 1, 91, 109, 91)  # Batch of 4 fMRI volumes
            
            recon, latent, vq_loss = model(fmri_batch)
            
            # Compute reconstruction loss
            recon_loss = F.mse_loss(recon, fmri_batch)
            
            # Total loss combines reconstruction and VQ losses
            total_loss = recon_loss + 0.1 * vq_loss  # VQ loss weighted appropriately
            ```
        """
        # Encode to latent space with skip feature extraction
        if self.use_vq:
            z, vq_loss, skip_features = self.encode(x)
        else:
            z, _, skip_features = self.encode(x)
            vq_loss = None
        
        # Decode using skip connections for high-quality reconstruction
        recon = self.decode(z, skip_features)
        
        # Return appropriate tuple based on VQ configuration
        if self.use_vq:
            return recon, z, vq_loss
        else:
            return recon, z