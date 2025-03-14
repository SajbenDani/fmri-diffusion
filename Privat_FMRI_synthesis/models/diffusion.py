import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class UNet3DDiffusion(nn.Module):
    def __init__(self, latent_shape=(8,8,8), num_classes=5, base_channels=32, time_emb_dim=64):
        """
        A deeper 3D UNet with skip connections for diffusion in latent space.
        Input shape: [B, 1, D, H, W] where D, H, W = latent_shape.
        """
        super(UNet3DDiffusion, self).__init__()
        self.latent_shape = latent_shape
        self.in_channels = 1  # Our latent is treated as a single-channel 3D volume
        
        # Time embedding (expects a scalar time per sample)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Label embedding module - similar structure to autoencoder
        self.label_mlp = nn.Sequential(
            nn.Linear(num_classes, 128), 
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256), 
            nn.LeakyReLU(0.2),
            nn.Linear(256, time_emb_dim)  # Changed to match time_emb_dim
        )
        
        # The conditioning projection to match bottleneck channels
        # Update this projection to match base_channels*8 instead of base_channels*4
        self.cond_proj = nn.Linear(time_emb_dim, base_channels*8)
        
        # Encoder path with more layers
        self.enc_conv1 = nn.Conv3d(self.in_channels, base_channels, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1)
        self.enc_conv4 = nn.Conv3d(base_channels*4, base_channels*8, kernel_size=3, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Conv3d(base_channels*8, base_channels*8, kernel_size=3, padding=1)
        
        # Decoder path with more layers
        self.dec_conv1 = nn.Conv3d(base_channels*8, base_channels*4, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv3d(base_channels*4, base_channels*2, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv3d(base_channels*2, base_channels, kernel_size=3, padding=1)
        self.dec_conv4 = nn.Conv3d(base_channels, self.in_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t, labels):
        # x: [B, 1, D, H, W]; t: [B] (as float), labels: one-hot [B, num_classes]
        B = x.size(0)
        
        # Compute conditioning embeddings
        t_emb = self.time_mlp(t.view(B, 1))          # [B, time_emb_dim]
        label_emb = self.label_mlp(labels)           # [B, time_emb_dim]
        combined_emb = t_emb + label_emb             # [B, time_emb_dim]
        
        # Project to match bottleneck channels
        cond_signal = self.cond_proj(combined_emb)   # [B, base_channels*8]
        cond_signal = cond_signal.view(B, -1, 1, 1, 1)  # [B, base_channels*8, 1, 1, 1] to match 3d tensors
        
        # Encoder
        e1 = F.silu(self.enc_conv1(x))               # [B, base_channels, D, H, W]
        e2 = F.silu(self.enc_conv2(e1))              # [B, base_channels*2, D, H, W]
        e3 = F.silu(self.enc_conv3(e2))              # [B, base_channels*4, D, H, W]
        e4 = F.silu(self.enc_conv4(e3))              # [B, base_channels*8, D, H, W]
        
        # Bottleneck with conditioning
        b = self.bottleneck(e4)                      # [B, base_channels*8, D, H, W]
        b = b + cond_signal                          # Add conditioning signal
        b = F.silu(b)                                # Apply activation after addition
        
        # Decoder with skip connections
        d1 = F.silu(self.dec_conv1(b))               # [B, base_channels*4, D, H, W]
        d1 = d1 + e3                                 # Skip connection from encoder layer e3
        
        d2 = F.silu(self.dec_conv2(d1))              # [B, base_channels*2, D, H, W]
        d2 = d2 + e2                                 # Skip connection from encoder layer e2
        
        d3 = F.silu(self.dec_conv3(d2))              # [B, base_channels, D, H, W]
        d3 = d3 + e1                                 # Skip connection from encoder layer e1
        
        d4 = self.dec_conv4(d3)                      # [B, 1, D, H, W]
        
        return d4

class LatentDiffusion:
    def __init__(self, latent_shape=(8,8,8), num_classes=5, device='cuda'):
        """
        Diffusion model that works in the latent space.
        Expects latent codes shaped as [B, 1, 8, 8, 8] (i.e. 8x8x8 with one channel).
        """
        self.latent_shape = latent_shape
        self.num_classes = num_classes
        self.device = device
        self.model = UNet3DDiffusion(latent_shape=latent_shape, num_classes=num_classes).to(device)
        self.optim = Adam(self.model.parameters(), lr=1e-4)
        self.timesteps = 1000
        
    def noise_schedule(self, t):
        # Simple cosine schedule for better sampling stability
        max_steps = self.timesteps
        return torch.cos(((t / max_steps + 0.008) / 1.008) * torch.pi / 2) ** 2

    def forward_diffusion(self, x):
        """
        Adds noise to the latent code x according to a sampled timestep.
        x: [B, 1, D, H, W]
        Returns: noisy x, sampled timestep t, and the true noise added.
        """
        t = torch.randint(0, self.timesteps, (x.size(0),), device=self.device).float()
        noise = torch.randn_like(x).to(self.device)
        alpha = self.noise_schedule(t).view(-1, 1, 1, 1, 1)
        noisy_x = alpha * x + (1 - alpha) * noise
        return noisy_x, t, noise
    
    # def train_step(self, x, labels):
    #     """
    #     Performs one training step:
    #       - x: latent codes [B, 1, D, H, W]
    #       - labels: one-hot encoded [B, num_classes]
    #     """
    #     noisy_x, t, true_noise = self.forward_diffusion(x)
    #     pred_noise = self.model(noisy_x, t, labels)
    #     loss = nn.MSELoss()(pred_noise, true_noise)
    #     self.optim.zero_grad()
    #     loss.backward()
    #     self.optim.step()
    #     return loss.item()
    
    # def sample(self, labels, steps=50):
    #     """
    #     Generate samples from noise conditioned on labels
    #     labels: [B, num_classes] one-hot encoded
    #     steps: number of diffusion steps for sampling (fewer = faster but lower quality)
    #     Returns: generated latent codes [B, 1, D, H, W]
    #     """
    #     self.model.eval()
    #     B = labels.size(0)
        
    #     # Start from random noise
    #     x = torch.randn(B, 1, *self.latent_shape, device=self.device)
        
    #     # Gradually denoise in reverse timesteps
    #     time_steps = torch.linspace(self.timesteps-1, 0, steps, device=self.device)
        
    #     with torch.no_grad():
    #         for i, t in enumerate(time_steps):
    #             # Broadcast t for the batch
    #             t_batch = torch.ones(B, device=self.device) * t
                
    #             # Predict noise
    #             pred_noise = self.model(x, t_batch, labels)
                
    #             # Current alpha
    #             alpha = self.noise_schedule(t_batch).view(-1, 1, 1, 1, 1)
                
    #             # Different alpha for the next step (or 1.0 for the final step)
    #             alpha_next = self.noise_schedule(torch.max(t_batch - 1, torch.zeros_like(t_batch))).view(-1, 1, 1, 1, 1)
                
    #             # Denoise step
    #             x = (x - (1 - alpha) * pred_noise) / torch.sqrt(alpha)
                
    #             # Add noise for all steps except the last one
    #             if i < steps - 1:
    #                 noise = torch.randn_like(x, device=self.device)
    #                 sigma = torch.sqrt((1 - alpha_next) / (1 - alpha) * (1 - alpha/alpha_next))
    #                 x = torch.sqrt(alpha_next) * x + sigma * noise
        
    #     self.model.train()
    #     return x
