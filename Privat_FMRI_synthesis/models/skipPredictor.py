import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.adjust_channels = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if self.adjust_channels:
            identity = self.adjust_channels(identity)
        out = self.norm(out + identity)
        return F.relu(out)

class SkipPredictor(nn.Module):
    def __init__(self, latent_dims=(8, 8, 8)):
        super().__init__()
        # Initial expansion from latent space
        self.initial_conv = nn.Conv3d(1, 32, kernel_size=3, padding=1)

        # e2 branch: outputs 64 channels
        self.e2_branch = nn.Sequential(
            ResidualBlock(32, 32),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # e1 branch: outputs 32 channels
        self.e1_branch = nn.Sequential(
            ResidualBlock(32, 32),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, latent_3d, target_shape_e1, target_shape_e2):
        """
        latent_3d: [B, 1, 8, 8, 8]
        target_shape_e1: tuple (D, H, W) for e1
        target_shape_e2: tuple (D, H, W) for e2
        """
        # Expand channels
        x = F.relu(self.initial_conv(latent_3d))

        # Predict e1
        pred_e1 = F.interpolate(x, size=target_shape_e1, mode='trilinear', align_corners=False)
        pred_e1 = self.e1_branch(pred_e1)

        # Predict e2
        pred_e2 = F.interpolate(x, size=target_shape_e2, mode='trilinear', align_corners=False)
        pred_e2 = self.e2_branch(pred_e2)

        return pred_e1, pred_e2