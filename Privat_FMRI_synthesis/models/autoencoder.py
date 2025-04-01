import torch
import torch.nn as nn
import torch.nn.functional as F

# Multiplicative FiLM-based label embedding
class FiLMLabelEmbedding(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = nn.Linear(num_classes, latent_dim)
        self.modulation = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()  # Sigmoid ensures scaling factor between 0 and 1
        )

    def forward(self, labels, latents):
        if labels.dim() == 1:  # Ha csak egy dimenziójú, akkor one-hot enkódolás kell
            labels = labels.long()
            labels = F.one_hot(labels, num_classes=self.num_classes).float()

        label_emb = self.embedding(labels)  # [B, latent_dim]
        scale = self.modulation(label_emb)  # [B, latent_dim]

        # Dinamikus átméretezés a latents dimenziói alapján
        while len(scale.shape) < len(latents.shape):
            scale = scale.unsqueeze(-1)

        return latents * scale  # Multiplicative FiLM-style conditioning

class fMRIAutoencoder(nn.Module):
    def __init__(self, latent_dim=1024, num_classes=5):
        super().__init__()
        self.label_embed = FiLMLabelEmbedding(num_classes, latent_dim)  # Using FiLM-based embedding

        # Encoder: Expect input shape: (1, 91, 109, 91)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(), # -> (32, 46, 55, 46)
            nn.Dropout(0.2),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(), # -> (64, 23, 28, 23)
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.ReLU(), # -> (128, 12, 14, 12)
            nn.Flatten(),
            nn.Linear(128*12*14*12, 2048),  # Adjusted for 1024 latent dim
            nn.Dropout(0.2),
            nn.Linear(2048, latent_dim)
        )
        
        # Decoder: Mirror the encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.Dropout(0.2),
            nn.Linear(2048, 128*12*14*12),
            nn.Dropout(0.2),
            nn.Unflatten(1, (128, 12, 14, 12)),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=(0, 1, 0)), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=(1, 0, 1)), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.ConvTranspose3d(32, 1, 3, stride=2, padding=1, output_padding=0), nn.Sigmoid()
        )

    def forward(self, x, labels):
        # Encode
        z = self.encoder(x)
        # Apply FiLM-based label embedding
        z = self.label_embed(labels, z)
        # Decode to reconstruct the input
        return self.decoder(z)
