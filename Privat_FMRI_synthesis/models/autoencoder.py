import torch
import torch.nn as nn

class fMRIAutoencoder(nn.Module):
    def __init__(self, latent_dim=256, num_classes=5):
        super().__init__()
        # Add label embedding
        self.label_embed = nn.Embedding(num_classes, latent_dim)

        # Encoder: Expect input shape: (1, 91, 109, 91)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1), nn.ReLU(), # -> (32, 46, 55, 46)
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.ReLU(), # -> (64, 23, 28, 23)
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.ReLU(), # -> (128, 12, 14, 12)
            nn.Flatten(),
            # This change ensures that the linear layer 
            # now accepts 258048 features, which is what we get from the convolutions.
            nn.Linear(128*12*14*12, latent_dim) # Since the actual flattened size is 258048 using a fully connected layer to get the 256
        )
        # Decoder: Mirror the encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, 128 * 12 * 14 * 12),  # + label dimension
            # nn.Linear(latent_dim, 128 * 12 * 14 * 12),
            nn.Unflatten(1, (128, 12, 14, 12)),
            # First deconvolution: (128,12,14,12) -> (64,23,28,23)
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=(0, 1, 0)), nn.ReLU(),
            # Second deconvolution: (64,23,28,23) -> (32,46,55,46)
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=(1, 0, 1)), nn.ReLU(),
            # Third deconvolution: (32,46,55,46) -> (1,91,109,91)
            nn.ConvTranspose3d(32, 1, 3, stride=2, padding=1, output_padding=0), nn.Sigmoid()
        )

    def forward(self, x, labels):
        # Encode
        z = self.encoder(x)
        # Get the label embedding
        label_emb = self.label_embed(labels)
        # Concatenate latent vector and label embedding along the feature dimension
        conditioned_z = torch.cat([z, label_emb], dim=-1)
        # Decode to reconstruct the input
        return self.decoder(conditioned_z)
