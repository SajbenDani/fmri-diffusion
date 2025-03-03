import torch
import torch.nn as nn
import torch.nn.functional as F

class Improved3DAutoencoder(nn.Module):
    def __init__(self, latent_dims=(8, 8, 8), num_classes=5):
        super().__init__()
        
        # Calculate total latent dimension size
        self.latent_size = latent_dims[0] * latent_dims[1] * latent_dims[2]
        self.latent_dims = latent_dims
        
        # Label embedding module
        self.label_embedding = nn.Sequential(
            nn.Linear(num_classes, 128), 
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256), 
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.latent_size)
        )
        
        # Encoder
        self.enc_conv1 = nn.Conv3d(1, 32, 3, stride=2, padding=1)
        self.enc_norm1 = nn.GroupNorm(8, 32)  # Group norm for better stability
        
        self.enc_conv2 = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.enc_norm2 = nn.GroupNorm(8, 64)
        
        self.enc_conv3 = nn.Conv3d(64, 128, 3, stride=2, padding=1)
        self.enc_norm3 = nn.GroupNorm(16, 128)
        
        # Final encoding layers
        self.enc_fc1 = nn.Linear(128 * 12 * 14 * 12, 2048)
        self.enc_dropout = nn.Dropout(0.2)
        self.enc_fc2 = nn.Linear(2048, self.latent_size)
        
        # Decoder initial layers
        self.dec_fc1 = nn.Linear(self.latent_size, 2048)
        self.dec_dropout = nn.Dropout(0.2)
        self.dec_fc2 = nn.Linear(2048, 128 * 12 * 14 * 12)
        
        # Decoder - modified to use dynamic output padding
        self.dec_conv1 = nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_norm1 = nn.GroupNorm(8, 64)
        
        self.dec_conv2 = nn.ConvTranspose3d(64 + 64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec_norm2 = nn.GroupNorm(8, 32)
        
        # Final decoder layer without specific output padding - we'll handle this dynamically
        self.dec_conv3 = nn.ConvTranspose3d(32 + 32, 1, 3, stride=2, padding=1, output_padding=0)
        
        # Store original input size for proper reconstruction
        self.original_size = None
    
    def encode(self, x):
        # Store original input size
        self.original_size = x.shape[2:]
        
        # Encoder forward pass
        e1 = F.leaky_relu(self.enc_norm1(self.enc_conv1(x)), 0.2)
        e2 = F.leaky_relu(self.enc_norm2(self.enc_conv2(e1)), 0.2)
        e3 = F.leaky_relu(self.enc_norm3(self.enc_conv3(e2)), 0.2)
        
        # Flatten
        flattened = e3.view(e3.size(0), -1)
        
        # FC layers
        encoded = F.leaky_relu(self.enc_fc1(flattened), 0.2)
        encoded = self.enc_dropout(encoded)
        encoded = self.enc_fc2(encoded)
        
        # Reshape to 3D latent space
        latent_3d = encoded.view(-1, *self.latent_dims)
        
        return encoded, latent_3d, e1, e2
    
    def decode(self, z, e1, e2):
        # FC layers
        d = F.leaky_relu(self.dec_fc1(z), 0.2)
        d = self.dec_dropout(d)
        d = F.leaky_relu(self.dec_fc2(d), 0.2)
        
        # Reshape
        d = d.view(-1, 128, 12, 14, 12)
        
        # Decoder with skip connections
        d = F.leaky_relu(self.dec_norm1(self.dec_conv1(d)), 0.2)
        
        # Resize encoder features if needed
        if d.shape[2:] != e2.shape[2:]:
            e2 = F.interpolate(e2, size=d.shape[2:], mode='trilinear', align_corners=False)
        
        # Skip connection from encoder
        d = torch.cat([d, e2], dim=1)
        d = F.leaky_relu(self.dec_norm2(self.dec_conv2(d)), 0.2)
        
        # Resize encoder features if needed
        if d.shape[2:] != e1.shape[2:]:
            e1 = F.interpolate(e1, size=d.shape[2:], mode='trilinear', align_corners=False)
        
        # Skip connection from encoder
        d = torch.cat([d, e1], dim=1)
        d = self.dec_conv3(d)
        
        # Ensure output size matches input size using interpolation
        if d.shape[2:] != self.original_size:
            d = F.interpolate(d, size=self.original_size, mode='trilinear', align_corners=False)
        
        d = torch.sigmoid(d)  # Apply sigmoid after resizing
        
        return d
    
    def forward(self, x, labels):
        # Encode
        z, latent_3d, e1, e2 = self.encode(x)
        
        # Convert labels to float if they're not already
        if labels.dtype != torch.float32:
            labels = labels.float()
        
        # Process label embedding
        label_features = self.label_embedding(labels)
        
        # Modulate latent representation with label features
        modulated_z = z * torch.sigmoid(label_features)  # Adaptive modulation with sigmoid scaling
        
        # Decode with skip connections
        recon = self.decode(modulated_z, e1, e2)
        
        return recon, latent_3d