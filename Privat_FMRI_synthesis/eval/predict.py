import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os

from models.diffusion import diffusion_model, LATENT_DIM, spatial_dim
from models.autoencoder import fMRIAutoencoder
from config import DEVICE, AUTOENCODER_CHECKPOINT, DIFFUSION_CHECKPOINT, LOG_DIR, NUM_TIMESTEPS

from diffusers import DDPMScheduler

# Set the directory for saving outputs
os.makedirs(LOG_DIR, exist_ok=True)

# Load the pre-trained autoencoder
autoencoder = fMRIAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
autoencoder.load_state_dict(torch.load(AUTOENCODER_CHECKPOINT, map_location=DEVICE))
autoencoder.eval()

# Load the pre-trained diffusion model
diffusion_model.load_state_dict(torch.load(DIFFUSION_CHECKPOINT, map_location=DEVICE))
diffusion_model.eval()

# Define the scheduler for the diffusion process
scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)

def predict_and_generate(num_steps=50, seed=42):
    """
    Generate one fMRI image from noise via reverse diffusion and create a GIF of the denoising process.
    
    Parameters:
      num_steps (int): Number of reverse diffusion steps.
      seed (int): Random seed for reproducibility.
    
    Returns:
      final_img (np.array): The final generated 2D slice (grayscale image).
      frames (list): List of 2D numpy arrays (frames for the GIF).
    """
    # For reproducibility
    torch.manual_seed(seed)
    
    # Generate initial noise with multiple intensity variations (for richer patterns)
    latent = torch.randn((1, 1, spatial_dim, spatial_dim), device=DEVICE) * 2  # Scaled noise
    initial_latent = latent.clone().detach()

    frames = []  # Store intermediate images for the GIF
    
    # Create a sequence of timesteps (from 999 to 0, linearly spaced)
    timesteps = torch.linspace(NUM_TIMESTEPS - 1, 0, steps=num_steps).long().to(DEVICE)
    
    for t in timesteps:
        with torch.no_grad():
            output = diffusion_model(latent, t.unsqueeze(0))
        predicted_noise = output.sample if hasattr(output, 'sample') else output
        
        step_output = scheduler.step(predicted_noise, t, latent)
        latent = step_output.prev_sample
        
        # Decode via the autoencoder
        latent_flat = latent.view(latent.shape[0], -1)
        with torch.no_grad():
            generated = autoencoder.decoder(latent_flat)
        
        # Extract the middle slice
        img_slice = generated[0, 0, generated.shape[2] // 2, :, :].cpu().numpy()
        
        # Normalize and apply colormap (Green Activation Map) to frames
        colormap = plt.get_cmap("Greens")

        # Normalize the frame for colormap application
        normalized_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

        # Apply the colormap
        colored_frame = colormap(normalized_slice)  # This will give an RGBA image

        # Convert to RGB by removing the alpha channel (if you want to keep the GIF format)
        colored_frame_rgb = (colored_frame[:, :, :3] * 255).astype(np.uint8)

        # Append the RGB frame to the frames list
        frames.append(colored_frame_rgb)
    
    # Final generated image
    latent_flat = latent.view(latent.shape[0], -1)
    with torch.no_grad():
        final_generated = autoencoder.decoder(latent_flat)
    
    final_img = final_generated[0, 0, final_generated.shape[2] // 2, :, :].cpu().numpy()
    
    # Normalize the final image
    final_img = (final_img - final_img.min()) / (final_img.max() - final_img.min() + 1e-8)
    final_img_255 = (final_img * 255).astype(np.uint8)

    # Create a noise image that isn't just a black screen
    noise_img = initial_latent[0, 0].cpu().numpy()
    noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min() + 1e-8)
    noise_img = (noise_img * 255).astype(np.uint8)

    # Display & save the images side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Initial Noise (Random Pattern)
    axes[0].imshow(noise_img, cmap='nipy_spectral')  # Use a colorful colormap
    axes[0].set_title("Initial Noise (Random)")
    axes[0].axis("off")
    plt.imsave(os.path.join(LOG_DIR, "initial_noise.png"), noise_img, cmap="nipy_spectral")

    # Final fMRI (Grayscale)
    axes[1].imshow(final_img_255, cmap="gray")
    axes[1].set_title("Generated fMRI (Grayscale)")
    axes[1].axis("off")
    plt.imsave(os.path.join(LOG_DIR, "generated_fmri_grayscale.png"), final_img_255, cmap="gray")

    # Final fMRI (Green Activation Map)
    axes[2].imshow(final_img, cmap="Greens", alpha=0.85)  # Make activation pop out
    axes[2].set_title("Generated fMRI (Activation Map)")
    axes[2].axis("off")
    plt.imsave(os.path.join(LOG_DIR, "generated_fmri_activation.png"), final_img, cmap="Greens")

    plt.show()

    # Save the GIF of the reverse diffusion process
    gif_filename = os.path.join(LOG_DIR, "diffusion_generation.gif")
    imageio.mimsave(gif_filename, frames, duration=0.1)
    print(f"GIF saved at: {gif_filename}")

    return final_img, frames

# Run the predictor to generate one sample
final_img, frames = predict_and_generate(num_steps=50, seed=42)
