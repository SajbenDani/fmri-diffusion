import subprocess

# Set the mode: choose from 'train_diffusion', 'train_autoencoder', or 'predict'
MODE = "predict"  # Change this to switch tasks

if MODE == "train_diffusion":
    subprocess.run(["python", "-m", "training.train_diffusion"])
elif MODE == "train_autoencoder":
    subprocess.run(["python", "-m", "training.train_autoencoder"])
elif MODE == "predict":
    subprocess.run(["python", "-m", "models.predict"])
else:
    print("Invalid mode selected. Choose 'train_diffusion', 'train_autoencoder', or 'predict'.")
