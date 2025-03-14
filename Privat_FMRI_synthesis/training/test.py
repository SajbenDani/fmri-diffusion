import os
import sys

# Add parent directory to sys.path to find config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CHECKPOINT_DIR, AUTOENCODER_CHECKPOINT

def test_checkpoint_directory():
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Checkpoint directory '{CHECKPOINT_DIR}' does not exist. Creating it...")
        os.makedirs(CHECKPOINT_DIR)
        print(f"Directory '{CHECKPOINT_DIR}' created successfully.")
    else:
        print(f"Checkpoint directory '{CHECKPOINT_DIR}' already exists.")

    test_file_path = os.path.join(CHECKPOINT_DIR, "test_checkpoint.txt")

    try:
        with open(test_file_path, "w") as f:
            f.write("This is a test file to check if writing to the checkpoint directory works.\n")
        print(f"Test file '{test_file_path}' written successfully.")

        with open(test_file_path, "r") as f:
            content = f.read()
            print("File content verification:", content.strip())

    except Exception as e:
        print(f"Error writing to '{test_file_path}': {e}")

if __name__ == "__main__":
    test_checkpoint_directory()
