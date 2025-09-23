# fMRI Diffusion Model Project

A comprehensive **Latent Diffusion Model (LDM)** framework for generating high-quality synthetic functional Magnetic Resonance Imaging (fMRI) data. This project implements a sophisticated two-stage deep learning pipeline for fMRI super-resolution and synthesis, combining 3D autoencoders with diffusion models for efficient and realistic brain activation pattern generation.

## 🧠 Project Overview

This project addresses the challenge of generating realistic fMRI brain activation patterns using state-of-the-art deep learning techniques. By operating in a learned latent space rather than directly on high-dimensional fMRI volumes, our approach achieves superior computational efficiency while maintaining high-quality synthesis results.

### Key Innovation: Latent Space Diffusion

Instead of applying diffusion models directly to high-resolution fMRI volumes (which would be computationally prohibitive), we:

1. **Stage 1**: Train a 3D autoencoder to learn a compact latent representation of fMRI data
2. **Stage 2**: Train a diffusion model in this learned latent space for efficient generation
3. **Inference**: Generate new latent codes with diffusion, then decode to full-resolution fMRI

This approach reduces computational requirements by ~10-100x while preserving spatial coherence and biological plausibility.

## 🏗️ Architecture Components

### 1. 3D Autoencoder (`models/autoencoder.py`)
- **Purpose**: Learn compressed latent representations of 3D fMRI volumes
- **Input**: fMRI volumes (typically 91×109×91 voxels)
- **Output**: Latent codes (e.g., 8 channels at reduced spatial resolution)
- **Features**:
  - U-Net architecture with skip connections for high-fidelity reconstruction
  - Optional Vector Quantization (VQ) for discrete latent spaces
  - Switchable skip connections (training vs. generation modes)
  - Residual blocks with group normalization for stable training

### 2. 3D Diffusion Model (`models/diffusion.py`)
- **Purpose**: Generate realistic latent codes via denoising diffusion
- **Architecture**: 3D U-Net with time conditioning and attention mechanisms
- **Features**:
  - Time-conditional generation with sinusoidal position embeddings
  - Multi-scale processing through encoder-decoder structure
  - Support for classifier-free guidance and conditioning
  - Memory-efficient operation in compressed latent space

### 3. Data Management (`utils/dataset.py`)
- **Purpose**: Efficient loading and preprocessing of fMRI datasets
- **Features**:
  - Patch-based loading for memory efficiency
  - Multi-view support (axial, coronal, sagittal orientations)
  - Robust error handling and caching mechanisms
  - Motor task label mapping (left/right hand, foot, tongue movements)

## 📊 Supported fMRI Tasks

The framework currently supports motor task fMRI data with the following conditions:

| Task Code | Description | Label ID |
|-----------|-------------|----------|
| `lh` | Left Hand Movement | 0 |
| `rh` | Right Hand Movement | 1 |
| `lf` | Left Foot Movement | 2 |
| `rf` | Right Foot Movement | 3 |
| `t` | Tongue Movement | 4 |

## 🚀 Quick Start

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install diffusers transformers
pip install monai nibabel pandas scikit-image
pip install matplotlib tqdm wandb

# Optional: for advanced loss functions
pip install pytorch-msssim
```

### Basic Usage

1. **Configure paths and parameters** in `config.py`:
```python
# Update these paths for your setup
TRAIN_DATA = "path/to/train.csv"
VAL_DATA = "path/to/val.csv"
DATA_DIR = "path/to/fmri/data"
```

2. **Train the autoencoder** (Stage 1):
```bash
python main.py  # Set MODE = "train_autoencoder"
```

3. **Train the diffusion model** (Stage 2):
```bash
python main.py  # Set MODE = "train_diffusion"
```

4. **Generate synthetic fMRI data**:
```bash
python main.py  # Set MODE = "predict"
```

### Advanced Training Options

**Custom autoencoder training:**
```bash
python -m training.train_autoencoder
```

**Diffusion with classifier-free guidance:**
```bash
python -m training.train_diffusion_with_cfg
```

**Fine-tuning pre-trained models:**
```bash
python -m training.finetune_autoencoder
```

## 📁 Project Structure

```
fmri-diffusion/
├── Privat_FMRI_synthesis/
│   ├── config.py                    # Global configuration parameters
│   ├── main.py                      # Main entry point and mode selector
│   │
│   ├── models/                      # Neural network architectures
│   │   ├── autoencoder.py          # 3D autoencoder with skip connections
│   │   ├── diffusion.py            # 3D diffusion U-Net model
│   │   └── predict.py              # Inference and generation scripts
│   │
│   ├── training/                    # Training pipelines
│   │   ├── train_autoencoder.py    # Autoencoder training (Stage 1)
│   │   ├── train_diffusion.py      # Diffusion training (Stage 2)
│   │   ├── train_diffusion_with_cfg.py  # CFG training
│   │   └── finetune_*.py           # Fine-tuning scripts
│   │
│   ├── utils/                       # Data handling and utilities
│   │   ├── dataset.py              # fMRI data loading and preprocessing
│   │   └── preprocess_data.py      # Data preparation utilities
│   │
│   └── eval/                        # Evaluation and analysis
│       ├── ComplexEval.py          # Comprehensive quantitative evaluation
│       ├── gridBased.py            # Grid-based visualization
│       └── *_visual_test.py        # Visual quality assessment
│
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## 🔬 Training Pipeline

### Stage 1: Autoencoder Training

**Objective**: Learn a compact latent representation of fMRI volumes

**Process**:
1. Load fMRI patch datasets (train/validation/test splits)
2. Train 3D U-Net autoencoder with reconstruction loss
3. Optional: Add Vector Quantization for discrete latents
4. Validate on held-out data and save best checkpoint

**Key Parameters**:
- `BATCH_SIZE`: 16 (adjust based on GPU memory)
- `LEARNING_RATE`: 1e-4
- `LATENT_CHANNELS`: 8 (latent space dimensionality)
- `USE_VQ`: True/False (vector quantization)

### Stage 2: Diffusion Training

**Objective**: Learn to generate realistic latent codes via denoising

**Process**:
1. Load pre-trained autoencoder (frozen encoder)
2. Train diffusion model to predict noise in latent space
3. Optional: Add classifier-free guidance for controllable generation
4. Validate generation quality and save checkpoints

**Key Parameters**:
- `NUM_TIMESTEPS`: 1000 (diffusion steps)
- `GUIDANCE_SCALE`: 7.5 (CFG strength)
- `SCHEDULER`: DDPM/DDIM for different sampling strategies

## 📈 Evaluation Metrics

The framework provides comprehensive evaluation through multiple metrics:

### Quantitative Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction fidelity
- **SSIM (Structural Similarity)**: Assesses perceptual similarity
- **MS-SSIM**: Multi-scale structural similarity for complex patterns

### Qualitative Assessment
- **Grid Visualizations**: Multi-slice brain activation patterns
- **Anatomical Views**: Axial, coronal, and sagittal orientations  
- **Activation Heatmaps**: Color-coded brain activity visualization

### Usage Example:
```bash
# Run comprehensive evaluation
python -m eval.ComplexEval

# Generate visualization grids
python -m eval.gridBased

# Autoencoder-specific evaluation
python -m eval.autoencoder_visual_test
```

## ⚙️ Configuration Options

### Hardware Settings
```python
DEVICE = "cuda"  # or "cpu"
BATCH_SIZE = 16  # Adjust based on GPU memory
NUM_WORKERS = 8  # Parallel data loading processes
```

### Model Architecture
```python
LATENT_CHANNELS = 8      # Latent space dimensionality
BASE_CHANNELS = 32       # Autoencoder base channels  
USE_VQ = True           # Vector quantization
VQ_EMBEDDINGS = 512     # VQ codebook size
```

### Training Parameters
```python
EPOCHS_AUTOENCODER = 50   # Stage 1 training epochs
EPOCHS_DIFFUSION = 100    # Stage 2 training epochs
LEARNING_RATE = 1e-4      # Optimizer learning rate
PATIENCE = 10             # Early stopping patience
```

## 🔧 Advanced Features

### Classifier-Free Guidance (CFG)
Enables controllable generation with conditioning:
```python
# Train with CFG support
python -m training.train_diffusion_with_cfg

# Generate with specific guidance scale
guidance_scale = 7.5  # Controls conditioning strength
```

### Vector Quantization (VQ)
Creates discrete latent spaces for improved generation:
```python
# Enable VQ in autoencoder
USE_VQ = True
VQ_EMBEDDINGS = 512  # Codebook size
```

### Multi-View Processing
Support for different anatomical orientations:
```python
VIEW = 'axial'     # or 'coronal', 'sagittal'
```

## 📊 Performance Optimization

### Memory Efficiency
- **Patch-based loading**: Reduces memory footprint vs. full volumes
- **Mixed precision training**: AMP for faster training with lower memory
- **Gradient accumulation**: Effective larger batch sizes
- **Pin memory**: Faster GPU data transfer

### Computational Efficiency  
- **Latent space operation**: ~10-100x faster than pixel-space diffusion
- **Progressive training**: Coarse-to-fine generation strategies
- **Cached datasets**: Pre-processed patches for faster loading
- **Multi-GPU support**: Data parallel training scaling

## 🎯 Applications

This framework enables several important applications:

### 1. Data Augmentation
Generate additional training samples for fMRI analysis pipelines

### 2. Super-Resolution
Enhance spatial resolution of existing fMRI acquisitions

### 3. Denoising
Remove artifacts and noise from fMRI signals

### 4. Missing Data Imputation
Fill in corrupted or missing fMRI volumes

### 5. Cross-Subject Synthesis
Generate fMRI patterns consistent with population statistics

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Code Style**: Follow PEP 8 with comprehensive docstrings
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update docs for any API changes
4. **Issues**: Use GitHub issues for bug reports and feature requests

### Development Setup
```bash
git clone https://github.com/SajbenDani/fmri-diffusion.git
cd fmri-diffusion
pip install -r requirements.txt
pip install -e .  # Development installation
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{fmri_diffusion_2024,
  title={fMRI Diffusion Model: Latent Diffusion for Synthetic Brain Activation Generation},
  author={SajbenDani and contributors},
  year={2024},
  url={https://github.com/SajbenDani/fmri-diffusion}
}
```

## 🆘 Support

- **Documentation**: See inline docstrings and comments throughout the codebase
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for general questions and ideas

## 🔮 Future Directions

Planned enhancements include:

- **Multi-modal conditioning**: Incorporate structural MRI, behavioral data
- **Temporal modeling**: Extend to fMRI time series (4D) generation  
- **Cross-domain transfer**: Adapt to other neuroimaging modalities
- **Clinical applications**: Disease-specific synthetic data generation
- **Real-time processing**: Optimizations for online/streaming applications

---

**Note**: This project is under active development. Please check the repository for the latest updates and improvements.
