# fMRI Diffusion Model Project

A comprehensive **Latent Diffusion Model (LDM)** framework for generating high-quality synthetic functional Magnetic Resonance Imaging (fMRI) data. This project implements a sophisticated two-stage deep learning pipeline for fMRI super-resolution and synthesis, combining 3D autoencoders with diffusion models for efficient and realistic brain activation pattern generation.

## 🧠 Project Overview

This project addresses the challenge of generating realistic fMRI brain activation patterns using state-of-the-art deep learning techniques. By operating in a learned latent space rather than directly on high-dimensional fMRI volumes, our approach achieves superior computational efficiency while maintaining high-quality synthesis results.

### Key Innovation: Latent Space Diffusion

Instead of applying diffusion models directly to high-resolution fMRI volumes (which would be computationally prohibitive), we:

1. **Stage 1**: Train a 3D autoencoder to learn a compact latent representation of fMRI data
2. **Stage 2**: Train a diffusion model in this learned latent space for efficient generation
3. **Inference**: Generate new latent codes with diffusion, then decode to full-resolution fMRI

This approach reduces computational requirements significantly while preserving spatial coherence and biological plausibility.

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
pip install requirements.txt

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

## 📊 Performance Optimization

### Memory Efficiency
- **Patch-based loading**: Reduces memory footprint vs. full volumes
- **Mixed precision training**: AMP for faster training with lower memory
- **Gradient accumulation**: Effective larger batch sizes
- **Pin memory**: Faster GPU data transfer

### Computational Efficiency  
- **Latent space operation**: significantly faster than pixel-space diffusion
- **Progressive training**: Coarse-to-fine generation strategies
- **Cached datasets**: Pre-processed patches for faster loading
- **Multi-GPU support**: Data parallel training scaling

## 🎯 Applications

This framework enables several important applications:

### 1. Data Augmentation
Generate additional training samples for fMRI analysis pipelines

### 2. Super-Resolution
Enhance spatial resolution of existing fMRI acquisitions

### 4. Missing Data Imputation
Fill in corrupted or missing fMRI volumes

## 📄 License

Property of Daniel Sajben, All rights reserved.

## 📚 Citation

If you use this work in your research, please cite.

---

**Note**: This project is under active development. Please check the repository for the latest updates and improvements.
