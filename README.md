# Image Deepfake Detection: Spatial-Frequency JPEG Robustness Analysis

## Overview
This repository investigates the robustness of deep learning detectors against AI-generated imagery under varying levels of JPEG compression degradation. It compares standard spatial convolutional neural networks (ResNet-18) against a proposed Spatial-Frequency Hybrid architecture that fuses spatial features with 2D Fast Fourier Transform (FFT) magnitude spectra.

## Directory Structure
- `dataset/`: Contains dataset loading protocols and preprocessing scripts.
- `models/`: Implementations of Baseline ResNet-18 and the Proposed Spatial-Frequency Hybrid model.
- `notebooks/`: Interactive notebooks for demonstrations and visual verification.
- `results/`: Spectral artifact comparisons and JPEG degradation performance plots.
- `literature_review/`: Summaries of baseline research papers and citations.

## Workflow Execution Order
1. Place dataset inside `dataset/processed/`.
2. Train baseline model: `python train.py`
3. Train hybrid model: `python train_hybrid.py`
4. Evaluate under JPEG compression: `python evaluate_jpeg.py`
5. Generate paper spectral figures: `python results/visualize_frequency.py`