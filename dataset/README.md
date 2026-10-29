# Dataset Directory Protocol

## 1. Expected Directory Structure
The training and evaluation scripts require images arranged in standard `ImageFolder` hierarchy:

```text
dataset/
└── processed/
    ├── train/
    │   ├── REAL/
    │   └── FAKE/
    ├── val/
    │   ├── REAL/
    │   └── FAKE/
    └── test/
        ├── REAL/
        └── FAKE/