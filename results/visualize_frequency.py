import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_fft_spectrum(img_path):
    img = Image.open(img_path).convert("L")  # Convert to grayscale
    img_arr = np.array(img, dtype=np.float32)
    
    # 2D Fast Fourier Transform
    f = np.fft.fft2(img_arr)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return img, magnitude_spectrum

def generate_frequency_comparison(real_img_path, fake_img_path, output_path="results/plots/fft_comparison.png"):
    if not (os.path.exists(real_img_path) and os.path.exists(fake_img_path)):
        print("Sample images not found. Run this after the dataset subset is populated.")
        return

    real_img, real_fft = get_fft_spectrum(real_img_path)
    fake_img, fake_fft = get_fft_spectrum(fake_img_path)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Real Image & Spectrum
    axes[0, 0].imshow(real_img, cmap='gray')
    axes[0, 0].set_title("Real Photo (Spatial)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(real_fft, cmap='magma')
    axes[0, 1].set_title("Real Photo (2D-FFT Spectrum)")
    axes[0, 1].axis('off')

    # Fake Image & Spectrum
    axes[1, 0].imshow(fake_img, cmap='gray')
    axes[1, 0].set_title("AI-Generated Photo (Spatial)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(fake_fft, cmap='magma')
    axes[1, 1].set_title("AI-Generated (2D-FFT Spectrum Artifacts)")
    axes[1, 1].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Spectral analysis plot saved at {output_path}")

if __name__ == "__main__":
    # Test path pointers
    sample_real = "dataset/processed/test/REAL/0000.jpg"
    sample_fake = "dataset/processed/test/FAKE/0000.jpg"
    generate_frequency_comparison(sample_real, sample_fake)