import os
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from models.baseline_resnet18 import build_resnet18_baseline
from models.proposed_model import SpatialFrequencyHybridDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compress_image_jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Applies JPEG compression artifacts in memory at specified quality."""
    if quality >= 100:
        return image
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

class JPEGCompressedDataset(Dataset):
    def __init__(self, root_dir, quality=100, transform=None):
        self.root_dir = root_dir
        self.quality = quality
        self.transform = transform
        self.samples = []
        
        for label_idx, cls_name in enumerate(["REAL", "FAKE"]):
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_folder):
                cls_folder = os.path.join(root_dir, cls_name.lower())
            if os.path.exists(cls_folder):
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(cls_folder, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = compress_image_jpeg(img, self.quality)
        if self.transform:
            img = self.transform(img)
        return img, label

def evaluate_models_under_jpeg(test_dir="dataset/processed/test"):
    if not os.path.exists(test_dir):
        print(f"Test directory '{test_dir}' not found yet. Awaiting dataset.")
        return

    qualities = [100, 80, 60, 40, 20]
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Instantiate models
    baseline = build_resnet18_baseline().to(device)
    hybrid = SpatialFrequencyHybridDetector().to(device)
    
    # Load weights if available
    if os.path.exists("models/resnet18_cifake.pth"):
        baseline.load_state_dict(torch.load("models/resnet18_cifake.pth", map_location=device))
    if os.path.exists("models/hybrid_cifake.pth"):
        hybrid.load_state_dict(torch.load("models/hybrid_cifake.pth", map_location=device))

    baseline.eval()
    hybrid.eval()

    baseline_accs, hybrid_accs = [], []

    for q in qualities:
        dataset = JPEGCompressedDataset(test_dir, quality=q, transform=eval_transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        for model, acc_list in [(baseline, baseline_accs), (hybrid, hybrid_accs)]:
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
            acc = accuracy_score(all_labels, all_preds)
            acc_list.append(acc)

    # Plot JPEG degradation curve for the research paper
    os.makedirs("results/plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(qualities, baseline_accs, marker="o", label="Spatial Baseline (ResNet-18)")
    plt.plot(qualities, hybrid_accs, marker="s", label="Proposed Spatial-Frequency Hybrid")
    plt.xlabel("JPEG Quality Factor (Q)")
    plt.ylabel("Accuracy")
    plt.title("Effect of JPEG Compression on Detector Performance")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid(True)
    plot_path = "results/plots/jpeg_robustness_curve.png"
    plt.savefig(plot_path)
    print(f"Evaluation complete. Curve plot saved at: {plot_path}")

if __name__ == "__main__":
    evaluate_models_under_jpeg()