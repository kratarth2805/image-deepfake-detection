import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models.baseline_resnet18 import build_resnet18_baseline

# Device configuration (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations: ResNet-18 requires 224x224 input
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_model(data_dir="dataset/processed", epochs=5, batch_size=32, lr=1e-3):
    if not os.path.exists(data_dir):
        print(f"Dataset path '{data_dir}' abhi exist nahi karta. Teammate ka data aane par run karein.")
        return

    # Data loaders
    image_datasets = {
        split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split])
        for split in ["train", "val"]
    }
    dataloaders = {
        split: DataLoader(image_datasets[split], batch_size=batch_size, shuffle=(split == "train"))
        for split in ["train", "val"]
    }

    model = build_resnet18_baseline().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            all_preds, all_labels = [], []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(image_datasets[phase])
            acc = accuracy_score(all_labels, all_preds)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Accuracy: {acc:.4f}")

    # Model save
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnet18_cifake.pth")
    print("Training complete. Model saved in models/resnet18_cifake.pth")

if __name__ == "__main__":
    train_model()