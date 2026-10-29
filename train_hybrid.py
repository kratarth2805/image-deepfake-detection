import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from models.proposed_model import SpatialFrequencyHybridDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train_hybrid(data_dir="dataset/processed", epochs=5, batch_size=32, lr=1e-3):
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found. Run after dataset is ready.")
        return

    image_datasets = {
        split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split])
        for split in ["train", "val"]
    }
    dataloaders = {
        split: DataLoader(image_datasets[split], batch_size=batch_size, shuffle=(split == "train"))
        for split in ["train", "val"]
    }

    model = SpatialFrequencyHybridDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimize both the frequency branch and classification head
    optimizer = optim.Adam([
        {'params': model.frequency_stream.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr}
    ])

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

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/hybrid_cifake.pth")
    print("Training complete. Hybrid model saved to models/hybrid_cifake.pth")

if __name__ == "__main__":
    train_hybrid()