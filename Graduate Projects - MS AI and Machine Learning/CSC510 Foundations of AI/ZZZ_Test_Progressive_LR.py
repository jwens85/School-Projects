import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Path Directories
CSV_PATH = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\data.csv"
IMAGE_DIR = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\image"

# Set device as GPU (faster) or CPU (slower)
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
print(f"Using device: {device}")

# Load CSV
df = pd.read_csv(CSV_PATH)
df["image"] = df["image"].str.replace("image/", "", regex=False)

# Convert categorical class labels into integers
class_mapping = {label: idx for idx, label in enumerate(df["classes"].unique())}
df["label"] = df["classes"].map(class_mapping)

# Print class mappings and distribution of damage labels
print("Class mapping:")
for class_name, idx in class_mapping.items():
    print(f"{idx}: {class_name}")

# Analyze the frequency distribution of class labels
class_counts = df["classes"].value_counts()
print("\nClass distribution:")
print(class_counts.to_string())

# SciKit-Learn Train/Test Split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=13)

# Data Augmentation with TorchVision
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# VehicleDamageDataset Class
class VehicleDamageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(IMAGE_DIR, self.dataframe.iloc[idx]["image"])
        label = self.dataframe.iloc[idx]["label"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# Create datasets and the dataset autoloader
train_dataset = VehicleDamageDataset(train_df, transform=transform)
val_dataset = VehicleDamageDataset(val_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load ResNet152
from torchvision.models import ResNet152_Weights

model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
num_ftrs = model.fc.in_features

# nn.Sequential and Dropout Layer
model.fc = nn.Sequential(
    nn.Dropout(0.1),  # 10% seems to be optimum
    nn.Linear(num_ftrs, len(class_mapping))
)

# Initially freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Define layers for progressive unfreezing - from shallow to deep
unfreezing_stages = [
    model.layer4,  # Unfreeze first (already in the original code)
    model.layer3,  # Unfreeze second (already in the original code)
    model.layer2,  # Unfreeze third (already in the original code)
    model.layer1,  # Additional layer to unfreeze
    model.bn1,  # Additional layer to unfreeze
    model.conv1  # Additional layer to unfreeze
]

# Start with the first three unfrozen as in the original code
unfrozen_idx = 3  # Start with layers 4, 3, and 2 unfrozen
for i in range(unfrozen_idx):
    for param in unfreezing_stages[i].parameters():
        param.requires_grad = True

# Also make sure to unfreeze the new dropout-enhanced fc layer
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

# **Calculate Class Weights for Imbalanced Dataset**
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label'].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("\nClass weights:")
for idx, weight in enumerate(class_weights):
    class_name = [k for k, v in class_mapping.items() if v == idx][0]
    print(f"{class_name}: {weight.item():.4f}")

# **Define Loss Function with Class Weighting**
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.00003)

# Define initial learning rate
initial_lr = 0.00003

# **Dynamic Learning Rate Scheduler** (unchanged from your original code)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.5
)

# **Training Loop with Balanced Model Saving**
num_epochs = 50
best_accuracy = 0.0
best_loss = float("inf")

# Variables for progressive unfreezing
no_improvement_count = 0
previous_best = 0.0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total

    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    scheduler.step(val_accuracy)

    # Print epoch statistics
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch + 1}: Training Loss = {running_loss / len(train_loader):.4f}, Training Accuracy = {train_accuracy:.2f}%")
    print(
        f"Epoch {epoch + 1}: Validation Loss = {val_loss / len(val_loader):.4f}, Validation Accuracy = {val_accuracy:.2f}%, Learning Rate = {current_lr:.8f}")

    # **Save Best Model Checkpoint (Fixed)**
    if val_accuracy > best_accuracy or (val_accuracy >= best_accuracy - 0.5 and val_loss < best_loss):
        best_accuracy = val_accuracy
        best_loss = val_loss
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_mapping': class_mapping  # Save class mappings
        }
        torch.save(checkpoint, "best_resnet152_vehicle_damage.pth")
        print(f"New best model saved. Accuracy: {best_accuracy:.2f}%, Loss: {best_loss:.4f}")
        no_improvement_count = 0  # Reset counter when improvement happens
    else:
        no_improvement_count += 1  # Increment counter when no improvement

    # Check if unfreezing should happen (after 5 epochs of no improvement)
    if no_improvement_count >= 3 and unfrozen_idx < len(unfreezing_stages):
        print(f"No improvement for {no_improvement_count} epochs. Unfreezing next layer.")
        for param in unfreezing_stages[unfrozen_idx].parameters():
            param.requires_grad = True
        unfrozen_idx += 1
        no_improvement_count = 0  # Reset counter after unfreezing

        # Reset learning rate to a higher value after unfreezing
        # Use a higher learning rate than current but not higher than initial
        new_lr = max(current_lr * 2, initial_lr)

        # Re-create optimizer with new learning rate
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=new_lr)

        # Reset scheduler with new optimizer
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )

        print(f"Learning rate reset to {new_lr:.8f} after unfreezing new layer")

    # **Save Final Model Checkpoint (After Training Ends)**
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_mapping': class_mapping
    }
    torch.save(checkpoint, "resnet152_vehicle_damage_dropout.pth")
    print("Final model checkpoint saved successfully.")

# Evaluate final performance on validation set
model.eval()
class_correct = list(0. for i in range(len(class_mapping)))
class_total = list(0. for i in range(len(class_mapping)))

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Print per-class accuracy
print("\nClass-wise validation accuracy:")
for i in range(len(class_mapping)):
    class_name = [k for k, v in class_mapping.items() if v == i][0]
    accuracy = 100 * class_correct[i] / class_total[i]
    print(f"{class_name}: {accuracy:.2f}% ({int(class_correct[i])}/{class_total[i]})")

print(f"\nBest validation accuracy: {best_accuracy:.2f}%")

input("Press Enter to exit...")