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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Paths
CSV_PATH = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\data.csv"
IMAGE_DIR = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\image"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CSV
df = pd.read_csv(CSV_PATH)
df["image"] = df["image"].str.replace("image/", "", regex=False)

# Encode labels into numeric values
class_mapping = {label: idx for idx, label in enumerate(df["classes"].unique())}
df["label"] = df["classes"].map(class_mapping)

# Print class distribution to check for imbalance
class_distribution = df["classes"].value_counts()
print("Class distribution:")
print(class_distribution)

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Enhanced Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(25),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simpler validation transform
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# **Dataset Class**
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


# Create datasets & dataloaders
train_dataset = VehicleDamageDataset(train_df, transform=transform)
val_dataset = VehicleDamageDataset(val_df, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# **Load Pretrained ResNet-152 Model**
from torchvision.models import ResNet152_Weights

model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, len(class_mapping))
)

# Initially freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Only enable training for the final classifier layer initially
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
print("Class weights:", class_weights)

# **Define Loss Function with Class Weighting**
criterion = nn.CrossEntropyLoss(weight=class_weights)

# **Define Optimizer**
optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-4)

# **Training Parameters**
num_epochs = 50
warmup_epochs = 5
best_accuracy = 0.0
best_loss = float("inf")

# Define schedulers for warmup + cosine annealing
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

# **Training Loop**
for epoch in range(num_epochs):
    # **Progressive Unfreezing Strategy**
    if epoch == 10:
        print("Unfreezing layer4")
        for param in model.layer4.parameters():
            param.requires_grad = True

    if epoch == 15:
        print("Unfreezing layer3")
        for param in model.layer3.parameters():
            param.requires_grad = True

    if epoch == 20:
        print("Unfreezing layer2")
        for param in model.layer2.parameters():
            param.requires_grad = True

    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train

    # **Validation Phase**
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    print(
        f"Epoch {epoch + 1}/{num_epochs} | Loss: {running_loss:.4f} | Accuracy: {accuracy:.2f}% | LR: {current_lr:.6f}")

    # **Save the Best Model (Based on Accuracy and Loss)**
    if accuracy > best_accuracy or (accuracy >= best_accuracy - 0.5 and val_loss < best_loss):
        best_accuracy = accuracy
        best_loss = val_loss
        torch.save(model.state_dict(), "best_resnet152_vehicle_damage.pth")
        print(f"Best model saved | Accuracy: {best_accuracy:.2f}% | Loss: {best_loss:.4f}")

# **Final Model Save**
torch.save(model.state_dict(), "final_resnet152_vehicle_damage.pth")
print("Final model saved successfully!")
