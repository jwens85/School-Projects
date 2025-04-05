import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import train_test_split

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

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# **Data Augmentation**
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet152 expects 224x224 images
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random cropping (80-100% of the image)
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images 50% of the time
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Random transformations
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Random brightness & contrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ResNet normalization
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
val_dataset = VehicleDamageDataset(val_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# **Load Pretrained ResNet-152 Model**
from torchvision.models import ResNet152_Weights

model = models.resnet152(weights=ResNet152_Weights.DEFAULT)  # Load ResNet152
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_mapping))  # Replace last layer

# **Fine-tune the last 2 ResNet blocks**
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers first
for param in model.layer3.parameters():  # Unfreeze the last two ResNet blocks
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

model = model.to(device)

# **Define Loss Function & Optimizer**
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00003)  # Lower LR for fine-tuning
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# **Training Loop**
num_epochs = 50  # Train for 50 epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # **Validation**
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    scheduler.step(accuracy)  # Reduce LR when accuracy stops improving
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

# **Save Model for Future Use**
torch.save(model.state_dict(), "resnet152_vehicle_damage_finetuned.pth")
print("Model saved successfully!")
