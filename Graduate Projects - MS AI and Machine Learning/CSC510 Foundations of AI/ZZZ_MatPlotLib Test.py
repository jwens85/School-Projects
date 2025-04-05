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
import matplotlib.pyplot as plt

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

# Define layers for progressive unfreezing - from deep to shallow
unfreezing_stages = [
    model.layer4,  # Unfreeze first layer
    model.layer3,  # Unfreeze second layer
    model.layer2,  # Unfreeze third layer
    model.layer1,  # Unfreeze first layer
    model.bn1,  # Unfreeze bn1
    model.conv1  # Unfreeze conv1
]

# Unfreeze the FC layer and the first three layers at the start of training
unfrozen_idx = 3  # Start with layers 4, 3, and 2 unfrozen
unfreezing_stages.append(model.fc)
for i in range(unfrozen_idx + 1):  # unfrozen_idx +1 to include model.fc
    for param in unfreezing_stages[i].parameters():
        param.requires_grad = True

# Move model to the appropriate device
model = model.to(device)

# Calculate Class Weights for Imbalanced Dataset
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

# Define Loss Function with Class Weighting
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.00003)

# Dynamic Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=1, factor=0.5, min_lr=1e-6
)

# Training Loop with Balanced Model Saving
num_epochs = 50
best_accuracy = 0.0
best_loss = float("inf")

# Variables for progressive unfreezing
no_improvement_count = 0
previous_best = 0.0

# For visualization tracking
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []
learning_rates = []
layer_unfreeze_epochs = []
best_model_epochs = []

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

    # Check for learning rate changes
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss / len(val_loader))
    new_lr = optimizer.param_groups[0]["lr"]

    if new_lr < old_lr:
        print(f"~~~Learning rate decreased: {old_lr:.8f} → {new_lr:.8f}~~~")

    # Print epoch statistics
    print(
        f"\nEpoch {epoch + 1}: Training Loss = {running_loss / len(train_loader):.4f}, Training Accuracy = {train_accuracy:.2f}%")
    print(
        f"Epoch {epoch + 1}: Validation Loss = {val_loss / len(val_loader):.4f}, Validation Accuracy = {val_accuracy:.2f}%, Learning Rate = {new_lr:.8f}")

    # Store metrics for visualization
    training_loss.append(running_loss / len(train_loader))
    validation_loss.append(val_loss / len(val_loader))
    training_accuracy.append(train_accuracy)
    validation_accuracy.append(val_accuracy)
    learning_rates.append(new_lr)

    # **Save Best Model Checkpoint (Fixed)**
    if val_accuracy > best_accuracy or (val_accuracy >= best_accuracy - 0.5 and val_loss < best_loss):
        best_accuracy = val_accuracy
        best_loss = val_loss
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_mapping': class_mapping  # Save class mappings
        }
        torch.save(checkpoint, "best_resnet152_vehicle_damage_candidate.pth")
        print(f"!!!New best model saved. Accuracy: {best_accuracy:.2f}%, Loss: {best_loss:.4f}!!!")
        no_improvement_count = 0  # Reset counter when improvement happens
        best_model_epochs.append(epoch)  # Track when best models were saved
    else:
        no_improvement_count += 1  # Increment counter when no improvement

    # Start to unfreeze layers after x epochs of no improvement
    if no_improvement_count >= 3 and unfrozen_idx < len(unfreezing_stages):
        print(f"***No improvement for {no_improvement_count} epochs. Unfreezing next layer.***")
        for param in unfreezing_stages[unfrozen_idx].parameters():
            param.requires_grad = True
        unfrozen_idx += 1
        no_improvement_count = 0  # Reset counter after unfreezing

        # Track when layers were unfrozen
        layer_unfreeze_epochs.append(epoch)

        # Re-create optimizer to properly register newly unfrozen parameters
        prev_lr = optimizer.param_groups[0]["lr"]
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=prev_lr)

        # KEY FIX: Re-create scheduler when optimizer is re-created
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=1, factor=0.9, min_lr=1e-6
        )

# **Save Final Model Checkpoint (After Training Ends)**
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_mapping': class_mapping
}
torch.save(checkpoint, "resnet152_vehicle_damage_dropout.pth")

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

# ------- IMPROVED VISUALIZATION CODE STARTS HERE -------

# Set a modern style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette
main_color = '#1f77b4'  # blue
secondary_color = '#ff7f0e'  # orange
highlight_color = '#2ca02c'  # green
accent_color = '#9467bd'  # purple
marker_color = '#d62728'  # red

# Visualization of training performance with improved aesthetics and better spacing
plt.figure(figsize=(15, 18))  # Taller figure for better spacing
epochs = range(1, len(training_loss) + 1)

# Configure shared properties
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

# Plot 1: Training and Validation Loss
ax1 = plt.subplot(3, 1, 1)
plt.plot(epochs, training_loss, color=main_color, linewidth=2, label='Training Loss')
plt.plot(epochs, validation_loss, color=secondary_color, linewidth=2, label='Validation Loss')
plt.title('Training and Validation Loss', fontweight='bold', pad=15)
plt.ylabel('Loss', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.3)
ax1.set_facecolor('#f8f9fa')  # Light background color

# Mark where layers were unfrozen with better spacing
for i, epoch in enumerate(layer_unfreeze_epochs):
    plt.axvline(x=epoch + 1, color=highlight_color, linestyle='--', alpha=0.6, linewidth=1.5)
    # Adjust vertical position to avoid overlapping
    vert_pos = max(max(training_loss), max(validation_loss)) * (0.9 - i * 0.1)
    plt.text(epoch + 1.5, vert_pos, 'Layer Unfrozen',
             rotation=90, color=highlight_color, fontsize=10, alpha=0.8)

# Mark where best models were saved with improved markers and better spacing
for i, epoch in enumerate(best_model_epochs):
    plt.plot(epoch + 1, validation_loss[epoch], marker='*', markersize=12,
             color=marker_color, markeredgecolor='white', markeredgewidth=1)

    # Alternate text positions to avoid overlapping
    if i % 2 == 0:
        vert_pos = validation_loss[epoch] * 1.1
        vert_align = 'bottom'
    else:
        vert_pos = validation_loss[epoch] * 0.9
        vert_align = 'top'

    plt.text(epoch + 1.5, vert_pos, 'Best Model',
             color=marker_color, fontweight='bold', fontsize=10,
             verticalalignment=vert_align, horizontalalignment='left')

plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')

# Plot 2: Training and Validation Accuracy
ax2 = plt.subplot(3, 1, 2)
plt.plot(epochs, training_accuracy, color=main_color, linewidth=2, label='Training Accuracy')
plt.plot(epochs, validation_accuracy, color=secondary_color, linewidth=2, label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontweight='bold', pad=15)
plt.ylabel('Accuracy (%)', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.3)
ax2.set_facecolor('#f8f9fa')

# Mark where layers were unfrozen with better spacing
for i, epoch in enumerate(layer_unfreeze_epochs):
    plt.axvline(x=epoch + 1, color=highlight_color, linestyle='--', alpha=0.6, linewidth=1.5)
    # Adjust vertical position to avoid overlapping
    vert_pos = min(min(training_accuracy), min(validation_accuracy)) * (1.1 + i * 0.1)
    plt.text(epoch + 1.5, vert_pos, 'Layer Unfrozen',
             rotation=90, color=highlight_color, fontsize=10, alpha=0.8)

# Mark where best models were saved with better spacing
for i, epoch in enumerate(best_model_epochs):
    plt.plot(epoch + 1, validation_accuracy[epoch], marker='*', markersize=12,
             color=marker_color, markeredgecolor='white', markeredgewidth=1)

    # Alternate text positions to avoid overlapping
    if i % 2 == 0:
        vert_pos = validation_accuracy[epoch] * 0.95
        vert_align = 'top'
    else:
        vert_pos = validation_accuracy[epoch] * 1.05
        vert_align = 'bottom'

    plt.text(epoch + 1.5, vert_pos, 'Best Model',
             color=marker_color, fontweight='bold', fontsize=10,
             verticalalignment=vert_align, horizontalalignment='left')

plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')

# Plot 3: Learning Rate
ax3 = plt.subplot(3, 1, 3)
plt.semilogy(epochs, learning_rates, color=main_color, linewidth=2, label='Learning Rate')
plt.title('Learning Rate over Time', fontweight='bold', pad=15)
plt.xlabel('Epochs', fontweight='bold')
plt.ylabel('Learning Rate (log scale)', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.3)
ax3.set_facecolor('#f8f9fa')

# Mark where learning rate decreased with better spacing
for i, idx in enumerate([i for i in range(1, len(learning_rates)) if learning_rates[i] < learning_rates[i - 1]]):
    plt.plot(idx + 1, learning_rates[idx], marker='o', markersize=8,
             color=marker_color, markeredgecolor='white', markeredgewidth=1)

    # Alternate text positions for better readability
    if i % 2 == 0:
        vert_pos = learning_rates[idx] * 0.7
        vert_align = 'top'
    else:
        vert_pos = learning_rates[idx] * 1.3
        vert_align = 'bottom'

    plt.text(idx + 1.5, vert_pos, 'LR Decreased',
             color=marker_color, fontsize=10,
             verticalalignment=vert_align, horizontalalignment='left')

# Mark where layers were unfrozen with better spacing
for i, epoch in enumerate(layer_unfreeze_epochs):
    plt.axvline(x=epoch + 1, color=highlight_color, linestyle='--', alpha=0.6, linewidth=1.5)

    # Stagger text vertically for better readability
    multiplier = 2 + i * 0.5
    plt.text(epoch + 1.5, min(learning_rates) * multiplier, 'Layer Unfrozen',
             rotation=90, color=highlight_color, fontsize=10, alpha=0.8)

plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')

# Add a super title with better styling
plt.suptitle('ResNet152 Vehicle Damage Classification Training Performance',
             fontsize=18, fontweight='bold', y=0.98)

# Adjust layout and save with higher quality
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(hspace=0.4)  # Add more space between subplots
plt.savefig('training_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Additional visualization: Class-wise performance with improved styling
# Reverse color scheme (red for better performance, blue for worse)
class_names = list(class_mapping.keys())
class_accuracies = [100 * class_correct[i] / class_total[i] for i in range(len(class_mapping))]

# Create a colormap with blue to red (cool to warm) for low to high accuracy
from matplotlib.colors import LinearSegmentedColormap

custom_cmap = LinearSegmentedColormap.from_list('blue_to_red', ['#1f77b4', '#d3d3d3', '#d62728'])
norm = plt.Normalize(min(class_accuracies), max(class_accuracies))
colors = custom_cmap(norm(class_accuracies))

plt.figure(figsize=(14, 7))
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# Create bars with new color scheme
bars = plt.bar(class_names, class_accuracies, color=colors,
               edgecolor='white', linewidth=0.8, alpha=0.85)

plt.title('Class-wise Validation Accuracy', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Damage Class', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold', labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.ylim(0, max(class_accuracies) * 1.15)  # Give more headroom for annotations
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Add accuracy values on top of bars with better styling
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 2,
             f'{height:.1f}%', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color='#333333')

# Add a color bar to show the accuracy scale with the new color scheme
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Accuracy Scale (%) - Higher is Better', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('class_performance.png', dpi=300, facecolor='white')
plt.show()