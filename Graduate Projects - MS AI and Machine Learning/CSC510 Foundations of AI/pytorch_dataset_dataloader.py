# import torch
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# # Paths
# CSV_PATH = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\data.csv"
# IMAGE_DIR = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\image"
#
# # Load CSV
# df = pd.read_csv(CSV_PATH)
# df["image"] = df["image"].str.replace("image/", "", regex=False)
#
# # Encode labels
# class_mapping = {label: idx for idx, label in enumerate(df["classes"].unique())}
# df["label"] = df["classes"].map(class_mapping)
#
# # Split dataset
# train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
#
# # Define transformations
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),  # Resize images
#     transforms.ToTensor(),  # Convert image to tensor
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
# ])
#
#
# # PyTorch Dataset Class
# class VehicleDamageDataset(Dataset):
#     def __init__(self, dataframe, transform=None):
#         self.dataframe = dataframe
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.dataframe)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(IMAGE_DIR, self.dataframe.iloc[idx]["image"])
#         label = self.dataframe.iloc[idx]["label"]
#
#         # Open image
#         image = Image.open(img_path).convert("RGB")
#
#         if self.transform:
#             image = self.transform(image)
#
#         return image, torch.tensor(label, dtype=torch.long)
#
#
# # Create train and validation datasets
# train_dataset = VehicleDamageDataset(train_df, transform=transform)
# val_dataset = VehicleDamageDataset(val_df, transform=transform)
#
# # Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#
# # Test: Load a batch
# images, labels = next(iter(train_loader))
# print(f"Train Batch Shape: {images.shape}, Labels: {labels}")
#__________________________________________________
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
CSV_PATH = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\data.csv"
IMAGE_DIR = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\image"

# Load CSV
df = pd.read_csv(CSV_PATH)
df["image"] = df["image"].str.replace("image/", "", regex=False)

# Encode labels
class_mapping = {label: idx for idx, label in enumerate(df["classes"].unique())}
df["label"] = df["classes"].map(class_mapping)

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])


# PyTorch Dataset Class
class VehicleDamageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(IMAGE_DIR, self.dataframe.iloc[idx]["image"])
        label = self.dataframe.iloc[idx]["label"]

        # Open image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# Create train and validation datasets
train_dataset = VehicleDamageDataset(train_df, transform=transform)
val_dataset = VehicleDamageDataset(val_df, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Test: Load a batch
images, labels = next(iter(train_loader))
print(f"Train Batch Shape: {images.shape}, Labels: {labels}")
