import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Data Preparation
# Define transformations (resize + convert to tensor)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load a simple dataset (e.g., CIFAR-10 or your own)
# Replace with your dataset path if needed
train_data = datasets.FakeData(transform=transform)  # Simulates images
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Step 2: Define the CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # First Conv Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling Layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Second Conv Layer
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Fully Connected Layer
        self.fc2 = nn.Linear(128, 2)  # Output Layer (2 classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Convolution + Activation + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Convolution + Activation + Pooling
        x = x.view(-1, 32 * 16 * 16)  # Flatten the tensor
        x = torch.relu(self.fc1(x))  # Fully Connected Layer
        x = self.fc2(x)  # Output
        return x

# Step 3: Training the CNN
model = SimpleCNN()  # Initialize the model
criterion = nn.CrossEntropyLoss()  # Define loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define optimizer

# Train the model
for epoch in range(5):  # Small number of epochs for simplicity
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Step 4: Testing
test_image, _ = next(iter(train_loader))
output = model(test_image)
print(f"Predicted: {torch.argmax(output, dim=1)}")
