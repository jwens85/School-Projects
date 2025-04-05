import torch
import torchvision.models as models
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the ResNet-50 model
model = models.resnet50(weights="IMAGENET1K_V1").to(device)
model.eval()

# Generate random input tensor (3 color channels, 224x224 image)
input_tensor = torch.randn(1, 3, 224, 224).to(device)

# Run inference and measure time
start_time = time.time()
output = model(input_tensor)
end_time = time.time()

# Display results
print(f"Output shape: {output.shape}")
print(f"Inference time: {end_time - start_time:.4f} seconds")
