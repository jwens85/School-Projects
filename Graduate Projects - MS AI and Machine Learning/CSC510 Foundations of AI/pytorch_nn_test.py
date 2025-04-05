import torch
import torch.nn as nn
import torch.optim as optim

# A simple feedforward network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Instantiate the network and move it to GPU
model = SimpleNet().to("cuda")
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate random input and target tensors on the GPU
inputs = torch.randn(5, 10).to("cuda")
targets = torch.randn(5, 1).to("cuda")

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, targets)

# Backward pass and optimization
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")

# This code was written by an LLM
# Grimoire. (2025). Conversation about PyTorch, TensorFlow, and AI projects. OpenAI.