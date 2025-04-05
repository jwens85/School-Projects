import torch
import torchvision.transforms as transforms
from torchvision.models import resnet152
from PIL import Image

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the checkpoint
checkpoint_path = "best_resnet152_vehicle_damage.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

# Load class mapping from checkpoint
class_mapping = checkpoint.get('class_mapping', {
    0: "total_loss",
    1: "head_lamp",
    2: "door_scratch",
    3: "no_damage",
    4: "glass_shatter",
    5: "tail_lamp",
    6: "bumper_dent",
    7: "door_dent",
    8: "bumper_scratch"
})
print("Loaded class mapping:", class_mapping)

# Load ResNet-152 model (architecture must match training)
model = resnet152(weights=None)  # No pre-trained weights, since we fine-tuned our own model
num_ftrs = model.fc.in_features

# Ensure model architecture matches training
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.1),  # Keep dropout as in training
    torch.nn.Linear(num_ftrs, len(class_mapping))
)

# Load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])

# Move model to device and set to evaluation mode
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_damage(image_path):
    """Predict damage type from an image file."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        damage_type = class_mapping[predicted.item()]

    print(f"Predicted Damage Type: {damage_type}")


if __name__ == "__main__":
    image_path = input("Enter the path to the vehicle damage image: ").strip()
    predict_damage(image_path)
