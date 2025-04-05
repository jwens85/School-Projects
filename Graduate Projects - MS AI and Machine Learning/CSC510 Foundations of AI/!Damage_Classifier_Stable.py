import torch
import torchvision.transforms as transforms
from torchvision.models import resnet152
from PIL import Image

# Load trained model
model_path = "best_resnet152_vehicle_damage.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the
checkpoint = torch.load(model_path, map_location=device, weights_only=True)

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

# Class Mapping Hotfix - Swaps the key:values from model weights to output
class_mapping = {v: k for k, v in class_mapping.items()}

# Load ResNet-152 Without the Pre-Trained Weights
model = resnet152(weights=None)
num_ftrs = model.fc.in_features

# Ensure model architecture matches training
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.1),  # Keep dropout as in training
    torch.nn.Linear(num_ftrs, len(class_mapping))
)

# Load the pre-trained weights
model.load_state_dict(checkpoint['model_state_dict'])

# Move model to the hardware device and set to evaluation mode
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the image loading function
def predict_damage(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    image = transform(image).unsqueeze(0).to(device)

    #Perform Image Inference and Generate a Confidence Score
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        damage_type = class_mapping[predicted.item()]

    print(f"Predicted Damage Type: {damage_type} (Confidence: {confidence.item() * 100:.2f}%)")

# The Main Function and Input Sanitization
if __name__ == "__main__":
    while True:
        image_path = input("Enter the path to the vehicle damage image (or type 'exit' to quit): ").strip()

        if image_path.lower() == 'exit':
            print("Exiting the system. Goodbye!")
            break

        # Remove surrounding quotes if present
        image_path = image_path.strip('"')

        predict_damage(image_path)