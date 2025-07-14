import os
from keras.datasets import cifar10
from PIL import Image

#CIFAR-10 class names corresponding to labels 0–9
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

#Root output directory
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

#Load CIFAR-10 training data
(X_train, y_train), (_, _) = cifar10.load_data()
y_train = y_train.flatten()

#Save images into class-specific subfolders
print("Saving images into class subfolders...")
for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    #Filter images for this class
    class_images = X_train[y_train == class_idx]

    for i, img_array in enumerate(class_images):
        img = Image.fromarray(img_array)
        filename = f"{i:05d}.png"
        img.save(os.path.join(class_dir, filename))

    print(f"Saved {len(class_images)} images to '{class_name}/'")

print("All classes processed successfully.")
