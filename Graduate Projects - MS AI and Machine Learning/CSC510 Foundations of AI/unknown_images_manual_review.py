import os
import pandas as pd
import shutil
from PIL import Image

# Paths
CSV_PATH = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\data.csv"
IMAGE_DIR = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\image"
UNKNOWN_DIR = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\unknown_images_review"

# Load CSV
df = pd.read_csv(CSV_PATH)
unknown_df = df[df["classes"] == "unknown"].copy()

# Label mapping
label_mapping = {
    1: "no_damage",
    2: "total_loss",
    3: "door_dent",
    4: "bumper_scratch",
    5: "door_scratch",
    6: "glass_shatter",
    7: "tail_lamp",
    8: "head_lamp",
    9: "bumper_dent"
}

print("\n=== IMAGE LABELING TOOL ===")
print("Instructions:")
print("1: No Damage | 2: Total Loss | Other: Enter the damage type number from below")
for key, value in label_mapping.items():
    print(f"{key}: {value}")
print("Type 'exit' to quit at any time.\n")

# Loop through unknown images
for index, row in unknown_df.iterrows():
    img_name = row["image"]
    img_path = os.path.join(IMAGE_DIR, img_name)
    review_path = os.path.join(UNKNOWN_DIR, img_name)

    if os.path.exists(img_path):
        # Open image in a new window (same behavior as before)
        img = Image.open(img_path)
        img.show()

        # Get user input
        while True:
            user_input = input(f"Enter label for {img_name} (1-9): ").strip()

            if user_input.lower() == "exit":
                print("\nExiting and saving progress...")
                df.to_csv(CSV_PATH, index=False)
                exit()

            if user_input.isdigit():
                label_number = int(user_input)
                if label_number in label_mapping:
                    # Update CSV immediately
                    df.loc[df["image"] == img_name, "classes"] = label_mapping[label_number]
                    df.to_csv(CSV_PATH, index=False)  # Save after every entry
                    print(f"Updated {img_name} → {label_mapping[label_number]}")

                    # Remove image from review folder
                    if os.path.exists(review_path):
                        os.remove(review_path)
                        print(f"Removed {img_name} from review folder")

                    break  # Move to the next image
                else:
                    print("Invalid number. Please enter a valid option.")
            else:
                print("Invalid input. Enter a number between 1-9 or type 'exit' to quit.")

print("Labeling complete. CSV file updated successfully!")
