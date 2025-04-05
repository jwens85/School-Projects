import os
import shutil
import pandas as pd

# Paths
CSV_PATH = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\data.csv"
IMAGE_DIR = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\image"
OUTPUT_DIR = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\unknown_images_review"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV and clean file paths
df = pd.read_csv(CSV_PATH)
df["image"] = df["image"].str.replace("image/", "", regex=False)  # Fix incorrect paths

# Save the cleaned CSV (optional)
df.to_csv(CSV_PATH, index=False)

# Filter unknown class
unknown_df = df[df["classes"] == "unknown"]

# Track missing files
missing_files = []

# Copy images to new folder
for img_name in unknown_df["image"]:
    src_path = os.path.join(IMAGE_DIR, img_name)
    dest_path = os.path.join(OUTPUT_DIR, img_name)

    if os.path.exists(src_path):  # Ensure file exists before copying
        shutil.copy(src_path, dest_path)  # Use copy instead of move
    else:
        missing_files.append(src_path)  # Track missing files

# Print results
print(f"Copied {len(unknown_df) - len(missing_files)} unknown images to {OUTPUT_DIR}")

# Show missing files
if missing_files:
    print(f"{len(missing_files)} missing files (paths might be wrong in CSV)")
    print("\nExample missing files:")
    print("\n".join(missing_files[:5]))  # Print first 5 missing files
