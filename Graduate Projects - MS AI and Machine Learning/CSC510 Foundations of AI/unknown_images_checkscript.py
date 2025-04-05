import os
import pandas as pd

CSV_PATH = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\data.csv"
IMAGE_DIR = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\image"

df = pd.read_csv(CSV_PATH)
unknown_df = df[df["classes"] == "unknown"]

missing_files = []

for img_name in unknown_df["image"]:
    img_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.exists(img_path):  # Check if file exists
        missing_files.append(img_path)

print(f"Missing files: {len(missing_files)}")
if missing_files:
    print("\nExample missing files:")
    print("\n".join(missing_files[:5]))  # Print first 5 missing files
