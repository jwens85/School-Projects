# import os
# import pandas as pd
#
# # Absolute path to the CSV file
# CSV_PATH = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\data.csv"
#
# # Check if file exists
# if not os.path.exists(CSV_PATH):
#     print(f"ERROR: CSV file not found at {CSV_PATH}")
# else:
#     print("CSV file found successfully!")
#
#     # Try loading the CSV
#     try:
#         df = pd.read_csv(CSV_PATH)
#         print("CSV loaded successfully!")
#         print(df.head())  # Show first few rows
#     except Exception as e:
#         print(f"Error loading CSV: {e}")
#___________________________________________________
# import os
# import pandas as pd
# from PIL import Image
#
# # Define paths
# CSV_PATH = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\data.csv"
# IMAGE_DIR = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\image"
#
# # Load CSV
# try:
#     df = pd.read_csv(CSV_PATH)
#     print("CSV file loaded successfully!")
# except Exception as e:
#     print(f"Error loading CSV: {e}")
#     exit()
#
# # Remove "image/" prefix if it exists
# df["image"] = df["image"].str.replace("image/", "", regex=False)  # Fix filenames
#
# # Construct correct file paths
# df["image_path"] = df["image"].apply(lambda x: os.path.join(IMAGE_DIR, x))
#
# # Check if images exist
# missing_files = [path for path in df["image_path"] if not os.path.exists(path)]
#
# if missing_files:
#     print(f"Warning: {len(missing_files)} image files not found!")
#     print("Example missing file:", missing_files[0])
# else:
#     print("All images found successfully!")
#
# # Try opening a sample image
# sample_path = df["image_path"].iloc[0]
# try:
#     img = Image.open(sample_path)
#     img.show()  # Opens the image
#     print("Sample image loaded successfully:", sample_path)
# except Exception as e:
#     print("Error opening image:", e)
#___________________________________________________


