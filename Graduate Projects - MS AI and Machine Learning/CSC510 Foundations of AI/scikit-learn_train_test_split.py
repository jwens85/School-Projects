import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
CSV_PATH = r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\CSC510 Portfolio Project\data\data.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Remove "image/" prefix if needed
df["image"] = df["image"].str.replace("image/", "", regex=False)

# Encode labels into numeric values
class_mapping = {label: idx for idx, label in enumerate(df["classes"].unique())}
df["label"] = df["classes"].map(class_mapping)

# Split dataset into 80% training, 20% validation
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Print dataset sizes
print(f"Total dataset size: {len(df)}")
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Label Mapping: {class_mapping}")
