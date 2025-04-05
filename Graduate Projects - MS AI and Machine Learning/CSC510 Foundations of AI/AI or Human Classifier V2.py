import numpy as nm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Load Kaggle dataset
dataset_path = r"C:\Users\jwens\.cache\kagglehub\datasets\shanegerami\ai-vs-human-text\versions\1\AI_Human.csv"

# Read dataset into Pandas DataFrame
df = pd.read_csv(dataset_path)

#Rename columns to match expected names
df.rename(columns={"text": "Text", "generated": "Label"}, inplace=True)

#Convert Label from (0,1) to ("Human", "AI")
df["Label"] = df["Label"].astype(int).map({1: "AI", 0: "Human"})

#Handle missing or duplicate values (if any)
df.dropna(inplace=True)
df.drop_duplicates(subset=["Text"], inplace=True)

# Standardize text by converting to lowercase and stripping extra spaces
df["Text"] = df["Text"].str.lower().str.strip()

#Preview dataset structure
print(df.head())
print("Columns in dataset:", df.columns)

#Convert text to numerical format using TF-IDF
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["Text"])
y = df["Label"]

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20250223)

#Train Naïve Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

#Evaluate Model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#User Input Classification (Runs Before Plotting)
def classify_text(input_text):
    transformed_text = vectorizer.transform([input_text])
    prediction = model.predict(transformed_text)
    return prediction[0]


# Ask for user input before displaying the confusion matrix
user_input = input("\nEnter a sentence to classify as AI or Human-written: ")
print(f"\nPredicted Origin: {classify_text(user_input)}")

#Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["AI", "Human"], yticklabels=["AI", "Human"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for AI vs. Human Classification")

# Show confusion matrix AFTER user input
plt.show()
