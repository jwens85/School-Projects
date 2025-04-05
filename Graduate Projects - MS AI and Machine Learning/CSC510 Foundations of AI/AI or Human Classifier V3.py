import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Import the Kaggle Dataset
dataset_path = r"C:\Users\jwens\.cache\kagglehub\datasets\shanegerami\ai-vs-human-text\versions\1\AI_Human.csv"

#Place the dataset into a Pandas DataFrame
df = pd.read_csv(dataset_path)

#Rename columns to match expected names
df.rename(columns={"text": "Text", "generated": "Label"}, inplace=True)

#AI and Human Labels
df["Label"] = df["Label"].astype(int).map({1: "AI", 0: "Human"})

#Handle missing or duplicated values (if any)
df.dropna(inplace=True)
df.drop_duplicates(subset=["Text"], inplace=True)

#Case standardization and stripping of excess spaces
df["Text"] = df["Text"].str.lower().str.strip()

#Print first 5 lines of dataset
print(df.head())
print("\nColumns in dataset:", df.columns)

#TF-IDF text-to-number conversion
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["Text"])
y = df["Label"]

#Split the data into training and test data 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20250223)

#Scikit-Learn Multinomial Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

#Model Evaluator
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Word Frequency Table
word_counts = X_train.sum(axis=0)
words = vectorizer.get_feature_names_out()
word_freq_df = pd.DataFrame(word_counts, columns=words)
print("\nWord Frequency Table (Top 10 words):\n", word_freq_df.iloc[:, :10])  # Show first 10 words

#Input User Text for Classification
def classify_text(input_text):
    transformed_text = vectorizer.transform([input_text])
    prediction = model.predict(transformed_text)
    probabilities = model.predict_proba(transformed_text)  # Get posterior probabilities
    print(f"\nPosterior Probabilities: AI={probabilities[0][1]:.4f}, Human={probabilities[0][0]:.4f}")
    return prediction[0]

#Solicit User Input
user_input = input("\nEnter a sentence to classify as AI or Human-written: ")
print(f"\nPredicted Origin: {classify_text(user_input)}")

#MatPlotLib and Seaborn CM Heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["AI", "Human"], yticklabels=["AI", "Human"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for AI vs. Human Classification")
plt.show()
