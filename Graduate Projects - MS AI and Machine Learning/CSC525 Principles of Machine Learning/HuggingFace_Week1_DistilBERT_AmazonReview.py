from transformers import pipeline
from collections import Counter
import matplotlib.pyplot as plt

# Load the Hugging Face sentiment analysis model
classifier = pipeline("sentiment-analysis")

# Step 1: Prompt user for URL
print("=== Amazon Review Sentiment Analyzer ===\n")
url = input("Enter Amazon product URL (for context only): ").strip()
print(f"\nURL recorded: {url}\n")

# Step 2: Gather reviews
print("Paste each review and press [Enter]. Type DONE when finished.\n")
reviews = []

while True:
    review = input("Review: ").strip()
    if review.upper() == "DONE":
        break
    if review:
        reviews.append(review)

# Step 3: Run sentiment analysis
print("\n=== Sentiment Results ===\n")
labels = []

for review in reviews:
    result = classifier(review)[0]
    print(f"{review}\n→ {result['label']} (confidence: {result['score']:.4f})\n")
    labels.append(result['label'])

# Step 4: Display chart (optional)
if labels:
    sentiment_counts = Counter(labels)
    plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color="skyblue")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print("No reviews entered — nothing to display.")
