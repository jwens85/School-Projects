from transformers import pipeline

# Load a sentiment analysis pipeline using DistilBERT
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

# Example inputs for classification
examples = [
    "This is an example sentence to see if the model is working",
    "I wonder if performance will be comparable to web-based models",
    "We'll see how it goes",
    "This is my first time using HuggingFace transformers library"
]
print(classifier.model.name_or_path)
# Evaluate each sentence
for sentence in examples:
    result = classifier(sentence)
    print(f"{sentence}\n→ {result[0]['label']} ({result[0]['score']:.4f})\n")
