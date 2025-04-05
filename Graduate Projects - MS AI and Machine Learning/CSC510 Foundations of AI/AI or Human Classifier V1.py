import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#Sample dataset: AI-generated vs. Human-written text
data = {
    'Text': [
        "The proliferation of generative AI models has revolutionized content creation.",
        "I love going to the beach on Sundays with my family.",
        "Neural networks simulate the function of biological neurons to process data.",
        "Can't believe my dog just stole my sandwich. Unbelievable!",
        "Transformers enable AI to process sequential data more effectively.",
        "Just finished my coffee, time to start the day!",
        "Chatbots have improved dramatically with the use of pre-trained models.",
        "That concert was insane! Best night ever!",
        "Large language models require extensive training on diverse datasets.",
        "Man, traffic was so bad today. Took me an hour to get home."
    ],
    'Label': ['AI', 'Human', 'AI', 'Human', 'AI', 'Human', 'AI', 'Human', 'AI', 'Human']
}

df = pd.DataFrame(data)

#Convert text to numerical format using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Text'])
y = df['Label']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Naïve Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

#Evaluate Model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#User Input for Custom Classification
def classify_text(input_text):
    transformed_text = vectorizer.transform([input_text])
    prediction = model.predict(transformed_text)
    return prediction[0]

#Example: User enters a sentence
user_input = input("Enter a sentence to classify as AI or Human-written: ")
print(f"Predicted Origin: {classify_text(user_input)}")
#(Grimoire, 2025)
