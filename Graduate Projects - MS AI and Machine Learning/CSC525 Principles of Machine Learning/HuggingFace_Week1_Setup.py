from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("In 2025, machine learning will", max_length=50)
print(result[0]['generated_text'])
