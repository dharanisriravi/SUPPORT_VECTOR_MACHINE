import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
import re

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Drop missing values
df = df.dropna(subset=['label', 'message'])

# Map labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

df['message'] = df['message'].apply(clean_text)

# Features and target
X = df['message']
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save pipeline
with open('spam_classifier_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Spam classifier model trained and saved successfully!")
print(f"✅ Dataset size: {df.shape[0]} messages")
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
