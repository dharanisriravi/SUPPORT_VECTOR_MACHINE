import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Sample manual dataset
data = {
    "label": ["spam", "ham", "ham", "spam", "ham", "spam", "ham", "spam"],
    "text": [
        "Win a $1000 cash prize now!",
        "Hey, are we still meeting for lunch?",
        "Don't forget the meeting at 3 PM",
        "Congratulations! You won a free vacation to Bahamas",
        "Can you send me the report?",
        "Exclusive offer just for you, click now!",
        "I'll see you at the party tonight",
        "Get cheap meds without prescription"
    ]
}

df = pd.DataFrame(data)

# Remove missing data
df.dropna(subset=["label", "text"], inplace=True)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "svm_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model & vectorizer saved successfully!")
