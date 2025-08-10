from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    if not message.strip():
        return render_template("result.html", prediction="Please enter some text.")

    # Transform & predict
    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]

    return render_template("result.html", prediction=prediction.capitalize())

if __name__ == "__main__":
    app.run(debug=True)
