from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open('spam_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    
    # Predict spam or ham
    prediction = model.predict([message])[0]
    probability = max(model.predict_proba([message])[0]) * 100

    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('result.html', message=message, result=result, probability=round(probability, 2))

if __name__ == '__main__':
    app.run(debug=True)
