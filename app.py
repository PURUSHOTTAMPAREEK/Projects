from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


model = pickle.load(open(r"C:\Users\user\Downloads\model.pkl", "rb"))


@app.route('/')
def home():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    
    age = int(request.form['age'])
    temperature = float(request.form['temperature'])
    cough = int(request.form['cough'])
    body_pain = int(request.form['body_pain'])
    fatigue = int(request.form['fatigue'])

    
    data = np.array([[age, temperature, cough, body_pain, fatigue]])
    
    
    prediction = model.predict(data)[0]


    result = "Positive for COVID" if prediction == 1 else "Negative for COVID"
    return render_template("result.html", prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

