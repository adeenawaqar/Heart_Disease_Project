from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load Model & Scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        features = [

            float(request.form["Age"]),
            float(request.form["Sex"]),
            float(request.form["Chest_pain_type"]),
            float(request.form["BP"]),
            float(request.form["Cholesterol"]),
            float(request.form["FBS_over_120"]),
            float(request.form["EKG_results"]),
            float(request.form["Max_HR"]),
            float(request.form["Exercise_angina"]),
            float(request.form["ST_depression"]),
            float(request.form["Slope_ST"]),
            float(request.form["Number_of_vessels_fluro"]),
            float(request.form["Thallium"])

        ]

        features = np.array(features).reshape(1, -1)

        # Scaling
        features = scaler.transform(features)

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=str(e))


if __name__ == "__main__":
    app.run(debug=True)
