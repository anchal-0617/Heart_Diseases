from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Full path to the model file
MODEL_PATH = os.path.join(BASE_DIR, "heart-disease-prediction-knn-model.pkl")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route("/submit", methods=["POST"])
def predict():
    try:
        res = request.get_json()

        values = [
            float(res[key]) for key in [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal"
            ]
        ]

        data = np.array([values])

        prediction = model.predict(data).tolist()

        return jsonify({
            "message": "Data received successfully",
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({
            "message": "Error processing input",
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)