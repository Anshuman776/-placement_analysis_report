from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# 📁 Correct path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("📁 MODELS_DIR:", MODELS_DIR)

# ✅ Load models ONCE only
try:
    logistic_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    random_forest_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    salary_model = joblib.load(os.path.join(MODELS_DIR, 'salary_predictor.pkl'))
    scaler_class = joblib.load(os.path.join(MODELS_DIR, 'scaler_classification.pkl'))
    scaler_salary = joblib.load(os.path.join(MODELS_DIR, 'scaler_salary.pkl'))

    print("✅ Models loaded successfully!")

except Exception as e:
    print("❌ Model loading failed:", e)
    logistic_model = None
    random_forest_model = None
    salary_model = None
    scaler_class = None
    scaler_salary = None

# ─────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({"message": "ML API is running 🚀"})

# ─────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": random_forest_model is not None
    })

# ─────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input"}), 400

        features = np.array([[ 
            float(data.get('cgpa', 0)),
            float(data.get('dsa', 0)),
            float(data.get('webdev', 0)),
            float(data.get('ml', 0)),
            float(data.get('aptitude', 0)),
            float(data.get('communication', 0)),
            int(data.get('internships', 0)),
            int(data.get('projects', 0)),
            int(data.get('hackathons', 0))
        ]])

        # 🚨 fallback if models fail
        if random_forest_model is None:
            return jsonify({
                "placement_probability": 50,
                "placement_status": "Model not loaded",
                "expected_salary": 4
            })

        features_scaled_class = scaler_class.transform(features)
        features_scaled_salary = scaler_salary.transform(features)

        prob = random_forest_model.predict_proba(features_scaled_class)[0][1]
        salary = salary_model.predict(features_scaled_salary)[0]

        return jsonify({
            "placement_probability": round(prob * 100, 2),
            "placement_status": "Likely Placed" if prob > 0.5 else "At Risk",
            "expected_salary": round(float(salary), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
