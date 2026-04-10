"""
Flask Microservice for ML Prediction API.
Serves placement prediction and salary estimation.
Runs on port 5001.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODELS_DIR = "models"

# ── Load models on startup ──────────────────────────────────────────────
print("🔄 Loading ML models...")
try:
    logistic_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    random_forest_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    salary_model = joblib.load(os.path.join(MODELS_DIR, 'salary_predictor.pkl'))
    scaler_class = joblib.load(os.path.join(MODELS_DIR, 'scaler_classification.pkl'))
    scaler_salary = joblib.load(os.path.join(MODELS_DIR, 'scaler_salary.pkl'))
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("   Please run train_models.py first.")

FEATURE_NAMES = [
    'CGPA', 'DSA_Score', 'WebDev_Score', 'ML_Score',
    'Aptitude_Score', 'Communication_Score',
    'Internships', 'Projects', 'Hackathons'
]

# Skill thresholds for suggestions
SKILL_THRESHOLDS = {
    'DSA_Score': {'min': 60, 'name': 'Data Structures & Algorithms',
                  'tip': 'Practice on LeetCode/GFG. Focus on arrays, trees, graphs, and DP.'},
    'WebDev_Score': {'min': 50, 'name': 'Web Development',
                     'tip': 'Learn React/Node.js. Build full-stack projects and deploy them.'},
    'ML_Score': {'min': 45, 'name': 'Machine Learning',
                 'tip': 'Take Andrew Ng\'s ML course. Practice on Kaggle competitions.'},
    'Aptitude_Score': {'min': 55, 'name': 'Aptitude & Reasoning',
                       'tip': 'Practice verbal, quant, and logical reasoning daily on IndiaBix.'},
    'Communication_Score': {'min': 50, 'name': 'Communication Skills',
                            'tip': 'Join public speaking clubs. Practice mock interviews and GDs.'},
}


def get_skill_suggestions(features_dict):
    """Generate skill improvement suggestions based on input scores."""
    suggestions = []
    for key, config in SKILL_THRESHOLDS.items():
        score = features_dict.get(key, 0)
        if score < config['min']:
            gap = config['min'] - score
            priority = 'high' if gap > 25 else 'medium' if gap > 10 else 'low'
            suggestions.append({
                'skill': config['name'],
                'current_score': score,
                'recommended_score': config['min'],
                'gap': gap,
                'priority': priority,
                'tip': config['tip']
            })

    # Sort by priority (high first)
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    suggestions.sort(key=lambda x: priority_order[x['priority']])

    # Additional suggestions based on CGPA
    if features_dict.get('CGPA', 0) < 7.0:
        suggestions.append({
            'skill': 'Academic Performance (CGPA)',
            'current_score': features_dict.get('CGPA', 0),
            'recommended_score': 7.0,
            'gap': round(7.0 - features_dict.get('CGPA', 0), 2),
            'priority': 'high',
            'tip': 'Focus on improving your CGPA. Many companies have a 7.0 CGPA cutoff.'
        })

    if features_dict.get('Internships', 0) < 1:
        suggestions.append({
            'skill': 'Internship Experience',
            'current_score': features_dict.get('Internships', 0),
            'recommended_score': 1,
            'gap': 1,
            'priority': 'high',
            'tip': 'Apply for internships on Internshala, LinkedIn. Even a 1-month internship helps.'
        })

    return suggestions


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'ML Service is running'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict placement probability and expected salary.
    
    Expected JSON body:
    {
        "cgpa": 8.5,
        "dsa": 75,
        "webdev": 60,
        "ml": 50,
        "aptitude": 70,
        "communication": 65,
        "internships": 2,
        "projects": 4,
        "hackathons": 1
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Extract features
        features_dict = {
            'CGPA': float(data.get('cgpa', 0)),
            'DSA_Score': float(data.get('dsa', 0)),
            'WebDev_Score': float(data.get('webdev', 0)),
            'ML_Score': float(data.get('ml', 0)),
            'Aptitude_Score': float(data.get('aptitude', 0)),
            'Communication_Score': float(data.get('communication', 0)),
            'Internships': int(data.get('internships', 0)),
            'Projects': int(data.get('projects', 0)),
            'Hackathons': int(data.get('hackathons', 0)),
        }

        features = np.array([[features_dict[col] for col in FEATURE_NAMES]])

        # Scale features for classification
        features_scaled_class = scaler_class.transform(features)
        features_scaled_salary = scaler_salary.transform(features)

        # ── Logistic Regression Prediction ──
        lr_prob = logistic_model.predict_proba(features_scaled_class)[0]
        lr_placement_prob = round(float(lr_prob[1]) * 100, 2)

        # ── Random Forest Prediction ──
        rf_prob = random_forest_model.predict_proba(features_scaled_class)[0]
        rf_placement_prob = round(float(rf_prob[1]) * 100, 2)

        # ── Salary Prediction ──
        predicted_salary = float(salary_model.predict(features_scaled_salary)[0])
        predicted_salary = round(max(predicted_salary, 3.0), 2)  # Min 3 LPA

        # ── Skill Suggestions ──
        suggestions = get_skill_suggestions(features_dict)

        # Use the better model (Random Forest) as primary
        primary_prob = rf_placement_prob

        response = {
            'placement_probability': primary_prob,
            'placement_status': 'Likely Placed' if primary_prob >= 50 else 'At Risk',
            'expected_salary': predicted_salary,
            'model_results': {
                'logistic_regression': {
                    'probability': lr_placement_prob,
                    'prediction': 'Placed' if lr_placement_prob >= 50 else 'Not Placed'
                },
                'random_forest': {
                    'probability': rf_placement_prob,
                    'prediction': 'Placed' if rf_placement_prob >= 50 else 'Not Placed'
                }
            },
            'skill_suggestions': suggestions,
            'input_summary': features_dict
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feature-importance', methods=['GET'])
def feature_importance():
    """Return feature importance from Random Forest model."""
    try:
        importances = random_forest_model.feature_importances_
        result = [
            {'feature': name, 'importance': round(float(imp), 4)}
            for name, imp in sorted(zip(FEATURE_NAMES, importances),
                                    key=lambda x: x[1], reverse=True)
        ]
        return jsonify({'feature_importance': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


"""
Flask Microservice for ML Prediction API.
Serves placement prediction and salary estimation.
Runs on port 5001.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODELS_DIR = "models"

# ── Load models on startup ──────────────────────────────────────────────
print("🔄 Loading ML models...")
try:
    logistic_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    random_forest_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    salary_model = joblib.load(os.path.join(MODELS_DIR, 'salary_predictor.pkl'))
    scaler_class = joblib.load(os.path.join(MODELS_DIR, 'scaler_classification.pkl'))
    scaler_salary = joblib.load(os.path.join(MODELS_DIR, 'scaler_salary.pkl'))
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("   Please run train_models.py first.")

FEATURE_NAMES = [
    'CGPA', 'DSA_Score', 'WebDev_Score', 'ML_Score',
    'Aptitude_Score', 'Communication_Score',
    'Internships', 'Projects', 'Hackathons'
]

# Skill thresholds for suggestions
SKILL_THRESHOLDS = {
    'DSA_Score': {'min': 60, 'name': 'Data Structures & Algorithms',
                  'tip': 'Practice on LeetCode/GFG. Focus on arrays, trees, graphs, and DP.'},
    'WebDev_Score': {'min': 50, 'name': 'Web Development',
                     'tip': 'Learn React/Node.js. Build full-stack projects and deploy them.'},
    'ML_Score': {'min': 45, 'name': 'Machine Learning',
                 'tip': 'Take Andrew Ng\'s ML course. Practice on Kaggle competitions.'},
    'Aptitude_Score': {'min': 55, 'name': 'Aptitude & Reasoning',
                       'tip': 'Practice verbal, quant, and logical reasoning daily on IndiaBix.'},
    'Communication_Score': {'min': 50, 'name': 'Communication Skills',
                            'tip': 'Join public speaking clubs. Practice mock interviews and GDs.'},
}


def get_skill_suggestions(features_dict):
    """Generate skill improvement suggestions based on input scores."""
    suggestions = []
    for key, config in SKILL_THRESHOLDS.items():
        score = features_dict.get(key, 0)
        if score < config['min']:
            gap = config['min'] - score
            priority = 'high' if gap > 25 else 'medium' if gap > 10 else 'low'
            suggestions.append({
                'skill': config['name'],
                'current_score': score,
                'recommended_score': config['min'],
                'gap': gap,
                'priority': priority,
                'tip': config['tip']
            })

    # Sort by priority (high first)
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    suggestions.sort(key=lambda x: priority_order[x['priority']])

    # Additional suggestions based on CGPA
    if features_dict.get('CGPA', 0) < 7.0:
        suggestions.append({
            'skill': 'Academic Performance (CGPA)',
            'current_score': features_dict.get('CGPA', 0),
            'recommended_score': 7.0,
            'gap': round(7.0 - features_dict.get('CGPA', 0), 2),
            'priority': 'high',
            'tip': 'Focus on improving your CGPA. Many companies have a 7.0 CGPA cutoff.'
        })

    if features_dict.get('Internships', 0) < 1:
        suggestions.append({
            'skill': 'Internship Experience',
            'current_score': features_dict.get('Internships', 0),
            'recommended_score': 1,
            'gap': 1,
            'priority': 'high',
            'tip': 'Apply for internships on Internshala, LinkedIn. Even a 1-month internship helps.'
        })

    return suggestions


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'ML Service is running'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict placement probability and expected salary.
    
    Expected JSON body:
    {
        "cgpa": 8.5,
        "dsa": 75,
        "webdev": 60,
        "ml": 50,
        "aptitude": 70,
        "communication": 65,
        "internships": 2,
        "projects": 4,
        "hackathons": 1
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Extract features
        features_dict = {
            'CGPA': float(data.get('cgpa', 0)),
            'DSA_Score': float(data.get('dsa', 0)),
            'WebDev_Score': float(data.get('webdev', 0)),
            'ML_Score': float(data.get('ml', 0)),
            'Aptitude_Score': float(data.get('aptitude', 0)),
            'Communication_Score': float(data.get('communication', 0)),
            'Internships': int(data.get('internships', 0)),
            'Projects': int(data.get('projects', 0)),
            'Hackathons': int(data.get('hackathons', 0)),
        }

        features = np.array([[features_dict[col] for col in FEATURE_NAMES]])

        # Scale features for classification
        features_scaled_class = scaler_class.transform(features)
        features_scaled_salary = scaler_salary.transform(features)

        # ── Logistic Regression Prediction ──
        lr_prob = logistic_model.predict_proba(features_scaled_class)[0]
        lr_placement_prob = round(float(lr_prob[1]) * 100, 2)

        # ── Random Forest Prediction ──
        rf_prob = random_forest_model.predict_proba(features_scaled_class)[0]
        rf_placement_prob = round(float(rf_prob[1]) * 100, 2)

        # ── Salary Prediction ──
        predicted_salary = float(salary_model.predict(features_scaled_salary)[0])
        predicted_salary = round(max(predicted_salary, 3.0), 2)  # Min 3 LPA

        # ── Skill Suggestions ──
        suggestions = get_skill_suggestions(features_dict)

        # Use the better model (Random Forest) as primary
        primary_prob = rf_placement_prob

        response = {
            'placement_probability': primary_prob,
            'placement_status': 'Likely Placed' if primary_prob >= 50 else 'At Risk',
            'expected_salary': predicted_salary,
            'model_results': {
                'logistic_regression': {
                    'probability': lr_placement_prob,
                    'prediction': 'Placed' if lr_placement_prob >= 50 else 'Not Placed'
                },
                'random_forest': {
                    'probability': rf_placement_prob,
                    'prediction': 'Placed' if rf_placement_prob >= 50 else 'Not Placed'
                }
            },
            'skill_suggestions': suggestions,
            'input_summary': features_dict
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feature-importance', methods=['GET'])
def feature_importance():
    """Return feature importance from Random Forest model."""
    try:
        importances = random_forest_model.feature_importances_
        result = [
            {'feature': name, 'importance': round(float(imp), 4)}
            for name, imp in sorted(zip(FEATURE_NAMES, importances),
                                    key=lambda x: x[1], reverse=True)
        ]
        return jsonify({'feature_importance': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
