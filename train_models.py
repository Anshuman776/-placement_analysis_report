"""
Train ML models for AKGEC Placement Prediction:
  1. Logistic Regression — Placement classification
  2. Random Forest — Placement classification (better accuracy)
  3. Linear Regression — Salary prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             r2_score, mean_absolute_error, mean_squared_error)
import joblib
import os

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'CGPA', 'DSA_Score', 'WebDev_Score', 'ML_Score',
    'Aptitude_Score', 'Communication_Score',
    'Internships', 'Projects', 'Hackathons'
]


def load_and_prepare_data():
    """Load dataset and prepare features/targets."""
    df = pd.read_csv("dataset/students.csv")
    print(f"📂 Dataset loaded: {df.shape[0]} records")

    # Encode target
    le = LabelEncoder()
    df['Placed'] = le.fit_transform(df['Placement_Status'])  # 1=Placed, 0=Not Placed

    # Features
    X = df[FEATURE_COLS].values
    y_classification = df['Placed'].values

    # For salary prediction, only use placed students
    placed_df = df[df['Placement_Status'] == 'Placed']
    X_salary = placed_df[FEATURE_COLS].values
    y_salary = placed_df['Salary_LPA'].values

    return X, y_classification, X_salary, y_salary, df


def train_logistic_regression(X_train, X_test, y_train, y_test, scaler):
    """Train Logistic Regression for placement classification."""
    print("\n" + "=" * 60)
    print("📈 Model 1: Logistic Regression (Placement Prediction)")
    print("=" * 60)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n   Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))

    # Save model
    model_path = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
    joblib.dump(model, model_path)
    print(f"   💾 Model saved: {model_path}")

    return model, accuracy


def train_random_forest(X_train, X_test, y_train, y_test, scaler):
    """Train Random Forest for placement classification."""
    print("\n" + "=" * 60)
    print("🌲 Model 2: Random Forest (Placement Prediction)")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n   Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    print("   📊 Feature Importance:")
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 50)
        print(f"      {feat:25s} {imp:.4f} {bar}")

    # Save model
    model_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    joblib.dump(model, model_path)
    print(f"\n   💾 Model saved: {model_path}")

    return model, accuracy


def train_salary_predictor(X_train, X_test, y_train, y_test, scaler):
    """Train Linear Regression for salary prediction."""
    print("\n" + "=" * 60)
    print("💰 Model 3: Linear Regression (Salary Prediction)")
    print("=" * 60)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n   R² Score:  {r2:.4f}")
    print(f"   MAE:       {mae:.4f} LPA")
    print(f"   RMSE:      {rmse:.4f} LPA")

    # Coefficients
    print("\n   📊 Feature Coefficients:")
    coefs = sorted(zip(FEATURE_COLS, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in coefs:
        sign = "+" if coef >= 0 else "-"
        print(f"      {feat:25s} {sign}{abs(coef):.4f}")

    # Save model
    model_path = os.path.join(MODELS_DIR, 'salary_predictor.pkl')
    joblib.dump(model, model_path)
    print(f"\n   💾 Model saved: {model_path}")

    return model, r2


def main():
    print("🎓 AKGEC Placement Analysis — Model Training")
    print("=" * 60)

    X, y_class, X_salary, y_salary, df = load_and_prepare_data()

    # ── Placement Classification ────────────────────────────────────
    scaler_class = StandardScaler()
    X_scaled = scaler_class.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    lr_model, lr_acc = train_logistic_regression(X_train, X_test, y_train, y_test, scaler_class)
    rf_model, rf_acc = train_random_forest(X_train, X_test, y_train, y_test, scaler_class)

    # ── Salary Prediction ───────────────────────────────────────────
    scaler_salary = StandardScaler()
    X_salary_scaled = scaler_salary.fit_transform(X_salary)

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_salary_scaled, y_salary, test_size=0.2, random_state=42
    )

    salary_model, salary_r2 = train_salary_predictor(
        X_train_s, X_test_s, y_train_s, y_test_s, scaler_salary
    )

    # Save scalers
    joblib.dump(scaler_class, os.path.join(MODELS_DIR, 'scaler_classification.pkl'))
    joblib.dump(scaler_salary, os.path.join(MODELS_DIR, 'scaler_salary.pkl'))
    print(f"\n   💾 Scalers saved to {MODELS_DIR}/")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📋 TRAINING SUMMARY")
    print("=" * 60)
    print(f"   Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
    print(f"   Random Forest Accuracy:       {rf_acc * 100:.2f}%")
    print(f"   Salary Prediction R² Score:   {salary_r2:.4f}")
    print(f"\n   Best Classification Model: {'Random Forest' if rf_acc > lr_acc else 'Logistic Regression'}")
    print(f"\n   All models saved to '{MODELS_DIR}/' directory ✅")


if __name__ == "__main__":
    main()
