import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import joblib

# ------------------------------
# Section 1: Setup and Artifacts
# ------------------------------

st.set_page_config(page_title="GlucoTrack Diabetes Risk Predictor", layout="wide", page_icon="🩺")

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "deployment_artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "balanced_ffnn_model.pth")


class BalancedFFNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super(BalancedFFNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open(os.path.join(ARTIFACTS_DIR, "feature_order.json"), "r") as f:
        feature_order = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "scaled_numericals.json"), "r") as f:
        scaled_numericals = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "categorical_encoded_map.json"), "r") as f:
        categorical_encoded_map = json.load(f)

    age_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "age_encoder.pkl"))
    genhlth_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "genhlth_encoder.pkl"))
    bmi_cat_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "bmi_category_encoder.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "standard_scaler.pkl"))

    input_dim = len(feature_order)
    model = BalancedFFNN(input_dim=input_dim)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model": model,
        "feature_order": feature_order,
        "scaled_numericals": scaled_numericals,
        "categorical_encoded_map": categorical_encoded_map,
        "encoders": {
            "Age": age_encoder,
            "GenHlth": genhlth_encoder,
            "bmi_category": bmi_cat_encoder,
        },
        "scaler": scaler,
    }


artifacts = load_artifacts()

# ------------------------------
# Section 2: UI Design
# ------------------------------

st.title("GlucoTrack: Diabetes Risk Predictor")
st.write(
    "Use this interactive tool to estimate your risk of diabetes based on your health indicators. "
    "Adjust the inputs in the left sidebar and click Predict to see your risk score."
)

# Sidebar Inputs
st.sidebar.header("Your Health Indicators")

# Helper: WHO BMI categorization

def create_bmi_category(bmi: float) -> int:
    if bmi < 18.5:
        return 0
    if bmi < 25.0:
        return 1
    if bmi < 30.0:
        return 2
    if bmi < 35.0:
        return 3
    if bmi < 40.0:
        return 4
    return 5

# Dataset variables (CDC BRFSS style)
# Binary 0/1 indicators used in dataset
binary_fields = [
    ("HighBP", "Have you been told you have high blood pressure?"),
    ("HighChol", "Have you been told you have high cholesterol?"),
    ("CholCheck", "Have you had your cholesterol checked in last 5 years?"),
    ("Smoker", "Have you smoked at least 100 cigarettes in your life?"),
    ("Stroke", "Have you ever had a stroke?"),
    ("HeartDiseaseorAttack", "Coronary heart disease or myocardial infarction?"),
    ("PhysActivity", "Any physical activity in past 30 days?"),
    ("Fruits", "Do you eat fruit 1+ times per day?"),
    ("Veggies", "Do you eat vegetables 1+ times per day?"),
    ("HvyAlcoholConsump", "Heavy alcohol consumption?"),
    ("AnyHealthcare", "Do you have any health care coverage?"),
    ("NoDocbcCost", "Couldn’t see doctor due to cost in past year?"),
    ("Sex", "Sex (1=Male, 0=Female)"),
    ("DiffWalk", "Serious difficulty walking or climbing stairs?"),
]

# Numeric inputs (per Week 2 scaling step)
menthlth = st.sidebar.slider("Mental Health (days of poor mental health in past 30 days)", 0, 30, 0)
physHlth = st.sidebar.slider("Physical Health (days of poor physical health in past 30 days)", 0, 30, 0)

# BMI numeric then categorized to bmi_category
bmi = st.sidebar.slider("Body Mass Index (BMI)", 10.0, 60.0, 27.0)

# Age group (BRFSS uses 14 levels typically; dataset appears 1-13 after cleaning)
age_options = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"
]
age_label = st.sidebar.selectbox("What is your age group?", options=age_options, index=6)

# General health 1 (excellent) to 5 (poor) in dataset
genhlth_options = ["Excellent", "Very good", "Good", "Fair", "Poor"]
# Map to ordinal labels that match fitted encoder's classes order
# We’ll rely on the original encoder learned from dataset; provide same text labels
genhlth_label = st.sidebar.selectbox("How would you rate your general health?", options=genhlth_options, index=2)

# Education (1=Never attended/Kindergarten only, 6=College 4 years or more)
education_labels = [
    "Never attended/Kindergarten",  # 1
    "Grades 1 through 8",           # 2
    "Grades 9 through 11",          # 3
    "Grade 12 or GED",              # 4
    "College 1 to 3 years",         # 5
    "College 4 years or more",      # 6
]
education_idx = st.sidebar.selectbox("What is your highest education level?", options=list(range(1, 7)), index=5, format_func=lambda i: f"{i}: {education_labels[i-1]}")

# Income (1=<$10k ... 8=>$75k)
income_labels = [
    "Less than $10,000",    # 1
    "$10,000 to less than $15,000",  # 2
    "$15,000 to less than $20,000",  # 3
    "$20,000 to less than $25,000",  # 4
    "$25,000 to less than $35,000",  # 5
    "$35,000 to less than $50,000",  # 6
    "$50,000 to less than $75,000",  # 7
    "$75,000 or more",      # 8
]
income_idx = st.sidebar.selectbox("What is your household income?", options=list(range(1, 9)), index=7, format_func=lambda i: f"{i}: {income_labels[i-1]}")

# For each binary field provide radio 0/1
user_inputs = {}
for col, label in binary_fields:
    user_inputs[col] = 1 if st.sidebar.radio(label, ["No", "Yes"], index=0, horizontal=True) == "Yes" else 0

# Assemble initial raw input
raw_input = {
    **user_inputs,
    "BMI": float(bmi),
    "Age": age_label,
    "GenHlth": genhlth_label,
    "MentHlth": int(menthlth),
    "PhysHlth": int(physHlth),
    "Education": int(education_idx),
    "Income": int(income_idx),
}

# ------------------------------
# Section 3: Preprocess & Predict
# ------------------------------

col_predict, _ = st.columns([1, 3])
predict_clicked = col_predict.button("Predict", type="primary")

if predict_clicked:
    with st.spinner("Processing your inputs and running the model..."):
        df = pd.DataFrame([raw_input])

        # BMI category
        df["bmi_category"] = df["BMI"].apply(create_bmi_category)

        # Encode categorical features
        enc = artifacts["encoders"]
        try:
            df["Age_encoded"] = enc["Age"].transform(df["Age"])
        except Exception:
            # If label not seen, map to nearest valid bucket by index
            # Fallback: use index in our age_options which should match training labels ordering
            df["Age_encoded"] = [age_options.index(df.loc[0, "Age"]) if df.loc[0, "Age"] in age_options else 0]
        try:
            df["GenHlth_encoded"] = enc["GenHlth"].transform(df["GenHlth"])
        except Exception:
            df["GenHlth_encoded"] = [genhlth_options.index(df.loc[0, "GenHlth"]) if df.loc[0, "GenHlth"] in genhlth_options else 2]
        try:
            df["bmi_category_encoded"] = enc["bmi_category"].transform(df["bmi_category"])
        except Exception:
            df["bmi_category_encoded"] = df["bmi_category"]

        # Scale numerical features (apply together to match fitted shape)
        scaler = artifacts["scaler"]
        scaled_cols = artifacts["scaled_numericals"]
        try:
            df[scaled_cols] = scaler.transform(df[scaled_cols])
        except Exception:
            # Fallback: compute z-score column-wise using fitted params
            # Assumes order of scaled_cols matches training order used to fit scaler
            if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                for i, col in enumerate(scaled_cols):
                    mean_i = float(scaler.mean_[i])
                    scale_i = float(scaler.scale_[i]) if getattr(scaler, "scale_", None) is not None else 1.0
                    df[col] = (df[col].astype(float) - mean_i) / (scale_i if scale_i != 0 else 1.0)
            else:
                raise

        # Prepare final feature vector in the same order as training
        feature_order = artifacts["feature_order"]
        # Ensure any missing columns exist with zeros
        for col in feature_order:
            if col not in df.columns:
                df[col] = 0
        x = torch.tensor(df[feature_order].values, dtype=torch.float32)

        # Predict
        model = artifacts["model"]
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).item()

    # ------------------------------
    # Section 4: Results
    # ------------------------------
    col1, col2 = st.columns(2)
    col1.metric("Estimated Diabetes Risk", f"{prob*100:.1f}%")

    if prob < 0.25:
        col2.success("✅ Low Risk. Your indicators suggest a low probability of diabetes. Maintain a healthy lifestyle.")
    elif prob < 0.50:
        col2.warning("⚠️ Moderate Risk. Some indicators show elevated risk. Consider discussing these results with a healthcare professional.")
    else:
        col2.error("🚨 High Risk. Indicators suggest a high probability of prediabetes or diabetes. Please consult a healthcare professional.")

st.divider()
st.caption(
    "This is an AI-powered tool and not a substitute for professional medical advice. Please consult a doctor for any health concerns."
)
