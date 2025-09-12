# GlucoTrack: Predicting Diabetes Likelihood from Lifestyle and Clinical Data

Live app: [glucotrack.streamlit.app](https://glucotrack.streamlit.app/)

## Deployment

Run the Streamlit app locally to interact with the champion model (Balanced FFNN) and get diabetes risk predictions.

Prerequisites:
- Python 3.9+
- Dependencies: `streamlit`, `torch`, `pandas`, `numpy`, `scikit-learn`, `joblib`

Quick start:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install --upgrade pip
pip install streamlit torch pandas numpy scikit-learn joblib

# Launch the app
cd yan-cotta
streamlit run app.py
```

Notes:
- The app loads artifacts from `yan-cotta/deployment_artifacts` (model weights and preprocessing objects) generated from the Week 2 and Week 3 notebooks.
- If you update preprocessing or retrain the model, re-run the last cells in those notebooks to refresh the files in `deployment_artifacts` before restarting the app.

### Deploy to Streamlit Cloud

- Ensure these files exist at the repository root:
  - `requirements.txt` (Python dependencies)
  - `runtime.txt` (Python version; not always honored on Cloud which may default to Python 3.13)
- In Streamlit Cloud, set the entrypoint to `yan-cotta/app.py`.
- PyTorch note: Cloud currently uses Python 3.13, so `torch==2.8.0` is pinned in `requirements.txt` because it provides wheels for 3.13.
- After deployment, use “Manage app” → Logs to inspect any missing dependency errors.

## 🧠 Project Overview

-> shared repository with all team members work: <https://github.com/SuperDataScience-Community-Projects/SDS-CP035-gluco-track>


GlucoTrack is a health-focused classification project that aims to predict whether an individual is diabetic, pre-diabetic, or non-diabetic using self-reported survey and biometric data. The dataset includes responses collected by the CDC on physical activity, BMI, mental health, general wellness, and other lifestyle variables.

The project is built around a real-world public health challenge: how do we identify individuals at risk before it’s too late? You’ll analyze patterns in health behavior and demographic variables to develop a predictive model that supports early interventions.

This project is split into two experience tracks:

- 🟢 **Beginner Track** – Build traditional classification models using scikit-learn and deploy a simple Streamlit app.
- 🔴 **Advanced Track** – Design and train a deep learning classifier with PyTorch or TensorFlow, integrate embeddings for categorical data, and deploy via Docker or Hugging Face Spaces.

---

## 🧪 Dataset Summary

- **Source**: CDC Behavioral Risk Factor Surveillance System (BRFSS)
- **Instances**: \~250,000
- **Target Variable**: `Diabetes_binary` (0 = No, 1 = Pre-diabetes or Diabetes)
- **Features**: 20+ features covering BMI, physical activity, mental health days, general health, smoking status, and more
- **Data Type**: Tabular, mostly binary and ordinal features
- **Task Type**: Classification

---

## 🧰 Tools & Libraries

- **ML Libraries**: scikit-learn, XGBoost, LightGBM (Beginner); PyTorch or TensorFlow (Advanced)
- **Visualization**: matplotlib, seaborn
- **Deployment**: Streamlit (Beginner), Docker + Hugging Face Spaces or Flask API (Advanced)
- **Experiment Tracking**: MLflow

---

## 📂 Track Structure

### 🟢 Beginner Track

➡️ [Beginner Scope of Works](./beginner/README.md)
➡️ [Beginner Report Template](./beginner/REPORT.md)
➡️ [Submit Your Work](./beginner/submissions/)

### 🔴 Advanced Track

➡️ [Advanced Scope of Works](./advanced/README.md)
➡️ [Advanced Report Template](./advanced/REPORT.md)
➡️ [Submit Your Work](./advanced/submissions/)

---

## 🚀 Learning Outcomes

By the end of this project, you will:

- Understand health-related feature engineering and data cleaning techniques
- Apply classification models and evaluate their performance with appropriate metrics
* Track experiments and manage reproducibility using MLflow
* Deploy interactive predictive tools for real-world use cases

---

## 👥 Who Should Join?

This project is ideal for:

- Aspiring machine learning engineers and data scientists interested in healthcare analytics
- Beginners seeking to master the end-to-end ML workflow
* Intermediate learners who want to experiment with neural networks and sequence modeling

---

## 📝 Acknowledgements

This project uses open public health data provided by the Centers for Disease Control and Prevention (CDC).
