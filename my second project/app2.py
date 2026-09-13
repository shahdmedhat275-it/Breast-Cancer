import streamlit as st
import pandas as pd
import joblib


# ==============================
# Load Model
# ==============================

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "brca_model.pkl"))

# ==============================
# Page Title
# ==============================

st.title("🩺 Breast Cancer Classification")
st.write("Enter the patient's measurements to predict the classification.")


# ==============================
# Input Fields
# ==============================

radius_mean = st.number_input("Radius Mean", value=14.0)
texture_mean = st.number_input("Texture Mean", value=19.0)
perimeter_mean = st.number_input("Perimeter Mean", value=90.0)
area_mean = st.number_input("Area Mean", value=600.0)

smoothness_mean = st.number_input("Smoothness Mean", value=0.10)
compactness_mean = st.number_input("Compactness Mean", value=0.10)
concavity_mean = st.number_input("Concavity Mean", value=0.08)
concave_pts_mean = st.number_input("Concave Points Mean", value=0.05)
symmetry_mean = st.number_input("Symmetry Mean", value=0.18)
fractal_dim_mean = st.number_input("Fractal Dimension Mean", value=0.06)

radius_se = st.number_input("Radius SE", value=0.25)
texture_se = st.number_input("Texture SE", value=1.0)
perimeter_se = st.number_input("Perimeter SE", value=1.5)
area_se = st.number_input("Area SE", value=20.0)

smoothness_se = st.number_input("Smoothness SE", value=0.007)
compactness_se = st.number_input("Compactness SE", value=0.015)
concavity_se = st.number_input("Concavity SE", value=0.02)
concave_pts_se = st.number_input("Concave Points SE", value=0.01)
symmetry_se = st.number_input("Symmetry SE", value=0.02)
fractal_dim_se = st.number_input("Fractal Dimension SE", value=0.003)

radius_worst = st.number_input("Radius Worst", value=16.0)
texture_worst = st.number_input("Texture Worst", value=25.0)
perimeter_worst = st.number_input("Perimeter Worst", value=105.0)
area_worst = st.number_input("Area Worst", value=800.0)

smoothness_worst = st.number_input("Smoothness Worst", value=0.13)
compactness_worst = st.number_input("Compactness Worst", value=0.25)
concavity_worst = st.number_input("Concavity Worst", value=0.20)
concave_pts_worst = st.number_input("Concave Points Worst", value=0.10)
symmetry_worst = st.number_input("Symmetry Worst", value=0.28)
fractal_dim_worst = st.number_input("Fractal Dimension Worst", value=0.08)


# ==============================
# Create Input DataFrame
# ==============================

input_data = pd.DataFrame([{
    "x.radius_mean": radius_mean,
    "x.texture_mean": texture_mean,
    "x.perimeter_mean": perimeter_mean,
    "x.area_mean": area_mean,
    "x.smoothness_mean": smoothness_mean,
    "x.compactness_mean": compactness_mean,
    "x.concavity_mean": concavity_mean,
    "x.concave_pts_mean": concave_pts_mean,
    "x.symmetry_mean": symmetry_mean,
    "x.fractal_dim_mean": fractal_dim_mean,

    "x.radius_se": radius_se,
    "x.texture_se": texture_se,
    "x.perimeter_se": perimeter_se,
    "x.area_se": area_se,
    "x.smoothness_se": smoothness_se,
    "x.compactness_se": compactness_se,
    "x.concavity_se": concavity_se,
    "x.concave_pts_se": concave_pts_se,
    "x.symmetry_se": symmetry_se,
    "x.fractal_dim_se": fractal_dim_se,

    "x.radius_worst": radius_worst,
    "x.texture_worst": texture_worst,
    "x.perimeter_worst": perimeter_worst,
    "x.area_worst": area_worst,
    "x.smoothness_worst": smoothness_worst,
    "x.compactness_worst": compactness_worst,
    "x.concavity_worst": concavity_worst,
    "x.concave_pts_worst": concave_pts_worst,
    "x.symmetry_worst": symmetry_worst,
    "x.fractal_dim_worst": fractal_dim_worst
}])


# ==============================
# Prediction
# ==============================

if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction")

    if prediction == "M":
        st.error("Prediction: Malignant (M)")
    else:
        st.success("Prediction: Benign (B)")

    st.write("Prediction probabilities:")
