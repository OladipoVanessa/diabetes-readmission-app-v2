import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Load model
model = xgb.Booster()
model.load_model("safe_model.json")

st.title("Diabetes 30-Day Readmission Predictor")

st.write("This demo app predicts 30-day readmission risk for diabetic patients at discharge.")

# ---- Intake Form (Simple 10–12 Questions) ----
discharge_destination = st.selectbox(
    "Discharge Destination",
    ["Home", "Skilled Nursing Facility", "Rehab", "Other"]
)

prior_inpatient_visits = st.selectbox(
    "Number of prior inpatient visits",
    ["0", "1–2", "3 or more"]
)

primary_diagnosis = st.selectbox(
    "Primary Diagnosis Type",
    ["Uncomplicated Diabetes", "Complicated Diabetes", "Comorbidity-Heavy Case"]
)

comorbidities = st.selectbox(
    "Comorbidity Burden",
    ["Low", "Moderate", "High"]
)

length_of_stay = st.selectbox(
    "Length of Stay",
    ["1–2 days", "3–4 days", "5+ days"]
)

med_change = st.selectbox(
    "Medication Change During Visit",
    ["No change", "New medication added", "Major change"]
)

a1c = st.selectbox(
    "Most Recent A1C Result",
    ["<7", "7–9", ">9"]
)

num_medications = st.selectbox(
    "Number of Medications",
    ["0–4", "5–9", "10+"]
)

intensity_of_care = st.selectbox(
    "Intensity of Care",
    ["Low", "Moderate", "High"]
)

total_visits = st.selectbox(
    "Total Hospital Visits in Past Year",
    ["0–1", "2–3", "4+"]
)

# ---- Backend Mapping Layer ----
def map_features():
    mapping = {
        "discharge_destination": {
            "Home": 0,
            "Skilled Nursing Facility": 1,
            "Rehab": 2,
            "Other": 3
        },
        "prior_inpatient_visits": {
            "0": 0,
            "1–2": 1,
            "3 or more": 2
        },
        "primary_diagnosis": {
            "Uncomplicated Diabetes": 0,
            "Complicated Diabetes": 1,
            "Comorbidity-Heavy Case": 2
        },
        "comorbidities": {
            "Low": 0,
            "Moderate": 1,
            "High": 2
        },
        "length_of_stay": {
            "1–2 days": 0,
            "3–4 days": 1,
            "5+ days": 2
        },
        "med_change": {
            "No change": 0,
            "New medication added": 1,
            "Major change": 2
        },
        "a1c": {
            "<7": 0,
            "7–9": 1,
            ">9": 2
        },
        "num_medications": {
            "0–4": 0,
            "5–9": 1,
            "10+": 2
        },
        "intensity_of_care": {
            "Low": 0,
            "Moderate": 1,
            "High": 2
        },
        "total_visits": {
            "0–1": 0,
            "2–3": 1,
            "4+": 2
        }
    }

    return [
        mapping["discharge_destination"][discharge_destination],
        mapping["prior_inpatient_visits"][prior_inpatient_visits],
        mapping["primary_diagnosis"][primary_diagnosis],
        mapping["comorbidities"][comorbidities],
        mapping["length_of_stay"][length_of_stay],
        mapping["med_change"][med_change],
        mapping["a1c"][a1c],
        mapping["num_medications"][num_medications],
        mapping["intensity_of_care"][intensity_of_care],
        mapping["total_visits"][total_visits],
    ]

# ---- Prediction Button ----
if st.button("Predict Readmission Risk"):
    input_features = np.array(map_features()).reshape(1, -1)
    dmatrix = xgb.DMatrix(input_features)

    prediction = model.predict(dmatrix)[0]

    st.subheader("Predicted Risk Score")
    st.write(f"{prediction:.2f}")

    if prediction < 0.33:
        st.success("Low Risk")
    elif prediction < 0.66:
        st.warning("Moderate Risk")
    else:
        st.error("High Risk")

