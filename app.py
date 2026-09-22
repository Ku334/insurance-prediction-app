import numpy as np
import joblib
import streamlit as st

## loading the model and scaler
model  = joblib.load("insurance_model.pkl")
scaler = joblib.load("insurance_scaler.pkl")

## Frontend using streamlit
st.title(" Medical Insurance Cost Predictor")
st.write("Enter your details below to get an estimated insurance cost.")

age      = st.number_input("Age", min_value=18, max_value=100, value=30)
sex      = st.selectbox("Sex", ["Female", "Male"])
bmi      = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker   = st.selectbox("Smoker", ["No", "Yes"])
region   = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Encode inputs
sex_map      = {"Female": 0, "Male": 1}
smoker_map   = {"No": 0, "Yes": 1}
region_map   = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}

if bmi < 18.5:
    bmi_category = 3   # Underweight
elif bmi < 25:
    bmi_category = 0   # Normal
elif bmi < 30:
    bmi_category = 2   # Overweight
else:
    bmi_category = 1   # Obese

smoker_bmi = bmi * smoker_map[smoker]

if st.button("Predict Cost"):
    input_data = np.array([[
        age,
        sex_map[sex],
        bmi,
        children,
        smoker_map[smoker],
        region_map[region],
        bmi_category,
        smoker_bmi
    ]])

    input_scaled = scaler.transform(input_data)
    charges = model.predict(input_scaled)[0]

    st.success(f"Estimated Insurance Cost: $ {charges:.2f} per year")