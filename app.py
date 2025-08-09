import streamlit as st
import pandas as pd
import joblib

# Load the trained RandomForestRegressor model
model = joblib.load("best_random_forest_model.pkl")

st.title("Loan Amount Prediction App")
st.write("This app predicts the loan amount using a trained RandomForestRegressor model.")

# Example input fields (replace with your dataset's features)
person_age = st.number_input("Person Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Person Income", min_value=0, value=50000)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input("Loan Amount", min_value=500, value=10000)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0)
cb_person_default_on_file = st.selectbox("Default on File", ["Y", "N"])
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)

# Convert inputs to DataFrame (must match training columns)
input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_home_ownership': [person_home_ownership],
    'loan_intent': [loan_intent],
    'loan_grade': [loan_grade],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'cb_person_default_on_file': [cb_person_default_on_file],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length]
})

# TODO: Apply same encoding/scaling as training
# Example: If you used LabelEncoder or OneHotEncoder, load and apply it here

if st.button("Predict Loan Amount"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Loan Amount: {prediction[0]:,.2f}")
