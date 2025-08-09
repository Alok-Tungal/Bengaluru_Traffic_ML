import streamlit as st
import pandas as pd
import joblib

# Load saved RandomForestRegressor model
model = joblib.load("best_random_forest_model.joblib")

st.title("Bengaluru Traffic Prediction App")
st.write("Predicts target value using your trained RandomForestRegressor model.")

# --- Input fields (replace with your exact training columns) ---
# Make sure these match the column names and order used in training X_train
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)
# Add/remove according to your dataset

# Create input dataframe in the same column order as training
input_df = pd.DataFrame([[feature1, feature2, feature3, feature4]],
                        columns=["feature1", "feature2", "feature3", "feature4"])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Value: {prediction:,.2f}")
