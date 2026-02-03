import streamlit as st
import pandas as pd
import pickle

# ------------------ Page Config ------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📉 Customer Churn Prediction")
st.write("Enter customer details to predict churn probability")

# ------------------ Load Model & Data ------------------
model = pickle.load(open("model.sav", "rb"))
df_1 = pd.read_csv("first_telc.csv")

# ------------------ User Inputs ------------------
SeniorCitizen = st.selectbox("Senior Citizen", ["0", "1"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

gender = st.selectbox("Gender", ["Male", "Female"])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
tenure = st.number_input("Tenure (months)", min_value=0)

# ------------------ Prediction ------------------
if st.button("Predict Churn"):

    data = [[SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner,
             Dependents, PhoneService, MultipleLines, InternetService,
             OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
             StreamingTV, StreamingMovies, Contract, PaperlessBilling,
             PaymentMethod, tenure]]

    new_df = pd.DataFrame(data, columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ])

    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    labels = ["{} - {}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(
        df_2.tenure.astype(int),
        range(1, 80, 12),
        right=False,
        labels=labels
    )

    df_2.drop(columns=['tenure'], inplace=True)

    df_dummies = pd.get_dummies(df_2[[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod',
        'tenure_group'
    ]])

    prediction = model.predict(df_dummies.tail(1))[0]
    probability = model.predict_proba(df_dummies.tail(1))[0][1]

    if prediction == 1:
        st.error("🚨 Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.write(f"**Confidence:** {round(probability * 100, 2)}%")
