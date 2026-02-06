import streamlit as st
import pandas as pd
import pickle

# ------------------ Page Config ------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📉 Customer Churn Prediction")
st.write("Enter customer details to predict churn probability")

# ------------------ Load Model & Reference Data ------------------
model = pickle.load(open("model_1.sav", "rb"))
df_1 = pd.read_csv("first_telc.csv")

# Ensure consistency with training
df_1['SeniorCitizen'] = df_1['SeniorCitizen'].astype(str)

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

    new_df = pd.DataFrame([[
        SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner,
        Dependents, PhoneService, MultipleLines, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling,
        PaymentMethod, tenure
    ]], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ])

    # Match training format
    new_df['SeniorCitizen'] = new_df['SeniorCitizen'].astype(str)

    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Ensure numeric columns
    df_2['TotalCharges'] = pd.to_numeric(df_2['TotalCharges'], errors='coerce')
    df_2['MonthlyCharges'] = pd.to_numeric(df_2['MonthlyCharges'], errors='coerce')

    # Tenure grouping
    labels = [f"{i} - {i+11}" for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(
        df_2['tenure'].astype(int),
        range(1, 80, 12),
        right=False,
        labels=labels
    )

    df_2.drop(columns=['tenure'], inplace=True)

    # Categorical & Numeric separation
    cat_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod',
        'tenure_group'
    ]

    num_cols = ['MonthlyCharges', 'TotalCharges']

    df_cat = pd.get_dummies(df_2[cat_cols], drop_first=False)
    df_final = pd.concat([df_cat, df_2[num_cols]], axis=1)

    # Align EXACTLY with training features
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    df_final = df_final.reindex(columns=model.feature_names_in_, fill_value=0)

    # Convert all to numeric (fix XGBoost error)
    df_final = df_final.astype(float)

    # Prediction
    prediction = model.predict(df_final.tail(1))[0]
    probability = model.predict_proba(df_final.tail(1))[0][1]

    # ------------------ Output ------------------
    if prediction == 1:
        st.error("🚨 This customer is likely to churn")
    else:
        st.success("✅ This customer is likely to stay")

    st.write(f"**Confidence:** {round(probability * 100, 2)}%")

    # ------------------ Risk Segmentation ------------------
    if probability < 0.30:
        st.success("🟢 Risk : Low Risk")
    elif probability < 0.60:
        st.warning("🟡 Risk : Medium Risk")
    else:
        st.error("🔴 Risk : High Risk")

    # ------------------ Churn Reasons ------------------
    reasons = []

    if Contract == "Month-to-month":
        reasons.append("Customer is on a month-to-month contract")

    if MonthlyCharges > 80:
        reasons.append("High monthly charges")

    if TechSupport == "No":
        reasons.append("No tech support service")

    if OnlineSecurity == "No":
        reasons.append("No online security")

    if InternetService == "Fiber optic":
        reasons.append("Fiber optic plan (historically higher churn)")

    if len(reasons) == 0:
        reasons.append("General usage and contract patterns")

    st.subheader("Possible Reasons for Churn")
    for r in reasons[:3]:
        st.write(f"- {r}")

    # ------------------ Retention Campaign Suggestions ------------------
    campaigns = []

    if Contract == "Month-to-month":
        campaigns.append("Offer 10–15% discount on annual contract")

    if MonthlyCharges > 80:
        campaigns.append("Provide limited-time billing discount")

    if TechSupport == "No":
        campaigns.append("Offer free tech support trial for 3 months")

    if OnlineSecurity == "No":
        campaigns.append("Bundle free online security for 2 months")

    if len(campaigns) == 0:
        campaigns.append("Offer loyalty reward or referral bonus")

    st.subheader("Recommended Retention Actions")
    for c in campaigns[:3]:
        st.write(f"- {c}")

