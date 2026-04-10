import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("thyroid_model.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")

st.title("🧠 Thyroid Recurrence Prediction App")

# ================= INPUT UI =================
def user_input():
    age = st.number_input("Age", 10, 100)

    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    adenopathy = st.selectbox("Adenopathy", ["Yes", "No"])
    focality = st.selectbox("Focality", ["Uni-Focal", "Multi-Focal"])
    stage = st.selectbox("Stage", ["I", "II", "III", "IV"])

    data = {
        "Age": age,
        "Gender": gender,
        "Smoking": smoking,
        "Adenopathy": adenopathy,
        "Focality": focality,
        "Stage": stage,
    }

    return pd.DataFrame([data])


input_df = user_input()

# ================= ENCODING =================
def encode_input(df):
    df_copy = df.copy()

    for col in df_copy.columns:
        if col in encoders:
            le = encoders[col]
            df_copy[col] = le.transform(df_copy[col])

    return df_copy


encoded_input = encode_input(input_df)

# Ensure same column order
encoded_input = encoded_input.reindex(columns=columns, fill_value=0)

# ================= PREDICTION =================
if st.button("Predict"):

    prediction = model.predict(encoded_input)[0]
    prob = model.predict_proba(encoded_input)[0][1]

    # Output
    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Recurrence: YES")
    else:
        st.success("✅ Recurrence: NO")

    st.write(f"👉 Recurrence Risk = {round(prob * 100, 2)}%")

    # ================= FEATURE IMPORTANCE =================
    st.subheader("📊 Feature Importance")

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(feat_imp.set_index("Feature"))

    top_features = feat_imp.head(2)["Feature"].values
    st.write(f"👉 {top_features[0]} and {top_features[1]} influenced prediction most")

    # ================= COUNTERFACTUAL =================
    st.subheader("💡 Counterfactual Suggestion")

    suggestion = ""

    if "Stage" in input_df.columns:
        if input_df["Stage"][0] in ["III", "IV"]:
            suggestion += "If stage was lower, recurrence risk may decrease. "

    if "Adenopathy" in input_df.columns:
        if input_df["Adenopathy"][0] == "Yes":
            suggestion += "Absence of adenopathy reduces recurrence risk. "

    if suggestion == "":
        suggestion = "Patient already in relatively low-risk category."

    st.info(f"👉 {suggestion}")
