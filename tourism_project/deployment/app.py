import streamlit as st
import pandas as pd
import joblib
import os

from huggingface_hub import hf_hub_download

# -------------------------------------------------
# Download model from Hugging Face Model Hub
# -------------------------------------------------
MODEL_FILENAME = "best_tourism_model_v1.joblib"

model_path = hf_hub_download(
    repo_id="sankar-guru/tourism-model",
    filename=MODEL_FILENAME,
    token=os.environ.get("HF_TOKEN")  # safe for Spaces + local
)

model = joblib.load(model_path)

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("Tourism Package Prediction App")

st.write(
    "This application predicts whether a customer is likely to take a tourism package "
    "based on customer profile details."
)

st.write("Please enter the customer details below.")

# -------------------------------------------------
# Collect user input
# -------------------------------------------------
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0)
Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, value=1)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips", min_value=0, value=1)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=30000.0)

# -------------------------------------------------
# Prepare input dataframe
# -------------------------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

# -------------------------------------------------
# Prediction
# -------------------------------------------------
classification_threshold = 0.45

if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= classification_threshold)

    if prediction == 1:
        st.success("✅ Customer is likely to take the tourism package.")
    else:
        st.warning("❌ Customer is unlikely to take the tourism package.")
