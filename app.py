import streamlit as st
import pickle
import numpy as np

# 1. Page Config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered", page_icon="❤️")
st.title("❤️ Heart Disease Risk Predictor (KNN Model)")
st.write("Input the patient's clinical data below to get a real-time prediction.")

# 2. Load your specific KNeighborsClassifier Model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ 'model.pkl' not found! Make sure the file is named exactly 'model.pkl' and is in the same folder as this script.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# 3. Create Inputs for the 13 Features
st.subheader("📋 Patient Clinical Metrics")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", min_value=1, max_value=120, value=50)
    
    sex_input = st.selectbox("Biological Sex", ["Female", "Male"])
    sex = 1 if sex_input == "Male" else 0
    
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
    trestbps = st.number_input("Resting Blood Pressure (trestbps in mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (chol in mg/dl)", min_value=100, max_value=600, value=200)
    
    fbs_input = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["No", "Yes"])
    fbs = 1 if fbs_input == "Yes" else 0

with col2:
    restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
    
    exang_input = st.selectbox("Exercise-Induced Angina (exang)", ["No", "Yes"])
    exang = 1 if exang_input == "Yes" else 0
    
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", [0, 1, 2])
    ca = st.slider("Major Vessels Colored by Fluoroscopy (ca)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia Status (thal)", [0, 1, 2, 3])

# 4. Prediction Logic
st.markdown("---")
if st.button("❤️ Run Health Assessment", use_container_width=True):
    # Constructing the exact 13-feature array required by your KNN model
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        heart_disease_prob = probabilities[1] * 100
        
        st.subheader("Analysis Result:")
        if prediction == 1:
            st.error(f"⚠️ **High Risk:** The model predicts presence of heart disease with a confidence of {heart_disease_prob:.1f}%.")
        else:
            st.success(f"✅ **Low Risk:** The model predicts a healthy profile. Confidence of no heart disease is {probabilities[0]*100:.1f}%.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
