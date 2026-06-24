import streamlit as st
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="centered", page_icon="❤️")
st.title("❤️ Heart Disease Risk Predictor")
st.write("Input the patient's clinical data below to assess heart disease probability.")

# 2. Load the Heart Disease Model
@st.cache_resource
def load_model():
    # Make sure your model file is named 'model.pkl' in your GitHub repo
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ 'model.pkl' not found. Please upload your model file to the repository.")
    st.stop()

# 3. User Inputs Form
st.subheader("📊 Patient Information")

# Layout columns for better presentation
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", min_value=1, max_value=120, value=50)
    
    sex_input = st.selectbox("Biological Sex", ["Female", "Male"])
    sex = 1 if sex_input == "Male" else 0
    
    cp_input = st.selectbox("Chest Pain Type", [
        "Typical Angina", 
        "Atypical Angina", 
        "Non-anginal Pain", 
        "Asymptomatic"
    ])
    # Mapping to standard 0-3 values
    cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp_input)
    
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

with col2:
    fbs_input = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs = 1 if fbs_input == "Yes" else 0
    
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    
    exang_input = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    exang = 1 if exang_input == "Yes" else 0
    
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

# Lower section for remaining complex metrics
st.subheader("🔬 Advanced Clinical Metrics")
col3, col4 = st.columns(2)

with col3:
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)

with col4:
    thal = st.selectbox("Thalassemia Status", [0, 1, 2, 3], help="0: Normal, 1: Fixed Defect, 2: Reversible Defect, 3: Unspecified")

# 4. Processing the Data
st.markdown("---")
if st.button("❤️ Run Health Assessment", use_container_width=True):
    # Order must perfectly match the exact feature order used during your model training
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    try:
        prediction = model.predict(input_data)
        
        # Check if the model outputs probabilities (predict_proba) or binary predictions
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1] * 100
            st.subheader(f"Analysis Result:")
            if prediction[0] == 1:
                st.error(f"⚠️ High Risk Detected: The model estimates a {probability:.1f}% probability of heart disease.")
            else:
                st.success(f"✅ Low Risk Detected: The model estimates a {probability:.1f}% probability of heart disease.")
        else:
            # Fallback for standard binary models
            if prediction[0] == 1:
                st.error("⚠️ High Risk: The model predicts indicators of heart disease.")
            else:
                st.success("✅ Low Risk: The model predicts a healthy cardiovascular profile.")
                
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("💡 Hint: Ensure the feature order in `input_data` matches how your model was originally trained.")
