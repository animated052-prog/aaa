import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="CardioCare | Heart Disease Predictor", 
    layout="wide", 
    page_icon="🫀",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling, hovering effects, and clean UI
st.markdown("""
    <style>
    .main-header {
        font-size: 40px;
        font-weight: 700;
        color: #E63946;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 18px;
        color: #457B9D;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #E63946;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D3557;
        color: white;
        transform: scale(1.02);
    }
    .info-box {
        background-color: #F1FAEE;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #457B9D;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_model():
    """Loads the KNN pickle model from disk. Caches it for performance."""
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        return str(e)

def create_gauge_chart(probability):
    """Creates a Plotly gauge chart to visualize risk percentage."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Risk", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1D3557"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "#A8E6CF"},     # Low risk - Green
                {'range': [30, 70], 'color': "#FFD3B6"},    # Medium risk - Orange
                {'range': [70, 100], 'color': "#FF8A8A"}    # High risk - Red
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ==========================================
# 3. SIDEBAR NAVIGATION & INFO
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/875/875569.png", width=100)
    st.title("CardioCare UI")
    st.markdown("---")
    st.info("💡 **How to use:** Fill out the patient metrics in the main panel and click 'Run Health Assessment' to see the KNN model's prediction.")
    st.markdown("---")
    st.markdown("### 👨‍💻 About the Project")
    st.markdown("Developed as an open-source Heart Disease Prediction tool using **K-Nearest Neighbors (KNN)**.")
    st.markdown("[View on GitHub](#) *(Link your repo here)*")

# ==========================================
# 4. MAIN APPLICATION UI
# ==========================================
st.markdown('<p class="main-header">🫀 CardioCare Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced K-Nearest Neighbors (KNN) Risk Assessment Tool</p>', unsafe_allow_html=True)

# Load the model
model = load_model()

if model is None:
    st.error("❌ **Critical Error:** `model.pkl` not found! Please ensure it is in the same directory as this script.")
    st.stop()
elif isinstance(model, str):
    st.error(f"❌ **Error loading model:** {model}")
    st.warning("Note: If the error mentions `_dist_metrics`, ensure your environment is running `scikit-learn==0.24.2`.")
    st.stop()

# Create App Tabs for better organization
tab_predict, tab_info = st.tabs(["🩺 Patient Assessment", "📚 Feature Glossary"])

with tab_info:
    st.markdown("""
    ### Understanding the Medical Features
    * **Age:** Patient's age in years.
    * **Sex:** 1 = Male; 0 = Female.
    * **Chest Pain Type (cp):** * 0: Typical angina
        * 1: Atypical angina
        * 2: Non-anginal pain
        * 3: Asymptomatic
    * **Resting Blood Pressure (trestbps):** Blood pressure in mm Hg on admission to the hospital.
    * **Cholesterol (chol):** Serum cholesterol in mg/dl.
    * **Fasting Blood Sugar (fbs):** 1 = > 120 mg/dl; 0 = < 120 mg/dl.
    * **Resting ECG (restecg):** 0 = Normal; 1 = ST-T wave abnormality; 2 = Left ventricular hypertrophy.
    * **Max Heart Rate (thalach):** Maximum heart rate achieved during stress test.
    * **Exercise Induced Angina (exang):** 1 = Yes; 0 = No.
    * **ST Depression (oldpeak):** ST depression induced by exercise relative to rest.
    * **Slope:** The slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping).
    * **Number of Major Vessels (ca):** 0-4 colored by fluoroscopy.
    * **Thalassemia (thal):** 1 = normal; 2 = fixed defect; 3 = reversable defect.
    """)

with tab_predict:
    st.markdown('<div class="info-box">Please enter the clinical parameters below accurately. Default values represent a baseline healthy adult.</div>', unsafe_allow_html=True)
    
    # ----------------------------------------
    # User Inputs organized into columns
    # ----------------------------------------
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Demographics")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        sex_input = st.radio("Biological Sex", ["Female", "Male"], horizontal=True)
        sex = 1 if sex_input == "Male" else 0
        
        st.markdown("#### 🫀 Symptom Data")
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], format_func=lambda x: f"{x} - {['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x]}")
        exang_input = st.radio("Exercise-Induced Angina", ["No", "Yes"], horizontal=True)
        exang = 1 if exang_input == "Yes" else 0

    with col2:
        st.markdown("#### 🩸 Vitals & Blood Work")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs_input = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], horizontal=True)
        fbs = 1 if fbs_input == "Yes" else 0
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)

    with col3:
        st.markdown("#### 📈 Test Results")
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2], format_func=lambda x: f"{x} - {['Normal', 'ST-T Abnormality', 'LV Hypertrophy'][x]}")
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment", [0, 1, 2], format_func=lambda x: f"{x} - {['Upsloping', 'Flat', 'Downsloping'][x]}")
        ca = st.slider("Major Vessels Colored (ca)", min_value=0, max_value=4, value=0)
        thal = st.selectbox("Thalassemia Status", [0, 1, 2, 3])

    # ----------------------------------------
    # Prediction Engine
    # ----------------------------------------
    st.markdown("---")
    
    # Center the button using columns
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        run_prediction = st.button("❤️ Run Comprehensive Health Assessment", use_container_width=True)

    if run_prediction:
        # Prepare the input array safely
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        with st.spinner("Analyzing patient metrics using K-Nearest Neighbors..."):
            try:
                # Making predictions
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                
                # KNN typically outputs probabilities based on neighbor votes
                heart_disease_prob = probabilities[1] * 100 
                
                st.markdown("---")
                st.markdown("### 📊 Assessment Report")
                
                res_col1, res_col2 = st.columns([1, 1.5])
                
                with res_col1:
                    st.write("") # Spacing
                    if prediction == 1:
                        st.error("#### ⚠️ High Risk Detected")
                        st.write("The model indicates a high probability of heart disease based on the clinical metrics provided.")
                        st.write("**Recommendation:** Immediate consultation with a cardiologist is highly advised for further diagnostic testing.")
                    else:
                        st.success("#### ✅ Low Risk Detected")
                        st.write("The model indicates a low probability of heart disease.")
                        st.write("**Recommendation:** Continue maintaining a healthy lifestyle, diet, and regular check-ups.")
                    
                    st.metric(label="Model Confidence", value=f"{max(probabilities)*100:.1f}%")
                
                with res_col2:
                    # Render the gauge chart
                    fig = create_gauge_chart(heart_disease_prob)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ An error occurred during prediction calculations: {e}")
