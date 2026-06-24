import streamlit as st
import pickle
import numpy as np

# 1. Set up the page title and layout
st.set_page_config(page_title="My ML Predictor", layout="centered")
st.title("🤖 My Machine Learning Model Web App")
st.write("Enter the required details below to get a prediction from the model.")

# 2. Load the model safely using Streamlit's caching
@st.cache_resource
def load_model():
    # Replace 'model.pkl' with the exact name of your pickle file
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Could not find your model file. Make sure it's named 'model.pkl' and is in the same folder!")
    st.stop()

# 3. Create the user inputs
st.subheader("📋 Input Features")

# CUSTOMIZE THESE: Change these sliders/inputs to match what your model expects
feature_1 = st.slider("Feature 1 (e.g., Age)", min_value=0, max_value=100, value=25)
feature_2 = st.slider("Feature 2 (e.g., Income in thousands)", min_value=10, max_value=200, value=50)
feature_3 = st.number_input("Feature 3 (e.g., Score)", min_value=0.0, max_value=10.0, value=5.0)

# 4. Make the prediction
if st.button("🚀 Run Prediction"):
    # Arrange inputs into the format your model expects (usually a 2D array)
    input_data = np.array([[feature_1, feature_2, feature_3]])
    
    # Generate prediction
    prediction = model.predict(input_data)
    
    # 5. Display the result
    st.success("🎉 Prediction Complete!")
    st.metric(label="Predicted Value", value=f"{prediction[0]}")
