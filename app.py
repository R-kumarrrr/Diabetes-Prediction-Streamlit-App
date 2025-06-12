import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    /* Overall app background and text color */
    [data-testid="stAppViewContainer"] {
        background-color: #1e1e1e; /* Dark background */
        color: white; /* Default text color */
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    /* Prediction results container */
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #2a2a2a; /* Slightly lighter dark for the box */
        margin: 1rem 0;
    }
    /* Metric box for probability (white box) */
    .metric-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #ffffff; /* White background */
        color: #1e1e1e; /* Dark text for white background */
        margin: 0.5rem 0;
    }
    /* Specific style for High Risk box */
    .risk-high-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #4a2c2c; /* Reverted to original dark red/brown from image */
        color: white;
        margin: 0.5rem 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.2em;
    }
    /* Specific style for Low Risk box */
    .risk-low-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #2c4a2c; /* Dark green for low risk */
        color: white;
        margin: 0.5rem 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.2em;
    }
    /* Adjust Streamlit subheader color in dark mode */
    h3 {
        color: white;
    }
    p {
        color: white; /* Ensure paragraph text is white by default in the dark theme */
    }
    </style>
    """, unsafe_allow_html=True)

# Loading the trained models 
try:
    scaler = joblib.load('scaler.pkl')
    models = {
        'Logistic Regression': joblib.load('logistic_regression_model.pkl'),
        'Decision Tree': joblib.load('decision_tree_model.pkl'),
        'Random Forest': joblib.load('random_forest_model.pkl'),
        'Gradient Boosting': joblib.load('gradient_boosting_model.pkl'),
        'Linear SVM': joblib.load('linear_svm_model.pkl')
    }
except FileNotFoundError:
    st.error("Model or scaler files not found. Please train and save them first by running healthcare_sl.py.")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("🏥 About")
    st.markdown("""
    This application uses machine learning to predict diabetes risk based on patient data.
    
    ### Features:
    - Multiple ML models
    - Real-time predictions
    - Probability scores
    - Input validation
    
    ### Data Fields:
    - **Pregnancies**: Number of times pregnant
    - **Glucose**: Plasma glucose concentration
    - **Blood Pressure**: Diastolic blood pressure (mm Hg)
    - **Skin Thickness**: Triceps skin fold thickness (mm)
    - **Insulin**: 2-Hour serum insulin (mu U/ml)
    - **BMI**: Body mass index
    - **Diabetes Pedigree**: Diabetes pedigree function
    - **Age**: Age in years
    """)

# Main content
st.title("Diabetes Risk Prediction System")
st.markdown("Enter the patient's details below to predict the likelihood of diabetes.")

# Model selection
selected_model_name = st.selectbox(
    'Select Prediction Model',
    list(models.keys()),
    help="Choose the machine learning model for prediction"
)

model = models[selected_model_name]

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Information")
    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=17,
        value=1,
        help="Number of times pregnant"
    )
    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=0,
        max_value=200,
        value=100,
        help="Plasma glucose concentration"
    )
    bloodpressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=0,
        max_value=150,
        value=70,
        help="Diastolic blood pressure"
    )
    skinthickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=20,
        help="Triceps skin fold thickness"
    )

with col2:
    st.subheader("Additional Information")
    insulin = st.number_input(
        "Insulin Level (mu U/ml)",
        min_value=0,
        max_value=900,
        value=80,
        help="2-Hour serum insulin"
    )
    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        help="Body mass index"
    )
    diabetespedigreefunction = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=2.5,
        value=0.5,
        help="Diabetes pedigree function"
    )
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=120,
        value=30,
        help="Age in years"
    )

# Prediction button
if st.button("Predict Diabetes Risk", type="primary"):
    # Input validation
    if glucose < 40 or bloodpressure < 40:
        st.warning("⚠️ Warning: Unusually low values detected for Glucose or Blood Pressure. Please verify the inputs.")
    
    input_data = pd.DataFrame([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]],
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    scaled_input_data = scaler.transform(input_data)
    prediction = model.predict(scaled_input_data)
    
    # Get prediction probability
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(scaled_input_data)[:, 1]
    else:
        decision_values = model.decision_function(scaled_input_data)
        prediction_proba = (decision_values > 0).astype(float)
    
    # Display results in a nice format
    st.markdown("---")
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    
    st.subheader(f"Prediction Results using {selected_model_name}")
    
    # Create columns for metrics
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        if prediction[0] == 1:
            st.markdown('<div class="risk-high-box">High Risk of Diabetes</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low-box">Low Risk of Diabetes</div>', unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:1em; margin-bottom:0; color:#1e1e1e;'>Probability of Diabetes</p><p style='font-size:2em; font-weight:bold; margin-top:0; color:white;'>{prediction_proba[0]:.1%}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add interpretation
    st.markdown("### Interpretation")
    if prediction[0] == 1:
        st.markdown("""
        <p style=\"font-size:18px; color:white;\">The model suggests a <b>high risk</b> of diabetes. It's crucial to consult with a healthcare professional for:</p>
        <ul>
            <li>Further medical evaluation and diagnosis</li>
            <li>Personalized lifestyle modifications (diet, exercise)</li>
            <li>Regular monitoring of blood sugar levels</li>
        </ul>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p style=\"font-size:18px; color:white;\">The model suggests a <b>low risk</b> of diabetes. It's recommended to:</p>
        <ul>
            <li>Maintain a healthy and balanced lifestyle</li>
            <li>Undergo regular medical check-ups</li>
            <li>Monitor blood sugar levels, especially with a family history</li>
        </ul>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
