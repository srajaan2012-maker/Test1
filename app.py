import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Customer Churn Prediction")
st.write("Enter customer details to predict churn risk")

# Simple model loading
def load_model():
    try:
        with open('churn_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 18, 80, 40)
    geography = st.selectbox('Country', ['France', 'Germany', 'Spain'])
    is_active = st.selectbox('Active Member', ['Yes', 'No'])

with col2:
    balance = st.number_input('Balance ($)', 0.0, 500000.0, 50000.0)
    credit_score = st.slider('Credit Score', 350, 850, 650)
    gender = st.selectbox('Gender', ['Male', 'Female'])

if st.button('Predict Churn'):
    assets = load_model()
    
    if assets is not None:
        try:
            # Prepare input
            gender_encoded = 1 if gender == 'Female' else 0
            active_encoded = 1 if is_active == 'Yes' else 0
            
            # Create feature vector
            feature_vector = [
                credit_score,           # CreditScore
                gender_encoded,         # Gender
                age,                    # Age
                5,                      # Tenure (default)
                balance,                # Balance
                2,                      # NumOfProducts (default)
                1,                      # HasCrCard (default)
                active_encoded,         # IsActiveMember
                50000.0,                # EstimatedSalary (default)
                1 if geography == 'France' else 0,    # Geo_France
                1 if geography == 'Germany' else 0,   # Geo_Germany
                1 if geography == 'Spain' else 0      # Geo_Spain
            ]
            
            # Transform and predict
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = assets['scaler'].transform(X)
            probability = assets['model'].predict_proba(X_scaled)[0][1]
            
            # Display result
            st.subheader("Prediction Result")
            if probability > 0.5:
                st.error(f"🚨 High Churn Risk: {probability:.1%}")
            else:
                st.success(f"✅ Low Churn Risk: {probability:.1%}")
                
            st.progress(float(probability))
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        # Fallback demo prediction
        base_prob = 0.2
        if age > 50: base_prob += 0.3
        if geography == 'Germany': base_prob += 0.25
        if is_active == 'No': base_prob += 0.2
        
        st.warning(f"🎭 Demo Prediction: {min(base_prob, 0.95):.1%} churn probability")

st.info("💡 Based on machine learning model analyzing customer behavior patterns")