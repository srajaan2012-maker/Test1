import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Customer Churn Prediction Dashboard")
st.markdown("Predict which customers are likely to churn and take proactive action!")

# Load model with better error handling
@st.cache_resource
def load_model():
    try:
        with open('churn_prediction_model.pkl', 'rb') as f:
            assets = pickle.load(f)
        
        # Verify the model has required components
        required_keys = ['model', 'scaler', 'feature_names']
        for key in required_keys:
            if key not in assets:
                st.error(f"❌ Model missing required key: {key}")
                return None
        
        st.success("✅ Model loaded successfully!")
        return assets
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("This usually means the model file is corrupted or in wrong format.")
        return None

def create_demo_prediction(age, geography, is_active_member, balance):
    """Create a demo prediction when model fails"""
    # Simple logic based on known risk factors
    base_prob = 0.2  # Base 20% churn rate
    
    # Age factor
    if age > 50:
        base_prob += 0.3
    elif age > 40:
        base_prob += 0.15
    
    # Geography factor
    if geography == "Germany":
        base_prob += 0.25
    elif geography == "Spain":
        base_prob += 0.05
    
    # Activity factor
    if is_active_member == "No":
        base_prob += 0.2
    
    # Balance factor
    if balance > 100000:
        base_prob += 0.1
    
    return min(base_prob, 0.95)  # Cap at 95%

def main():
    assets = load_model()
    
    # Sidebar with model info
    st.sidebar.header("📊 Model Information")
    
    if assets is None:
        st.sidebar.warning("⚠️ Using demo mode")
        st.warning("⚠️ **Demo Mode Active** - Showing demo predictions based on common churn factors.")
        demo_mode = True
    else:
        st.sidebar.metric("Accuracy", "86.5%")
        st.sidebar.metric("Recall", "46.4%")
        st.sidebar.metric("AUC Score", "85.0%")
        demo_mode = False
    
    # Main input form
    st.header("📋 Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        age = st.slider('Age', 18, 80, 40)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        geography = st.selectbox('Country', ['France', 'Germany', 'Spain'])
        
        st.subheader("Financial Information")
        credit_score = st.slider('Credit Score', 350, 850, 650)
        balance = st.number_input('Account Balance ($)', 0.0, 500000.0, 50000.0)
    
    with col2:
        st.subheader("Banking Relationship")
        tenure = st.slider('Tenure (Years)', 0, 10, 5)
        num_products = st.slider('Number of Products', 1, 4, 2)
        has_credit_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
        is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])
        estimated_salary = st.number_input('Estimated Salary ($)', 0.0, 200000.0, 50000.0)
    
    # Prediction
    st.header("🎯 Churn Prediction Results")
    
    if st.button('🔍 Predict Churn Risk', type='primary', use_container_width=True):
        if demo_mode:
            # Use demo prediction logic
            probability = create_demo_prediction(age, geography, is_active_member, balance)
            prediction = 1 if probability > 0.5 else 0
        else:
            # Use actual model
            try:
                # Convert inputs
                gender_encoded = 1 if gender == 'Female' else 0
                has_credit_card_encoded = 1 if has_credit_card == 'Yes' else 0  
                is_active_member_encoded = 1 if is_active_member == 'Yes' else 0
                
                # Create feature vector
                feature_vector = []
                for feature in assets['feature_names']:
                    if feature.startswith('Geo_'):
                        geo_feature = f"Geo_{geography}"
                        feature_vector.append(1 if feature == geo_feature else 0)
                    else:
                        feature_mapping = {
                            'CreditScore': credit_score,
                            'Gender': gender_encoded,
                            'Age': age,
                            'Tenure': tenure,
                            'Balance': balance,
                            'NumOfProducts': num_products,
                            'HasCrCard': has_credit_card_encoded,
                            'IsActiveMember': is_active_member_encoded,
                            'EstimatedSalary': estimated_salary
                        }
                        feature_vector.append(feature_mapping[feature])
                
                feature_array = np.array(feature_vector).reshape(1, -1)
                feature_array_scaled = assets['scaler'].transform(feature_array)
                
                with st.spinner('Analyzing customer data...'):
                    prediction = assets['model'].predict(feature_array_scaled)[0]
                    probability = assets['model'].predict_proba(feature_array_scaled)[0][1]
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
                probability = create_demo_prediction(age, geography, is_active_member, balance)
                prediction = 1 if probability > 0.5 else 0
                demo_mode = True
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Result")
            if demo_mode:
                st.warning("🎭 Demo Prediction")
            
            if prediction == 1:
                st.error("🚨 HIGH CHURN RISK")
            else:
                st.success("✅ LOW CHURN RISK")
            
            st.metric("Churn Probability", f"{probability:.1%}")
            st.progress(float(probability))
            
            # Risk level
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            st.metric("Risk Level", risk_level)
        
        with col2:
            st.subheader("Customer Profile")
            st.write(f"**Age:** {age}")
            st.write(f"**Gender:** {gender}")
            st.write(f"**Country:** {geography}")
            st.write(f"**Tenure:** {tenure} years")
            st.write(f"**Active Member:** {is_active_member}")
            st.write(f"**Balance:** ${balance:,.0f}")
            
            if demo_mode:
                st.info("💡 Based on common churn patterns: Age, Geography, Activity, Balance")

if __name__ == '__main__':
    main()