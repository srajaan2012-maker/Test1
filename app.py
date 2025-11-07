import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys

# Set page configuration
st.set_page_config(
    page_title=\"Customer Churn Predictor\",
    page_icon=\"🏦\", 
    layout=\"wide\"
)

st.title(\"🏦 Customer Churn Prediction Dashboard\")
st.markdown(\"Predict which customers are likely to churn and take proactive action!\")

# Load model using native pickle (no joblib dependency)
@st.cache_resource
def load_model():
    try:
        with open('churn_prediction_model.pkl', 'rb') as f:
            assets = pickle.load(f)
        st.success(\"✅ Model loaded successfully!\")
        return assets
    except Exception as e:
        st.error(f\"❌ Error loading model: {e}\")
        return None

def main():
    assets = load_model()
    if assets is None:
        st.info(\"💡 Please ensure the model file is properly uploaded\")
        return
    
    model = assets['model']
    scaler = assets['scaler'] 
    feature_names = assets['feature_names']
    
    # Sidebar with model info
    st.sidebar.header(\"📊 Model Information\")
    st.sidebar.metric(\"Accuracy\", \"86.5%\")
    st.sidebar.metric(\"Recall\", \"46.4%\")
    st.sidebar.metric(\"AUC Score\", \"85.0%\")
    
    # Main input form
    st.header(\"📋 Enter Customer Information\")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(\"Demographics\")
        age = st.slider('Age', 18, 80, 40)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        geography = st.selectbox('Country', ['France', 'Germany', 'Spain'])
        
        st.subheader(\"Financial Information\")
        credit_score = st.slider('Credit Score', 350, 850, 650)
        balance = st.number_input('Account Balance ($)', 0.0, 500000.0, 50000.0)
    
    with col2:
        st.subheader(\"Banking Relationship\")
        tenure = st.slider('Tenure (Years)', 0, 10, 5)
        num_products = st.slider('Number of Products', 1, 4, 2)
        has_credit_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
        is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])
        estimated_salary = st.number_input('Estimated Salary ($)', 0.0, 200000.0, 50000.0)
    
    # Convert inputs
    gender_encoded = 1 if gender == 'Female' else 0
    has_credit_card_encoded = 1 if has_credit_card == 'Yes' else 0  
    is_active_member_encoded = 1 if is_active_member == 'Yes' else 0
    
    # Create feature vector
    feature_vector = []
    for feature in feature_names:
        if feature.startswith('Geo_'):
            geo_feature = f\"Geo_{geography}\"
            feature_vector.append(1 if feature == geo_feature else 0)
        else:
            # Map feature names to input variables
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
    
    # Prediction
    st.header(\"🎯 Churn Prediction Results\")
    
    if st.button('🔍 Predict Churn Risk', type='primary', use_container_width=True):
        try:
            feature_array = np.array(feature_vector).reshape(1, -1)
            feature_array_scaled = scaler.transform(feature_array)
            
            with st.spinner('Analyzing customer data...'):
                prediction = model.predict(feature_array_scaled)[0]
                probability = model.predict_proba(feature_array_scaled)[0][1]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(\"Prediction Result\")
                if prediction == 1:
                    st.error(f\"🚨 HIGH CHURN RISK\")
                else:
                    st.success(f\"✅ LOW CHURN RISK\")
                
                st.metric(\"Churn Probability\", f\"{probability:.1%}\")
                st.progress(float(probability))
                
                # Risk level
                if probability < 0.3:
                    risk_level = \"Low\"
                    risk_color = \"green\"
                elif probability < 0.7:
                    risk_level = \"Medium\" 
                    risk_color = \"orange\"
                else:
                    risk_level = \"High\"
                    risk_color = \"red\"
                
                st.metric(\"Risk Level\", risk_level)
            
            with col2:
                st.subheader(\"Customer Profile\")
                st.write(f\"**Age:** {age}\")
                st.write(f\"**Gender:** {gender}\")
                st.write(f\"**Country:** {geography}\")
                st.write(f\"**Tenure:** {tenure} years\")
                st.write(f\"**Active Member:** {is_active_member}\")
                st.write(f\"**Balance:** ${balance:,.0f}\")
                
        except Exception as e:
            st.error(f\"Prediction error: {e}\")

if __name__ == '__main__':
    main()
```"