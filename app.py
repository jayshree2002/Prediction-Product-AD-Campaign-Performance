import streamlit as st
import pandas as pd
import joblib

# Load the trained model
loaded_model = joblib.load('ad_campaign_model.pkl')

# Define categorical and numerical feature names
categorical_features = ['limit_infor', 'campaign_type', 'campaign_level', 'product_level']
numerical_features = ['resource_amount', 'email_rate', 'price', 'discount_rate', 'hour_resources', 'campaign_fee']

# Streamlit UI
st.title("Ad Campaign Performance Prediction")
st.write("This app predicts the expected number of orders based on campaign features.")

# Sidebar inputs
st.sidebar.header("Input Features")
def user_input_features():
    limit_infor = st.sidebar.selectbox("Limit Information", ["0", "1","10"])  # Replace with actual options
    campaign_type = st.sidebar.slider("Campaign Type", 0,6)
    campaign_level = st.sidebar.selectbox("Campaign Level", ["0", "1"])
    product_level = st.sidebar.selectbox("Product Level", ["1", "2", "3"])
    resource_amount = st.sidebar.slider("Resource Amount", 0,9)
    email_rate = st.sidebar.number_input("Email Rate", min_value=0.08, max_value=0.84)
    price = st.sidebar.number_input("Price", min_value=100, max_value=197)
    discount_rate = st.sidebar.number_input("Discount Rate", min_value=0.49, max_value=0.98)
    hour_resources = st.sidebar.number_input("Hour Resources", min_value=2, max_value=3410)
    campaign_fee = st.sidebar.number_input("Campaign Fee", min_value=20, max_value=33380)
    
    input_data = pd.DataFrame([[limit_infor, campaign_type, campaign_level, product_level, resource_amount, email_rate, price, discount_rate, hour_resources, campaign_fee]],
                              columns=categorical_features + numerical_features)
    return input_data

input_df = user_input_features()

# Prediction
if st.button("Predict Orders"):
    prediction = loaded_model.predict(input_df)
    st.subheader("Predicted Orders")
    st.write(f"Estimated Number of Orders: **{prediction[0]:.2f}**")
