import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title("House Price Prediction App")
st.write("Predict house prices using a trained Random Forest model.")

# Define input fields for all 5 features
st.header("Input Features")
gr_liv_area = st.number_input("Living Area (GrLivArea)", value=0.0, step=10.0)
bedrooms = st.number_input("Number of Bedrooms (Bedrooms)", value=0, step=1)
total_bath = st.number_input("Total Bathrooms (TotalBath)", value=0.0, step=0.5)
kitchen_abv_gr = st.number_input("Number of Kitchens Above Ground (KitchenAbvGr)", value=0, step=1)
condition = st.number_input("Condition (Overall Quality, Condition)", value=0, step=1)

# Prediction button
if st.button("Predict"):
    # Prepare the input data
    input_features = np.array([[gr_liv_area, bedrooms, total_bath, kitchen_abv_gr, condition]])
    try:
        # Predict using the loaded model
        prediction = model.predict(input_features)
        st.subheader("Predicted House Price")
        st.write(f"${prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"Error during prediction: {e}")