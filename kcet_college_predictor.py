import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder, StandardScaler

with open('kcet_trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('kcet_encoders_scaler.pkl', 'rb') as f:
    encoders = pickle.load(f)

category_encoder = encoders['category']
branch_encoder = encoders['branch']
location_encoder = encoders['location']
rank_scaler = encoders['scaler']

# UI
st.title("🎓 KCET College Predictor")
st.write("Enter your details to find the top 3 colleges you might get!")

category = st.selectbox("Category", category_encoder.categories_[0].tolist())
branch = st.selectbox("Preferred Branch", branch_encoder.categories_[0].tolist())
location = st.selectbox("Preferred Location", location_encoder.categories_[0].tolist())
user_rank = st.number_input("Enter your KCET Rank", min_value=1, max_value=200000, value=5000)

if st.button("Predict Colleges"):
    category_input = category_encoder.transform([[category]])
    branch_input = branch_encoder.transform([[branch]])
    location_input = location_encoder.transform([[location]])
    rank_input = rank_scaler.transform([[user_rank]])

    input_combined = np.concatenate([category_input, branch_input, location_input, rank_input], axis=1)

    probs = model.predict_proba(input_combined)[0]
    top_indices = np.argsort(probs)[::-1][:3]
    top_colleges = model.classes_[top_indices]

    st.success("🏆 Top 3 College Recommendations:")
    for i, college in enumerate(top_colleges, start=1):
        st.markdown(f"**{i}. {college}**")
