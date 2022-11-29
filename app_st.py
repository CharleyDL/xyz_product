"""
Created on Monday 28 Nov. 2022
@author: Charley ∆ Lebarbier
"""

# Import librairies
import joblib
import numpy as np
import pandas as pd
import streamlit as st


#load model ML - Social Network
model = joblib.load('model/model_sn_joblib')

# Caching the model for fast loading
@st.cache


# Function to predict
def predict(gender, age, salary):
  """
    Use the ML Model to predict Future Purchaser
    Get 3 params : Gender, Age and Salary
  """
  input_predict = np.array([gender, age, salary])
  input_predict = input_predict.reshape(1, -1)

  prediction = (model['Model']).predict((model['Scaler']).transform(input_predict))

  return prediction



######################################################
###################### STREAMLIT #####################
######################################################

# METADATA WEBAPP
st.set_page_config(
                    page_title = "Purchaser Prediction",
                    page_icon = ":crystal_ball:",
                    layout = "wide")

#background
page_bg_img = f"""
  <style>
    .stApp {{
    background-image: url("https://github.com/CharleyDL/xyz_product/blob/main/img/bck_cover.jpg?raw=true");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
  </style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# HEADER
st.markdown("<h1 style='text-align: center;'>Purchaser Prediction</h1>", unsafe_allow_html=True)


# WEB CONTAIN
col1, col2, col3 = st.columns(3)

with col1:
  st.header("Gender")
  gender = st.radio("", ('male', 'female'), horizontal = True)
  if gender == 'male':
    gender = 1
  else:
    gender = 0

with col2:
  st.header("Age")
  age = st.slider("", 18, 70)

with col3:
  st.header("Estimate Salary / Year")
  salary = st.number_input("", min_value=15000)

#Button
st.markdown("----", unsafe_allow_html=True)
columns = st.columns((2, 1, 2))

if columns[1].button('Future Purchaser?'):
  prediction = predict(gender, age, salary)
  if prediction == 1:
    st.success("Yes")
  else:
    st.error("No")