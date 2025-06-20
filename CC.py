#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Load & preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_csv("Tesla.csv", encoding="ISO-8859-1")
    df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=False)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    return df

def train_model(data):
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)
    return model

# --- Streamlit UI ---
st.title("📈 Share Price Predictor")
st.markdown("Enter today's stock details to predict the **Close Price** using a simple Linear Regression model.")

# Load data & train model
data = load_data()
model = train_model(data)

# Input form
with st.form("predict_form"):
    st.subheader("Enter Today's Stock Data")
    open_price = st.number_input("Open Price", min_value=0.0, format="%.2f")
    high_price = st.number_input("High Price", min_value=0.0, format="%.2f")
    low_price = st.number_input("Low Price", min_value=0.0, format="%.2f")
    volume = st.number_input("Volume", min_value=0.0, format="%.0f")

    predict_button = st.form_submit_button("Predict Closing Price")

    if predict_button:
        input_data = pd.DataFrame([[open_price, high_price, low_price, volume]],
                                  columns=['Open', 'High', 'Low', 'Volume'])
        prediction = model.predict(input_data)[0]
        st.success(f"📊 Predicted Closing Price: **${prediction:.2f}**")

# Optional: show training data
if st.checkbox("Show Sample Training Data"):
    st.write(data.head())


# In[ ]:




