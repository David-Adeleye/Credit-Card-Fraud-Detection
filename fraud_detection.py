# Importing libraries
import streamlit as st
import pandas as pd
import xgboost 
import joblib
import lightgbm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from joblib import dump, load
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from annotated_text import annotated_text

# Load models
rf_loaded = joblib.load('rf_model.joblib')
xgb_loaded = joblib.load('xgb_model.joblib')
lgbm_loaded = joblib.load('lgbm_model.joblib')
#load preprocessor
preprocessor = joblib.load('preprocessor.joblib')

st.title("Credit Card Fraud Detection")

# Taking Inputs
trans_amt = st.number_input("Enter Transacton Amount: ")

card1 = st.number_input("Enter Credit limit: ", max_value=99999)

card2 = st.number_input("Enter first three digits of the card number: ", step=1, max_value=999)

card3 = st.number_input("Enter last three digits of the card number: ", step=1, max_value=999)

card4 = st.selectbox('Select Card Type:',('Visa', 'Mastercard', 'Discover', 'American Express'))

card5 = st.number_input("Enter CVV: ", step=1, max_value=999)

card6 = st.selectbox('Select Card Category:',('Debit', 'Credit'))

addr1 = st.number_input("Enter ZIP code: ", step=1, max_value=999)

dist1 = st.number_input("Enter distance between billing and purchase address in miles: ", step=1, max_value=99999)

P_email = st.selectbox('Enter email address of purchaser:',('Google', 'Yahoo Mail', 'Microsoft', 'Others'))

# Converting inputs to dataframe
input_data = pd.DataFrame({'TransactionAmt': [trans_amt],
                           'card1': [card1],
                          'card2': [card2],
                          'card3': [card3],
                          'card4': [card4.lower()],
                          'card5': [card5],
                          'card6': [card6.lower()],
                          'addr1': [addr1],
                          'dist1': [dist1],
                          'P_emaildomain': [P_email]})

# Predict outcome
if st.button("Predict"):
    # Display input Dataframe
    st.text("Inputted values:")
    st.dataframe(input_data, width=1000)

    # Transforming the input data to the required standard  
    input_preprocessed = preprocessor.transform(input_data)

    #Making prediction wuth Random Forest
    rf_predictions = rf_loaded.predict(input_preprocessed)
    rf_prob = rf_loaded.predict_proba(input_preprocessed)[0][1]
    rf_prob_all = rf_loaded.predict_proba(input_preprocessed)
    st.subheader("Predicting with Random Forest Method")
    st.text(f"Predicted value:  {rf_predictions}")    
    st.text(f"Propability values: {rf_prob_all}")    
    if rf_predictions ==0:                
        classification1 = 'Genuine'
        color1 = '#004'        
    else:  
        classification1 = 'Fraudulent'
        color1 = '#1f9'        
    
    annotated_text(
        ("Transaction Type: ", "#330"),
        (classification1, color1),
        )   
        
    st.markdown("""---""")   
    #Xgboost Model    
    xgb_predictions = xgb_loaded.predict(input_preprocessed)
    xgb_prob = xgb_loaded.predict_proba(input_preprocessed)[0][1]
    xgb_prob_all = xgb_loaded.predict_proba(input_preprocessed)    
    st.subheader("Predicting with Xgboost Method")
    st.text(f"Predicted value:  {xgb_predictions}")
    st.text(f"Propability values: {xgb_prob_all}")    
    if xgb_predictions ==0:        
        classification2 = 'Genuine'
        color2 = '#004'
    else:        
        classification2 = 'Fraudulent'
        color2 = '#1f9'   
        
    annotated_text(
        ("Transaction Type: ", "#330"),
        (classification2, color2),
        )    
        
    st.markdown("""---""")    
    #ligbtgbm Model
    lgbm_predictions = lgbm_loaded.predict(input_preprocessed)
    lgbm_prob = lgbm_loaded.predict_proba(input_preprocessed)[0][1]
    lgbm_prob_all = lgbm_loaded.predict_proba(input_preprocessed)
    st.subheader("Predicting with ligbtgbm Method")
    st.text(f"Predicted value:  {lgbm_predictions}")
    st.text(f"Propability values: {lgbm_prob_all}")    
    if lgbm_predictions ==0:        
        classification3 = 'Genuine'
        color3 = '#004'
    else:      
        classification3 = 'Fraudulent'
        color3 = '#1f9'
        
    annotated_text(
        ("Transaction Type: ", "#330"),
        (classification3, color3),
        )  
        
    st.markdown("""---""")
    Avg_prob = (rf_prob + xgb_prob + lgbm_prob)/3
    if Avg_prob >= 0.7:              
        st.error('The transaction is classified as: Fraudulent')
    elif Avg_prob >= 0.4 and Avg_prob < 0.7:        
        st.warning('The transaction is classified as: Moderate')
    else:        
        st.success('The transaction is classified as: Safe/Genuine')
            
st.markdown("""---""")
st.text("Designed by David Adeleye")