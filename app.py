import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler;
import joblib

st.set_page_config(layout='wide',page_icon=":sunglasses:",page_title='rainy day or not ')
st.title('Rainfall prediction system :blue[cool]:',anchor=False,)
# this is the header for the prediction page
rain=joblib.load('Rain.joblib')
# this is the selection area for the prediction page
st.sidebar.header('Select from the condition indicators')

df=pd.read_csv('weatherAUS.csv')
# loading the data file from the storage file parquets
# all inputs data 
train_inputs=pd.read_parquet('train_inputs.parquet')
val_inputs=pd.read_parquet('val_inputs.parquet')
test_inputs=pd.read_parquet('test_inputs.parquet')

# all output data form the training data to testing data
train_target=pd.read_parquet('train_target.parquet')
val_target=pd.read_parquet('val_target.parquet')
test_inputs=pd.read_parquet('test_inputs.parquet')



# choose a country from list
dates=pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
country=st.sidebar.multiselect('Choose your Location',df['Location'].unique())
allDate=st.sidebar.multiselect('These previously dates of occurances',dates)
windGustDirection=st.sidebar.multiselect('Wind Gust Direction',df['WindGustDir'].unique())
windDirection9am=st.sidebar.multiselect('Wind Direction 9am',df['WindDir9am'].unique())
windDirection3pm=st.sidebar.multiselect('Wind Direction 3am',df['WindDir3pm'].unique())
RainToday=st.sidebar.multiselect('Choose whether it rain Today',df['RainToday'].unique())

st.header('New Prediction')
col,col2=st.columns(2)

with col:
    with st.expander('Original Data File',icon='📚'):
        st.write(df)
    st.date_input(label='Select Data to Record')  
    MinTemp= st.number_input('Enter Min Temp',label_visibility='visible',min_value=0.)
    MaxTemp= st.number_input('Enter Max Temp',label_visibility='visible',min_value=0.)
    Rainfall= st.number_input('Enter Rainfall measure',label_visibility='visible',min_value=0.)
    Evaporation= st.number_input('Enter Evaporation measure',label_visibility='visible',min_value=0.)
    sunshine= st.number_input('Enter Sunshine measures',label_visibility='visible',min_value=0.)
    windGustSpeed= st.number_input('Enter WindGustSpeed measures',label_visibility='visible',min_value=0.)
       
# loading all the joblib files in the directory to this platform

with col2:
   windSpeed9am= st.number_input('Enter WindSpeed 9am Temp measures',label_visibility='visible',min_value=0.)
   windSpeed3am= st.number_input('Enter WindSpeed 3pm Temp measures',label_visibility='visible',min_value=0.)
   Humidity9am= st.number_input('Enter Humidity 9am measures',label_visibility='visible',min_value=0.)
   Humidity3pm= st.number_input('Enter Humidity 3pm',label_visibility='visible',min_value=0.)
   Pressure9am= st.number_input('Enter Pressure 9am',label_visibility='visible',min_value=0.)
   Cloud9am= st.number_input('Enter Cloud 9am measures',label_visibility='visible',min_value=0.)
   Cloud3pm= st.number_input('Enter Cloud 3pm measures',label_visibility='visible',min_value=0.)
   Temp9am= st.number_input('Enter Temperature at 9am',label_visibility='visible',min_value=0.)
   Temp3pm= st.number_input('Enter Temperature at 3pm',label_visibility='visible',min_value=0.)
   
   # creating a model prediction area for the inputs
   
   for i in rain:
       st.write(i)
DataFrame=pd.DataFrame([
country,MinTemp,MaxTemp,Rainfall,Evaporation,sunshine,windGustSpeed,
windSpeed9am,windSpeed3am,Humidity9am,Humidity3pm,Pressure9am,Cloud9am,Cloud3pm,Temp9am,Temp3pm])
# creating a helper function for the program to initialize an new data
def prediction(Data):
    # we accept the input of the data
    input_df=pd.DataFrame([Data])
    # we transform it to numerical values 
    input_df[rain['numerical_cols']]=rain['imputer'].transform(input_df[rain['numerical_cols']])
   # we transform it by scaling all values to one min value for all to be in range not to greater for working
    input_df[rain['numerical_cols']]=rain['scaler'].transform(input_df[rain['numerical_cols']])
    # we encode the columns using OneHotEncoder for the categorical columns
    input_df[rain['encoded_cols']]=rain['encoder'].transform(input_df[rain['categorical_cols']])
    
    X_input_df=input_df[rain['encoded_cols' +'numerical_cols']]
    pred=rain['model'].predict(X_input_df)[0]
    probability=rain['model'].prob_predict(X_input_df)[0][list(rain['model'].classes_).index(pred)]
    return st.write(pred, probability)
    