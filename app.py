import streamlit as st
import pandas as pd
import os

 # for profiling the data
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# ML libraries
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("logo.png")
    st.title("AutoML Classification App")
    choice=st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This App can be used to creare mahcine learning model and data profiling using pandas and pycaret")
     
if os.path.exists("sourcedata.csv"):
   df=pd.read_csv("sourcedata.csv",index_col=None)

if choice=="Upload":
    st.title("Upload your file for modeling")
    file=st.file_uploader("upload your file here")
    if file:
        df=pd.read_csv(file,index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)
    
if choice=="Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report=ProfileReport(df)
    st_profile_report(profile_report)

if choice=="ML":
    st.title("Machine learning Model")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):
        setup(df,target=chosen_target)
        setup_df=pull()
        st.info("This is the Ml expriment settings")
        st.dataframe(setup_df)
        best_model=compare_models()
        compare_df=pull()
        st.info("This is the Ml model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, "best_model")
        
if choice=="Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download Model",f,file_name="best_model.pkl")