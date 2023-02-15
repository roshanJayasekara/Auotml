import streamlit as st
import pandas as pd
import os

 # for profiling the data
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt

# ML libraries
from pycaret.classification import setup, compare_models, pull, save_model,plot_model

with st.sidebar:
    st.image("logo.png")
    st.title("AutoML Classification App")
    st.title("Upload your file for modeling")
    file=st.file_uploader("upload your file here")
    choice=st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This App can be used to creare mahcine learning model and data profiling using pandas and pycaret")
     
if file is not None:
    df = pd.read_csv(file)
   

if choice=="Upload":
    if file:
        st.dataframe(df)
    else:
        st.title("Upload your file for modeling")
    
if choice=="Profiling":
    if file:
        st.title("Automated Exploratory Data Analysis")
        profile_report=ProfileReport(df)
        st_profile_report(profile_report)
    else:
        st.title("Upload your file for modeling")

if choice=="ML":
    if file:
        st.title("Machine learning Model")
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            try:
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
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(plot_model(best_model, plot = 'confusion_matrix',display_format="streamlit",plot_kwargs = {'percent' : True}))
            except ValueError as e:
                 st. write(str(e))
        
    else:
        st.title("Upload your file for modeling")
           
if choice=="Download":
    if file:
        with open("best_model.pkl", 'rb') as f:
            st.download_button("Download Model",f,file_name="best_model.pkl")
    
    else:
        st.title("Upload your file for modeling")