import numpy as np
import joblib
import streamlit as st

#load the trained model
best_model=joblib.load("model/rf_clf_4.pkl")

#streamlit app title

st.title("machine learning model deployment ")
st.write("enter your medical details to know about your diabetic status")

#define the input fields
st.sidebar.header(" Your medical records")

preg=st.sidebar.number_input("preg",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
plas=st.sidebar.number_input("plas",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
pres=st.sidebar.number_input("pres",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
skin=st.sidebar.number_input("skin",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
test=st.sidebar.number_input("test",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
mass=st.sidebar.number_input("mass",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
pedi=st.sidebar.number_input("pedi",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
age=st.sidebar.number_input("age",min_value=0.0,max_value=100.0,value=50.0,step=0.1)

input_data=np.array([[preg,plas,pres,skin,test,mass,pedi,age]])

if st.sidebar.button("Predict"):
    prediction=best_model.predict(input_data)
    st.success(f"Prediction:{prediction[0]}")
