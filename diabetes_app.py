# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:52:47 2024

@author: hp
"""

import numpy as np
import joblib
import streamlit as st

# loading the saved model
loaded_model = joblib.load("D:/ai and data science course/MACHINE LEARNING MIAN PROJECT/dibetes detector/diabetes_model.joblib")

# creating a function for Prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():
    # giving a title
    st.title('Diabetes Prediction Web App')

    # getting the input data from the user
    children = st.text_input('Number of children')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        input_data = [float(children), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
        diagnosis = diabetes_prediction(input_data)

    st.success(diagnosis)

if __name__ == '__main__':
    main()