import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np
image="Insurance.jpg"
st.set_page_config(page_title="student marks prediction",page_icon="😎")
st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.title("Insurance Cost Prediction")
st.write("""
Made by *Prabhat Kumar*
""")
prabhat=joblib.load("insurance_cost_prediction_model.pkl")
# exp=st.sidebar.slider("Hours",1,10,2)
# st.write(f"Hours",exp)
# y_pred=prabhat.predict([[exp]]).round(3)
# st.write(f"Obtained Marks is: ",float(y_pred))
age = st.number_input('Enter your age ')
# sex=  st.selectbox(("male"),("female"))
sex = st.selectbox('Select your choice Male or Female?',('male','female'))
if sex=="male":
    sex1=1
else:
    sex1=0
Bmi=st.number_input('Enter your BMI ')
smoker=st.selectbox('You are smoker or not?',('yes','no'))
if sex=="yes":
    smokers=1
else:
    smokers=0

if st.button("Predict cost"):
    input=(age,sex1,Bmi,smokers)
    input2=np.asarray(input)
    # input3=np.reshape(1,-1)
    final_value=prabhat.predict([input2])[0][0].round(2)
    st.write(f"The Incurance cost is: ",float(final_value))
    