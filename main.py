from ast import Mod
import streamlit as st
from multiapp import MultiApp
from app import About, dashboard, Deployment
import json

import time
import requests

import streamlit as st
from streamlit_lottie import st_lottie




def load_lottifile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url : str):

    r=  requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()






st.markdown("<h1 style='text-align: center; '>BOOTCAMP WEEK 2</h1>", unsafe_allow_html=True)

app= MultiApp()


app.add_app("Medical Cost",Deployment.app)
app.add_app("Dashboard",dashboard.app)
app.add_app("About",About.app)

lottie_hello= load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_fp4hravo.json")
st_lottie(
    lottie_hello,
    speed=1,
    reverse= False,
    loop=True,
    quality= "low",
    height= None,
    width= None,
    key=None,
)

st.sidebar.title("Medical Cost Application")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.image("./img/Logo-talan.png", use_column_width=True)
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")


st.sidebar.title("About")
st.sidebar.markdown("""We will build a Linear regression model for Medical cost dataset.
     The dataset consists of age, sex, BMI(body mass index), children,
      smoker and region feature, which are independent and charge as a dependent feature.
       We will predict individual medical costs billed by health insurance.""")



app.run()