import streamlit as st
from streamlit_disqus import st_disqus
import json
import requests
from streamlit_lottie import st_lottie

def load_lottifile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url : str):

    r=  requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def app():

    st.markdown("""This DEMO was created in bootcamp phase 2 at Talan Innovation Factory
                By : ABID Haythem,
                     SOUIBGUI Mohamed ,
                     and AOUINI Oussama   
                """)
    st.subheader("INSIGHTS")
    st.text("")
    st.text("")
    st.markdown("* Medical cost : This is where you can find out how much your medical insurance costs.")
    
    st.markdown("""
    -  Dashboard : This is where you can visualize your input data an get some insights about it, You just have to upload a .csv file that contains data about:
        - age : type float.
        - sex : male or female.
        - bmi : type float.
        - children: number of children pocessed , type float.
        - smoker : yes for smoker or no for non-smoker.
        - region : one of these propositions : 'southwest', 'southeast', 'northwest', 'northeast'.
        - charges : Medical costs.
    """)

    lottie_hello= load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_1ixgi8rs.json")
    st_lottie(
        lottie_hello,
        speed=1,
        reverse= False,
        loop=True,
        quality= "low",
        height= None,
        width= None,
        key=11,
)
    st.markdown("---")
    
    
       
       
    st.header("DISCUSSION")
    st.subheader("Let's chat") 

    st.markdown(" Leave a comment about our application") 
    st_disqus("streamlit-disqus-demo")