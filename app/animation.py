import time
import requests

import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner


def load_lottifile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url : str):

    r=  requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_hello= load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_3vbOcw.json")
