from email.mime import image
import streamlit as st
from PIL import Image

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

#For dataset
import pandas as pd
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization

# We'll convert the image to bytes 
# so that it can be displayed using an <img> HTML element.
#  The helper function below takes the path to the image
#  and converts it to bytes
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def app():
    st.title(" Let's explore our DATA together ")

    lottie_hello= load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_HLo5AP.json")
    st_lottie(
        lottie_hello,
        speed=1,
        reverse= False,
        loop=True,
        quality= "low",
        height= None,
        width= None,
        key=14,
)

    st.write(" ### Upload Your Data ")
   
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        
        #read csv
        df =pd.read_csv(uploaded_file)

        st.write(" ### Librairies used")

        st.code("""
        # Librairies
        import pandas  as pd #Data manipulation
        import numpy as np #Data manipulation
        import matplotlib.pyplot as plt #Visualization
        import seaborn as sns #Visualization 
        """)

        st.subheader(" Let's explore our data ")
        st.markdown(" Some random data")
        st.write(df.head())

        st.write("")
        st.write("")
        st.write("")

        st.subheader("Let's do some visualization now ")

        st.markdown("For our visualization purpose will fit line using seaborn library only for bmi as independent variable and charges as dependent variable")
        
        st.code(""" # Visualization
            sns.lmplot(x='bmi',y='charges',data=df,aspect=2,height=6)
            plt.xlabel('Boby Mass Index$(kg/m^2)$: as Independent variable')
            plt.ylabel('Insurance Charges: as Dependent variable')
            plt.title('Charge Vs BMI')
        """)

        #figure
        #fig = plt.figure(figsize =(4, 4))
        #sns.lineplot(x='bmi',y='charges',data=df )
        #st.pyplot(fig)

        f= plt.figure(figsize=(8,8))
        sns.lineplot(x="age", y="charges",
             hue="sex",
             data=df)
        plt.title("Charges vs AGE")
        st.write(f)
        
        

        st.write("")
        st.write("")
        st.write("")

        st.subheader("Now, some Exploratory Data Analysis (EDA) ")
        st.code("df.describe()")
        st.write(df.describe())

        st.write("")
        st.write("")
        st.write("")

        st.subheader("Correlation between features")
        st.code("""
            # Correlation using heatmap
            corr = df.corr()
            sns.heatmap(corr, cmap = 'Wistia', annot= True);
        
        """)

        # correlation fig
        corr = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, ax=ax , cmap = 'Wistia', annot= True)
        st.write(fig)

        st.subheader(" Distribution of insurance charges ")
        st.code(
            """ 
            f= plt.figure(figsize=(12,4))
            ax=f.add_subplot(121)
            sns.distplot(df['charges'],bins=50,color='r',ax=ax)
            ax.set_title('Distribution of insurance charges')
            ax=f.add_subplot(122)
            sns.distplot(np.log10(df['charges']),bins=40,color='b',ax=ax)
            ax.set_title('Distribution of insurance charges in $log$ sacle')
            ax.set_xscale('log'); 
            """
        )
        f= plt.figure(figsize=(12,4))
        ax=f.add_subplot(121)
        sns.distplot(df['charges'],bins=50,color='r',ax=ax)
        ax.set_title('Distribution of insurance charges')
        ax=f.add_subplot(122)
        sns.distplot(np.log10(df['charges']),bins=40,color='b',ax=ax)
        ax.set_title('Distribution of insurance charges in $log$ sacle')
        ax.set_xscale('log')
        st.write(f)

        st.subheader(" Charges vs Sex & Charges vs Smokers Using 'Violin plot' ")
        st.code(
            """
            f = plt.figure(figsize=(14,6))
            ax = f.add_subplot(121)
            sns.violinplot(x='sex', y='charges',data=df,palette='Wistia',ax=ax)
            ax.set_title('Violin plot of Charges vs sex')
            ax = f.add_subplot(122)
            sns.violinplot(x='smoker', y='charges',data=df,palette='magma',ax=ax)
            ax.set_title('Violin plot of Charges vs smoker');
            """
        )
        #fig
        f = plt.figure(figsize=(14,6))
        ax = f.add_subplot(121)
        sns.violinplot(x='sex', y='charges',data=df,palette='Wistia',ax=ax)
        ax.set_title('Violin plot of Charges vs sex')
        ax = f.add_subplot(122)
        sns.violinplot(x='smoker', y='charges',data=df,palette='magma',ax=ax)
        ax.set_title('Violin plot of Charges vs smoker')
        st.write(f)

        st.subheader("Charges vs Children using 'Box Plot' ")
        st.code(
            """
            plt.figure(figsize=(14,6))
            sns.boxplot(x='children', y='charges',hue='sex',data=df,palette='rainbow')
            plt.title('Box plot of charges vs children');
            """
        )

        # fig
        f= plt.figure(figsize=(14,6))
        sns.boxplot(x='children', y='charges',hue='sex',data=df,palette='rainbow')
        plt.title('Box plot of charges vs children')
        st.write(f)

        st.markdown(" Let's see charges related to number of childrens: ")
        st.code("df1 = df.groupby('children').agg(['mean','min','max'])['charges']")
        df1 = df.groupby('children').agg(['mean','min','max'])['charges']
        st.write(df1)

        st.subheader("Charges vs Age & Charges vs BMI using 'Scatter Plot' ")
        st.code(""" 
        f = plt.figure(figsize=(14,6))
        ax = f.add_subplot(121)
        sns.scatterplot(x='age',y='charges',data=df,palette='magma',hue='smoker',ax=ax)
        ax.set_title('Scatter plot of Charges vs age')
        ax = f.add_subplot(122)
        sns.scatterplot(x='bmi',y='charges',data=df,palette='viridis',hue='smoker')
        ax.set_title('Scatter plot of Charges vs bmi')
        """ )

        f = plt.figure(figsize=(14,6))
        ax = f.add_subplot(121)
        sns.scatterplot(x='age',y='charges',data=df,palette='magma',hue='smoker',ax=ax)
        ax.set_title('Scatter plot of Charges vs age')
        ax = f.add_subplot(122)
        sns.scatterplot(x='bmi',y='charges',data=df,palette='viridis',hue='smoker')
        ax.set_title('Scatter plot of Charges vs bmi')
        st.write(f)

        




    else:
        st.warning(" you need to upload a csv or excel file !!! ")
    

    
    

    
