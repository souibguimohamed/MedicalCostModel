
from tkinter import BitmapImage
import streamlit as st
import pandas as pd
import streamlit as st
from scipy.stats import boxcox
import os
import numpy as np
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
# Model 

def app():
    st.title('You wanna try our app ?')
    ####################################################################################
    
    #####################################################################################
    ### Partie Backend
    #
    #
    ###

    df = pd.read_csv("app/insurance.csv")

    # Dummy variable
    categorical_columns = ['sex','children', 'smoker', 'region']
    df_encode = pd.get_dummies(data = df, prefix = 'OHE', prefix_sep='_',
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')
    

    
    y_bc,lam, ci= boxcox(df_encode['charges'],alpha=0.05)


    ## Log transform
    df_encode['charges'] = np.log(df_encode['charges'])


    # train , test and split data
    from sklearn.model_selection import train_test_split
    X = df_encode.drop('charges',axis=1) # Independet variable
    y = df_encode['charges'] # dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)


    # Step 1: add x0 =1 to dataset
    X_train_0 = np.c_[np.ones((X_train.shape[0],1)),X_train]
    X_test_0 = np.c_[np.ones((X_test.shape[0],1)),X_test]

    # Step2: build model
    theta = np.matmul(np.linalg.inv( np.matmul(X_train_0.T,X_train_0) ), np.matmul(X_train_0.T,y_train)) 

    # The parameters for linear regression model
    parameter = ['theta_'+str(i) for i in range(X_train_0.shape[1])]
    columns = ['intersect:x_0=1'] + list(X.columns.values)
    parameter_df = pd.DataFrame({'Parameter':parameter,'Columns':columns,'theta':theta})

    # Scikit Learn module
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train,y_train) # Note: x_0 =1 is no need to add, sklearn will take care of it.

    #Parameter
    sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)
    parameter_df = parameter_df.join(pd.Series(sk_theta, name='Sklearn_theta'))


    y_pred_sk = lin_reg.predict(X_test)

    # Non let's create an empty dataframe for our input data

    n_df= pd.DataFrame(columns=['age','sex','bmi', 'children', 'smoker', 'region'])

    ############################################################################
    ############################################################################


    ### Partie Frontend
    #
    #
    ###
    st.subheader(" Let's find out how much charges you will get ")
    st.markdown(" Please fill in you infos down here: ")

    #Input

    #Age
    age = st.slider("How old are you ?" ,1 , 100, 25)
    st.write(" You are ", age , "years old")


    #Sex 
    sex = st.selectbox(
     'Please select your gender',
     ('male', 'female'))

    st.write('You selected:', sex)

    # for our prediction we turn sex to float
    if sex == 'male':
        OHE_male= 1
    else:
        OHE_male= 0	

    #Bmi 
    bmi = st.number_input('Please insert your bmi ( Body Mass Index )')
    st.write('Your bmi is ', bmi)

    #Child 
    child = st.slider("How many childrens do you have ?" ,1 , 5, 2)
    st.write(" You have ", child , "children(s)")

    OHE_1 = 0
    OHE_2= 0
    OHE_3= 0
    OHE_4= 0
    OHE_5 = 0

    if child == 1:
        OHE_1= 1
    elif child == 2:
        OHE_2= 1
    elif child == 3:
        OHE_3= 1
    elif child == 4:
        OHE_4= 1
    else:
        OHE_5= 1

    #Smoke 
    smoke = st.radio(
     "Are you a smoker ?",
     ('yes', 'no'))

    OHE_yes =0

    if smoke == 'Yes':
        st.write('You are a smoker.')
        OHE_yes= 1
    else:
        st.write("You're not a smoker.")
    



    #Region 

    region = st.radio(
     "Select your region",
     ('southwest', 'southeast', 'northwest', 'northeast'))

    OHE_southwest= 0
    OHE_southeast= 0
    OHE_northwest= 0

    if region == 'southwest':
        st.write('You are from southwest.')
        OHE_southwest=1

    elif region == 'southeast':
        st.write('You are from southeast.')
        OHE_southeast = 1
    elif region == 'northwest':
        st.write('You are from northwest.')
        OHE_northwest= 1
    else:
        st.write("You are from northeast.")

    
    st.markdown("---")
    
    st.subheader(" Click down here to predict your charges  :")


    # Making predictions with the inserted values

    #n_df.insert(age,sex,bmi, child, smoke, region)

    n_df= pd.DataFrame(columns=['age', 'bmi', 'OHE_male', 'OHE_1', 'OHE_2', 'OHE_3', 'OHE_4', 'OHE_5',
       'OHE_yes', 'OHE_northwest', 'OHE_southeast', 'OHE_southwest'])

    data = [ [age, bmi, OHE_male, OHE_1, OHE_2, OHE_3, OHE_4, OHE_5,
       OHE_yes, OHE_northwest, OHE_southeast, OHE_southwest]]
    n_df = pd.DataFrame( data , columns = ['age', 'bmi', 'OHE_male', 'OHE_1', 'OHE_2', 'OHE_3', 'OHE_4', 'OHE_5',
       'OHE_yes', 'OHE_northwest', 'OHE_southeast', 'OHE_southwest'])

    pred= lin_reg.predict(n_df)

    charge = np.exp(pred)

    note = charge[0]

    
    

    if st.button('Get your charges'):
        st.write('Your charges are', note ,"$")
        lottie_hello= load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_de909vf3.json")
        st_lottie(
            lottie_hello,
            speed=1,
            reverse= False,
            loop=True,
            quality= "low",
            height= None,
            width= None,
            key=12,
        )



################################################################
# side bar






    


   
