import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
#import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Loan_Data.csv')

st.markdown("<h1 style = 'color: #114232; text-align: center; font-size: 60px; font-family: Monospace'>LOAN PREDICTION APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #87A922; text-align: center; font-family: cursive '>Built by Matthew</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

st.image('pngwing.com_bank_icon.png')

#Add Project proble statement
st.markdown("<h2 style = 'color: #FF9800; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)

st.markdown("Loan prediction involves using historical data on loan applicants to develop predictive models that can assess the creditworthiness of future loan applicants. This typically includes analyzing factors such as income, coapplicant income, and other relevant financial information to determine the likelihood of a borrower repaying a loan. By studying these factors and their impact on loan approval and repayment, financial institutions can make more informed decisions when evaluating loan applications. This helps reduce the risk of default and ensures that loans are granted to individuals who are more likely to repay them.</p>", unsafe_allow_html=True)


st.sidebar.image('pngwing.com_user_icon.png')
st.divider()
st.header('Project Data')
st.dataframe(data,use_container_width=True)

app_income = st.sidebar.number_input('Applicant Income', data['ApplicantIncome'].min(), data['ApplicantIncome'].max())
loan_amt = st.sidebar.number_input('Loan Amount', data['LoanAmount'].min(), data['LoanAmount'].max())
coapp_income = st.sidebar.number_input('CoApplicant Income', data['CoapplicantIncome'].min(), data['CoapplicantIncome'].max())
dep = st.sidebar.selectbox('Dependents', data['Dependents'].unique())
prop_area = st.sidebar.selectbox('Property Area', data['Property_Area'].unique())
cred_hist = st.sidebar.number_input('Credit History', data['Credit_History'].min(), data['Credit_History'].max())
loan_amt_term = st.sidebar.number_input('Loan Amount Term', data['Loan_Amount_Term'].min(), data['Loan_Amount_Term'].max())

#users input
input_var = pd.DataFrame()
input_var['ApplicantIncome'] = [app_income]
input_var['LoanAmount'] = [loan_amt]
input_var['CoapplicantIncome'] = [coapp_income]
input_var['Dependents'] = [dep]
input_var['Property_Area'] = [prop_area]
input_var['Credit_History'] = [cred_hist]
input_var['Loan_Amount_Term'] = [loan_amt_term]

st.markdown("<br>",unsafe_allow_html=True)
st.divider()
st.subheader('User Input')
st.dataframe(input_var, use_container_width=True)

app_income = joblib.load('ApplicantIncome_scaler.pki')
coapp_income = joblib.load('CoapplicantIncome_scaler.pki')
prop_area = joblib.load('Property_Area_encoder.pki')


# transform the users input with the imported scalers
input_var['ApplicantIncome'] = app_income.transform(input_var[['ApplicantIncome']])
input_var['CoapplicantIncome'] = coapp_income.transform(input_var[['CoapplicantIncome']])
input_var['Property_Area'] = prop_area.transform(input_var[['Property_Area']])

model = joblib.load('loanmodel.pki')

predict= model.predict(input_var)

if st.button('Check your Loan Status'):
    if predict[0] == 0:
        st.error(f'Unfortunately.... your loan request has been denied')
        st.image('pngwing.com (1)_decline.png', width=200)
    else:
        st.success(f'Congratulations... your loan request has been approved.')
        st.image('pngwing.com (1)_ approve.png', width=200)    
