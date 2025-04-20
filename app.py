import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Churn Predictor", page_icon="📉")

@st.cache_resource
def load_scaler(path: str = 'Model/scaler.pkl'):
    return joblib.load(path)

@st.cache_resource
def load_model(path: str = 'Model/svm_churn_model.pkl'):
    return joblib.load(path)


scaler = load_scaler()
model  = load_model()


def preprocess_input(data: dict) -> pd.DataFrame:
    # Map categorical to numeric
    data['Geography'] = {'France': 2, 'Germany': 1, 'Spain': 0}[data['Geography']]
    data['Gender']    = {'Female': 1, 'Male': 0}[data['Gender']]
    data['Num Of Products'] = {1: 0, 2: 1, 3: 1, 4: 1}[data['Num Of Products']]
    df_new = pd.DataFrame([data])
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']
    df_new[num_cols] = scaler.transform(df_new[num_cols])
    cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'Num Of Products', 'Has Credit Card', 'Is Active Member', 'Estimated Salary', 'Zero Balance']
    return df_new[cols]


st.title("Bank Customer Churn Predictor")
st.markdown("Enter customer information to predict churn.")


CreditScore    = st.number_input('Credit Score', 350, 850, 600)
Geography      = st.selectbox('Geography', ['France','Germany','Spain'])
Gender         = st.selectbox('Gender', ['Male','Female'])
Age            = st.slider('Age', 18, 100, 30)
Tenure         = st.slider('Tenure (years)', 0, 10, 3)
Balance        = st.number_input('Account Balance', 0.0, 250000.0, 0.0)
NumOfProducts  = st.selectbox('Number of Products', [1,2,3,4])
HasCrCard      = st.selectbox('Has Credit Card', [0,1])
IsActiveMember = st.selectbox('Is Active Member', [0,1])
EstimatedSalary= st.number_input('Estimated Salary', 0.0, 250000.0, 0.0)


if st.button('Predict Churn'):
    inp = {'CreditScore': CreditScore, 'Geography': Geography, 'Gender': Gender,
           'Age': Age, 'Tenure': Tenure, 'Balance': Balance,
           'Num Of Products': NumOfProducts, 'Has Credit Card': HasCrCard,
           'Is Active Member': IsActiveMember, 'Estimated Salary': EstimatedSalary, 
           'Zero Balance': (1 if Balance > 0 else 0)}
    df_in = preprocess_input(inp)
    pred = model.predict(df_in)[0]
    st.subheader('Prediction:')
    st.write('This Customer is likely to Churn' if pred == 1 else 'This Customer is not likely to Churn')
    #st.write(pred)
    # score = model.decision_function(df_in)[0]
    # st.subheader('Decision Score:')
    # st.write(f"{score:.4f}")
