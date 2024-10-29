import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ['GROQ_API_KEY'])


def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)


xgboost_model = load_model("xgb_model.pkl")
# xgboost_smote_model = load_model("xgb-smote-model.pkl")
# xgboost_feature_engineered_model = load_model("xgb-feature-engineered_model.pkl")
voting_classifier_model = load_model("voting_clf.pkl")
voting_classifier_hard_model = load_model("voting_hard_clf.pkl")
naive_bayes_model = load_model("nb_model.pkl")
random_forest_model = load_model("rf_model.pkl")
decision_tree_model = load_model("dt_model.pkl")
svm_model = load_model("svm_model.pkl")
knn_model = load_model("knn_model.pkl")


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):
  input_dict = {
      "CreditScore": credit_score,
      "Age": age,
      "Tenure": tenure,
      "Balance": balance,
      "NumOfProducts": num_products,
      "HasCrCard": int(has_credit_card),
      "IsActiveMember": int(is_active_member),
      "EstimatedSalary": estimated_salary,
      "Geography_France": 1 if location == "France" else 0,
      "Geography_Germany": 1 if location == "Germany" else 0,
      "Geography_Spain": 1 if location == "Spain" else 0,
      "Gender_Male": 1 if gender == "Male" else 0,
      "Gender_Female": 1 if gender == "Female" else 0
  }
  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict


def make_predictions(input_df, input_dict):
  probabilities = {
      'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
      "Random Forest": random_forest_model.predict_proba(input_df)[0][1],
      'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
  }

  avg_probability = np.mean(list(probabilities.values()))

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"{avg_probability:.2%} probability of churning")

  with col2:
    fig_probs = ut.create_model_prob_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)

  # st.markdown('### Model Probabilites')
  # for model, prob in probabilities.items():
  #   st.write(f"{model} {prob}")
  # st.write(f"Average Probability: {avg_probability}")

  return avg_probability


def explain_prediction(probability, input_dict, surname):
  prompt = f"""You are an expert data scientist at a bank, where you specialize in interpeting and explaining predictions of machine learning models.
  Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.
  Here is the customers information:
  {input_dict}
  Here are the machine learning model's top 10 most important features of predicting churn:
      Feature           | Importance
      -------------------------------
      NumOfProducts     | 0.323888
      IsActiveMember    | 0.164146
      Age               | 0.109550
      Geography_Germany | 0.091373
      Balance           | 0.052786
      Geography_France  | 0.045463
      Gender_Female     | 0.045283
      Geography_Spain   | 0.036585
      CreditScore       | 0.032655
      EstimatedSalary   | 0.032555
      HasCrCard         | 0.031940
      Tenure            | 0.030504
      Gender_Male       | 0.000000
  {pd.set_option('display.max_columns', None)}
  Here are summary statistics for churned customers:
  {df[df['Exited'] == 1].describe()}
  Here are summary statistics for non-churned customers:
  {df[df['Exited'] == 0].describe()}
  - If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning.
  - If the customer has less than 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.
  - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importance provided.
  Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features", just explain the prediction.
  """
  print("EXPLANATION PROMPT", prompt)
  raw_response = client.chat.completions.create(
      model="llama-3.2-3b-preview",
      messages=[{
          "role": "user",
          "content": prompt
      }],
  )
  return raw_response.choices[0].message.content

def generate_email(probablity, input_dict, explanation, surname):
  prompt = f"""
  You are a bank manager and you are responsible for ensuring customers stay with the bank. Make offers to incentivize customers staying if you must.

  You noticed a customer named {surname} has a {round(probablity*100, 1)}% chance of churning.

  This is the user's information:
  {input_dict}

  along with a potential explanation as to why this customer might churn:
  {explanation}

  Generate an email to the customer based on this information, asking them to stay (with incentives if necessary), or offering more incentives to they remain loyal to your bank.

  Make sure to list out these incentives based on their information in bullet point format. Do not ever mention the machine learning model or their proabability of churning.
  """

  raw_response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{
      "role": "user",
      "content": prompt
    }]
  )

  print("\n\n-------EMAIL PROMPT--------\n", prompt)

  return raw_response.choices[0].message.content



st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

selected_customer = st.selectbox("Select a customer", customers)

if selected_customer:
  selected_split = selected_customer.split(" - ")
  selected_customer_id = int(selected_split[0])

  selected_surname = selected_split[1]

  customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]

  col1, col2 = st.columns(2)

  with col1:
    credit_score = st.number_input("Credit Score",
                                   min_value=300,
                                   max_value=850,
                                   value=int(customer['CreditScore']))

    location = st.selectbox("Location", ["Spain", "France", "Germany"],
                            index=["Spain", "France",
                                   "Germany"].index(customer['Geography']))

    gender = st.radio("Gender", ["Male", "Female"],
                      index=0 if customer["Gender"] == "Male" else 1)

    age = st.number_input("Age",
                          min_value=18,
                          max_value=100,
                          value=int(customer["Age"]))

    tenure = st.number_input("Tenure (years)",
                             min_value=0,
                             max_value=50,
                             value=int(customer["Tenure"]))

  with col2:
    balance = st.number_input("Balance",
                              min_value=0.0,
                              value=float(customer["Balance"]))

    num_products = st.number_input("Num Products",
                                   min_value=1,
                                   max_value=10,
                                   value=int(customer["NumOfProducts"]))

    has_credit_card = st.checkbox("Credit Card?",
                                  value=bool(customer["HasCrCard"]))

    is_active_member = st.checkbox("Active Member?",
                                   value=bool(customer["IsActiveMember"]))

    estimated_salary = st.number_input("Est. Salary",
                                       min_value=0.0,
                                       value=float(
                                           customer["EstimatedSalary"]))

  input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                       tenure, balance, num_products,
                                       has_credit_card, is_active_member,
                                       estimated_salary)
  avg_probability = make_predictions(input_df, input_dict)

  explanation = explain_prediction(avg_probability, input_dict,
                                   customer["Surname"])
  st.markdown("---")
  st.subheader("Explanation of Prediction")
  st.markdown(explanation)


  email=generate_email(avg_probability, input_dict, explanation, customer["Surname"])
  st.markdown("---")
  st.subheader("Email")
  st.markdown(email)