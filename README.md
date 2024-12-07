# Streamlit_App_Churn_Analysis
 
# Telco Customer Churn Prediction Model App

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Features](#features)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Modeling](#modeling)
- [Evaluation Metrics](#evaluation-metrics)
- [Conclusion](#conclusion)


## Overview
The **Telco Customer Churn Prediction Model App** is a machine learning application that predicts whether a customer will churn (leave the service) based on their demographic information, account details, and usage data. Customer Lifetime Value (CLV) is also included to assess how valuable a customer is to the business. 

The goal is to help telecommunications companies take proactive steps in retaining customers who are likely to churn, potentially increasing overall customer retention rates and profitability.

## Project Structure

```plaintext
telco-churn-prediction-app/
│
├── data/
│   ├── telco_churn_data.csv       # Dataset used for training and testing
├── Models/
│   ├── premodel.pkl               # Preprocessing pipeline
│   ├── LogisticRegression.pkl     # Logistic Regression model
│   ├── RandomForest.pkl           # Random Forest model
│   ├── XGBoost.pkl                # XGBoost model
│   ├── KNearest.pkl               # K-Nearest Neighbors model
│   ├── DecisionTree.pkl           # Decision Tree model
├── app.py                         # Streamlit app for churn prediction
├── README.md                      # Project documentation
└── requirements.txt               # Required dependencies
```

## Dataset Description
The dataset includes various customer features collected by a telecom company, including demographic information, account information, and usage data, along with whether the customer churned or not. Additionally, Customer Lifetime Value (CLV) has been included as an extra feature, which helps to understand the potential future revenue from a customer.

### Features

| **Column Name**         | **Description** |
|-------------------------|-----------------|
| **Gender**               | The gender of the customer (Male, Female). |
| **SeniorCitizen**        | Whether the customer is a senior citizen (1) or not (0). |
| **Partner**              | Whether the customer has a partner (Yes, No). |
| **Dependents**           | Whether the customer has dependents (Yes, No). |
| **Tenure**               | Number of months the customer has stayed with the company. |
| **PhoneService**         | Whether the customer has phone service (Yes, No). |
| **MultipleLines**        | Whether the customer has multiple lines (Yes, No, No phone service). |
| **InternetService**      | Customer's internet service provider (DSL, Fiber optic, No). |
| **OnlineSecurity**       | Whether the customer has online security (Yes, No, No internet service). |
| **OnlineBackup**         | Whether the customer has online backup (Yes, No, No internet service). |
| **DeviceProtection**     | Whether the customer has device protection (Yes, No, No internet service). |
| **TechSupport**          | Whether the customer has tech support (Yes, No, No internet service). |
| **StreamingTV**          | Whether the customer has streaming TV service (Yes, No, No internet service). |
| **StreamingMovies**      | Whether the customer has streaming movies (Yes, No, No internet service). |
| **Contract**             | The contract term (Month-to-month, One year, Two year). |
| **PaperlessBilling**     | Whether the customer has opted for paperless billing (Yes, No). |
| **PaymentMethod**        | The payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)). |
| **MonthlyCharges**       | The amount charged to the customer monthly. |
| **TotalCharges**         | The total amount charged to the customer overall. |
| **Churn**                | Whether the customer churned (Yes, No). |
| **CLV**                  | Customer Lifetime Value - estimated potential value of the customer to the business. |

## Installation

### Requirements
To run this project, you need the following installed on your system:
- Python 3.7 or later
- Required dependencies listed in `requirements.txt`

### Installing Dependencies
You can install the required libraries using `pip`. Run the following command in the project directory:

```bash
pip install -r requirements.txt
```

## How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/telco-churn-prediction-app.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd telco-churn-prediction-app
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **App Interface:**
   - In the Streamlit app, you can input customer details manually to predict the churn likelihood for individual customers.
   - You can also upload a CSV file with customer data to perform bulk predictions.

## Modeling

### Preprocessing
A preprocessing pipeline has been used to handle:
- Encoding categorical features (e.g., Gender, Partner, Dependents).
- Scaling numerical features (e.g., Tenure, MonthlyCharges, TotalCharges).
- Handling missing values, if any.

### Models Used
The following models have been trained for predicting customer churn:
- **Logistic Regression**: Simple and interpretable linear model for classification.
- **Random Forest**: Ensemble model that combines multiple decision trees to improve prediction performance.
- **XGBoost**: Gradient boosting algorithm optimized for predictive accuracy and speed.
- **K-Nearest Neighbors**: Distance-based classification model.
- **Decision Tree**: Tree-based model that splits data on feature values to make predictions.

These models can be selected and used for predictions within the Streamlit app.

## Evaluation Metrics
To evaluate the performance of the churn prediction models, the following metrics were used:
- **Accuracy**: The proportion of correctly predicted churn instances.
- **Precision**: The ratio of true positive churn predictions to all positive predictions.
- **Recall (Sensitivity)**: The ratio of true positive churn predictions to all actual churn cases.
- **F1-Score**: Harmonic mean of precision and recall, providing a single performance score.
- **ROC-AUC**: The area under the ROC curve, which measures the model's ability to distinguish between classes.

These metrics provide insights into how well the model is performing, both in terms of identifying churners and avoiding false positives.

## Conclusion
The **Telco Customer Churn Prediction Model App** offers an efficient and scalable solution for telecom companies to predict customer churn. By understanding which customers are likely to leave, businesses can deploy targeted retention strategies to reduce churn and maximize Customer Lifetime Value (CLV). The app also provides insights into the factors driving churn, enabling proactive decision-making.
