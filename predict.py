import streamlit as st
import pickle
import pandas as pd
import os

# Function to load the pre-trained pipeline (premodel.pkl) from the Models directory
@st.cache_resource
def load_pipeline():
    pipeline_path = os.path.join(os.getcwd(), "Models", "premodel.pkl")  # Dynamic path for compatibility

    # Attempt to load the pipeline, handle exceptions if any occur
    if os.path.exists(pipeline_path):
        try:
            with open(pipeline_path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            st.error(f"An error occurred while loading the pipeline: {e}")
            return None
    else:
        st.error(f"premodel.pkl not found at {pipeline_path}")
        return None

# Function to load individual models from the Models folder
@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            st.error(f"An error occurred while loading the model: {e}")
            return None
    else:
        st.error(f"{model_path} not found.")
        return None

# Main function for the prediction page
def predict_page():
    st.title("PREDICT EXECUTION")  # Page title
    st.sidebar.title("Predict Section")  # Sidebar title
    st.sidebar.write('''This section allows users to input customer data and
                     receive predictions based on a trained machine learning model.
                     ''')

    # Load the pre-trained pipeline
    pipeline = load_pipeline()
    if pipeline is None:
        return  # Stop execution if pipeline loading fails

    # Dictionary of available models and their file paths
    models_paths = {
        'Logistic Regression': os.path.join("Models", "LR.pkl"),
        'Random Forest': os.path.join("Models", "RF.pkl"),
        'XGB': os.path.join("Models", "XGB.pkl"),
        'K Nearest': os.path.join("Models", "K-Nearest Neighbors.pkl"),
        'Decision Tree': os.path.join("Models", "Decision_Tree.pkl")
    }

    # Select model from dropdown and load it
    model_choice = st.selectbox("Select a model", list(models_paths.keys()))
    model = load_model(models_paths[model_choice])
    if model is None:
        return  # Stop if model loading fails

    st.write(f"Loaded model type: {type(model)}")  # Display model type

    # Section for single customer prediction input
    st.subheader("Single Customer Prediction")
    gender = st.selectbox("Gender", ['Male', 'Female'])
    senior_citizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)
    phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])

    # Prediction for a single customer when the "Predict" button is pressed
    if st.button("Predict Single"):
        # Create DataFrame from the input values for the single customer
        data = pd.DataFrame({
            'Gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'Tenure': [tenure],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract]
        })

        # Use the pipeline to predict and calculate churn probability
        prediction = pipeline.predict(data)[0]
        probability = pipeline.predict_proba(data)[0][1] * 100

        # Display the prediction and probability results
        st.write(f"Prediction: {'Churn' if prediction == 1 else 'Not Churn'}")
        st.write(f"Churn Probability: {probability:.2f}%")

    # Section for bulk prediction
    st.header("Bulk Prediction")
    st.write("Upload a CSV file with customer data for bulk prediction")

    upload_file = st.file_uploader("Choose the file to upload", type='csv')  # File upload widget
    if upload_file is not None:
        try:
            bulk_data = pd.read_csv(upload_file)  # Read CSV data
            st.write("Data Preview", bulk_data.head())  # Display first few rows

            # Required columns for the bulk prediction
            required_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                'MonthlyCharges', 'TotalCharges'
            ]

            # Check if all required columns are present in the uploaded file
            if all(col in bulk_data.columns for col in required_columns):
                bulk_predictions = pipeline.predict(bulk_data)  # Predict churn for all records
                bulk_probability = pipeline.predict_proba(bulk_data)[:, 1] * 100  # Churn probabilities

                # Create a copy of the data with predictions and probabilities
                bulk_results = bulk_data.copy()
                bulk_results["Predictions"] = ['Churn' if pred == 1 else 'Not Churn' for pred in bulk_predictions]
                bulk_results['Churned Probability'] = bulk_probability

                # Display bulk prediction results
                st.write("Bulk Prediction Results:")
                st.dataframe(bulk_results)

                # Save results to a CSV file
                result_file = "data/bulk_predictions.csv"
                bulk_results.to_csv(result_file, index=False)
                st.success(f"Results saved successfully to {result_file}")
            else:
                st.error("The uploaded CSV does not have the required columns.")
        except Exception as e:
            st.error(f"Error during bulk prediction: {e}")

# Entry point for Streamlit app
if __name__ == "__main__":
    predict_page()
