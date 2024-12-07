import streamlit as st
#from PIL import Image


# Define the home function
def home_page():
    st.title("Embedded a ML model in GUI's --used Streamlit")

    #image = Image.open(r'C:\Users\GAMING\Desktop\Streamlit_Project\Logo.png')
    st.image('https://www.google.com/url?sa=i&url=https%3A%2F%2Fparcusgroup.com%2Ftelecom-customer-churn-prediction-models%2F&psig=AOvVaw26eHwTPM1BLnsRcwOoPlM9&ust=1733608994868000&source=images&cd=vfe&opi=89978449&ved=2ahUKEwiw2ejzkpSKAxXjbkEAHT-QAusQjRx6BAgAEBk',
             caption='Telco Churn Dashboard', 
             use_container_width=True)

   # Project description
    st.markdown("""This app uses machine learning to classify whether a customer is likely to churn or not.""")

    # Instructions for the app usage
    st.subheader("Instructions")
    st.markdown("""
    - Upload a CSV file with customer data
    - Select the features you want to use for classification
    - Choose a machine learning model from the dropdown
    - Click on 'Classify' to get the predicted results
    - The app gives you a report on the performance of the model
    - Expect it to give metrics like F1 score, recall, precision, and accuracy
    """)

    st.header("App features")
    st.markdown("""
    - **Data View**: Access the cutomer data.
    - **Predict View**: Shows the various models and predictions you will make            
    - **Dashboard**: Shows data visualization for insights.                  
    """)

    st.subheader("User Benefits")
    st.markdown("""
    - **Data Driven Decision**: You make an informed decision backed by data.
    - **Access Machine Learning**:Utilize machine learning algorithms.
                                       
    """)

    st.write("### How to Run the Application")
    with st.container(border=True):
        st.code("""
        # Actvate the virtual environment       
        env/scripts/activate 
                
        # Run the App        
        streamlit run app.py
        """)

    #Adding the embedded link
    #st.video("https://www.youtube.com/watch?v=vmf4R1Xk_GM",start_time=0,autoplay=True)

    #Addinga clickablelink
    #st.markdown("[Watch a Demo](https://www.youtube.com/watch?v=vmf4R1Xk_GM)")


# way 2
    st.divider()
    st.write("+++" * 15)

    st.write("Need Help")
    st.write("Contact me on:")

# add an image / way
    st.markdown("[Visit Linkedln Profile](http://www.linkedin.com/in/william-dzorgenu)")