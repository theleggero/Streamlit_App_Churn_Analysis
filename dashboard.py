import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


def dashboard_page():
    st.title("Customer Churn Dashboard")

    # Load data
    data_file = os.path.join(os.getcwd(), "data", "train_data.csv")
    data = pd.read_csv(data_file)

    # Sidebar filters
    st.sidebar.title("Filters")
    filters = {
        "gender": st.sidebar.multiselect("Gender", options=data["gender"].unique(), default=data["gender"].unique()),
        "InternetService": st.sidebar.multiselect("Internet Service", options=data["InternetService"].unique(), default=data["InternetService"].unique()),
        "Contract": st.sidebar.multiselect("Contract Type", options=data["Contract"].unique(), default=data["Contract"].unique()),
        "PaymentMethod": st.sidebar.multiselect("Payment Method", options=data["PaymentMethod"].unique(), default=data["PaymentMethod"].unique())
    }

    # Filter data based on user selections
    def apply_filters(data, filters):
        for column, selected_values in filters.items():
            data = data[data[column].isin(selected_values)]
        return data

    # Plot helper for stacked bar charts
    def plot_stacked_bar(data, column, hue_column, colors, title, ylabel):
        crosstab = pd.crosstab(data[column], data[hue_column], normalize="index")
        crosstab.rename({0: "Not Churned", 1: "Churned"}, axis=1, inplace=True)
        fig, ax = plt.subplots(facecolor="#f5f5f5")
        crosstab.plot(kind="bar", stacked=True, ax=ax, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        return fig

    # Filtered data
    filtered_data = apply_filters(data, filters)

    # Display summary
    st.write("### Summary of Filtered Data")
    st.write(filtered_data.describe())

    # Churn distribution
    st.write("### Churn Distribution")
    churn_counts = filtered_data["Churn"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%", startangle=90, colors=["#ff9999", "#66b3ff"])
    ax.axis("equal")
    st.pyplot(fig)

    # Monthly charges distribution
    st.write("### Monthly Charges Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_data["MonthlyCharges"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # Scatter plot: Total Charges vs Tenure
    st.write("### Total Charges vs Tenure")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_data, x="tenure", y="TotalCharges", hue="Churn", ax=ax)
    st.pyplot(fig)

    # Stacked bar charts
    st.write("### Churn Rate by Contract Type")
    fig = plot_stacked_bar(filtered_data, "Contract", "Churn", ["black", "red"], "Churn Rate by Contract Type", "Churn Rate")
    st.pyplot(fig)

    st.write("### Churn Rate by Payment Method")
    fig = plot_stacked_bar(filtered_data, "PaymentMethod", "Churn", ["blue", "pink"], "Churn Rate by Payment Method", "Churn Rate")
    st.pyplot(fig)

    st.write("### Churn Rate by Internet Service")
    fig = plot_stacked_bar(filtered_data, "InternetService", "Churn", ["lightgreen", "lightblue"], "Churn Rate by Internet Service", "Churn Rate")
    st.pyplot(fig)
