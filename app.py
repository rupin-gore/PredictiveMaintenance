from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns

class CustomLabelEncoder:
    def __init__(self):
        self.label_mapping = {}

    def fit(self, labels):
        unique_labels = set(labels)
        self.label_mapping = {label: i for i, label in enumerate(unique_labels)}

    def transform(self, labels):
        return [self.label_mapping[label] for label in labels]

    def inverse_transform(self, encoded_labels):
        return [label for i, label in sorted((i, label) for label, i in self.label_mapping.items())]

# Function to load dataset
def load_dataset():
    df = pd.read_csv('machininfo.csv')
    return df

# Function to preprocess data
# Function to preprocess data
def preprocess_data(df):
    # Encoding categorical features
    custom_label_encoder = CustomLabelEncoder()
    custom_label_encoder.fit(df['Type'])
    df['Type'] = custom_label_encoder.transform(df['Type'])

    # Drop unnecessary columns
    df.drop(columns=['UDI', 'Product ID'], inplace=True)

    return df

# Function to train Random Forest Classifier
def train_model(df):
    X = df.drop(columns=["Failure Type"], axis=1)
    y = df["Failure Type"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    return model

# Display prediction form and result
def display_prediction_form(df, model, label_encoder):
    st.subheader("Predictive Maintenance")
    
    # Input form for user to enter metrics
    st.write("### Enter Input Metrics:")
    input_metrics = {}
    for column in df.columns:
        if column != 'Failure Type':  # Exclude target column
            input_metrics[column] = st.number_input(f"{column}", value=0)
    
    # Make prediction if user clicks the button
    if st.button("Predict"):
        input_data = pd.DataFrame([input_metrics])
        prediction = model.predict(input_data)
        prediction_label = label_encoder.inverse_transform(prediction)
        
        st.write("### Prediction:")
        st.success(prediction_label[0])

def main():
    # Title of the web app
    st.title("Predictive Maintenance App")

    # Load the dataset
    df = load_dataset()

    # Preprocess the data
    df = preprocess_data(df)

    # Train the model
    model = train_model(df)
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Failure Type'])

    # Create dashboard layout
    st.sidebar.title("Dashboard")

    # Add components to the sidebar
    st.sidebar.subheader("Navigation")
    selected_page = st.sidebar.radio("", ["Overview", "Dataset Details", "Data Visualization", "Predict"])

    # Page content
    if selected_page == "Overview":
        st.subheader("Overview")
        st.write("Welcome to the Predictive Maintenance App!")

        st.write("### Introduction to Predictive Maintenance:")
        st.write("Predictive maintenance (PdM) is a proactive maintenance strategy utilized across various industries to predict when equipment failures are likely to occur based on data analysis and machine learning algorithms. Unlike traditional reactive maintenance approaches, which involve fixing equipment after it fails, predictive maintenance aims to prevent failures by identifying potential issues before they escalate, thereby minimizing downtime, reducing maintenance costs, and improving operational efficiency.")

        st.write("### How Predictive Maintenance Works:")
        st.write("Predictive maintenance relies on the continuous monitoring of equipment health and the analysis of various data sources, including sensor data, equipment logs, and historical maintenance records. Machine learning algorithms are then employed to analyze this data and identify patterns or anomalies indicative of potential failures. By leveraging advanced analytics techniques, predictive maintenance models can forecast when equipment failures are likely to occur, allowing maintenance teams to take proactive measures such as scheduling maintenance tasks or replacing components before they fail.")

        st.write("### The Role of the Predictive Maintenance App:")
        st.write("The Predictive Maintenance App serves as a user-friendly interface for leveraging predictive maintenance techniques to anticipate equipment failures. It empowers maintenance teams and equipment operators to input relevant data, such as sensor readings, operating conditions, and historical performance metrics, and receive real-time predictions regarding the likelihood of equipment failure and the expected type of failure. By providing actionable insights at their fingertips, the app enables users to optimize maintenance schedules, allocate resources more effectively, and minimize unplanned downtime, ultimately enhancing overall operational efficiency and reliability.")

        st.write("### Key Features of the App:")
        st.write("- User-friendly interface for inputting equipment metrics and parameters.")
        st.write("- Real-time predictions of equipment failure likelihood and type based on machine learning models.")
        st.write("- Visualization of historical equipment performance data and failure trends.")
        st.write("- Integration with existing maintenance systems for seamless workflow management.")
    
        st.write("### Disclaimer:")
        st.write("While predictive maintenance techniques can provide valuable insights and help optimize maintenance activities, it is important to note that they are not foolproof. Predictions are based on historical data and statistical models, and there may be factors that influence equipment performance beyond the scope of the model. Users should use the predictions as a tool to inform decision-making and supplement their existing maintenance practices.")


    elif selected_page == "Dataset Details":
        st.subheader("Dataset Details")

        # Display dataset preview
        st.write("### Dataset Preview:")
        st.write(df.head())

        # Display dataset shape and description
        st.write("### Dataset Information:")
        st.write(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
        st.write("")

        # Display data types and unique values for each column
        st.write("### Column Information:")
        for column in df.columns:
            st.write(f"**{column}:**")
            st.write(f"  - Data Type: {df[column].dtype}")
            st.write(f"  - Number of Unique Values: {df[column].nunique()}")
            st.write("")

        # Display summary statistics for numerical columns
        numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_columns:
            st.write("### Summary Statistics for Numerical Columns:")
            st.write(df[numerical_columns].describe())
            st.write("")

    elif selected_page == "Data Visualization":
        st.subheader("Data Visualization")
        
        # Create a new Matplotlib figure
        st.write("### Histogram of Target Variable:")
        fig = plt.figure(figsize=(20, 15))
        df.hist(ax=fig.gca())
        st.pyplot(fig)

        # Setting up theme for data visualization in seaborn
        sns.set_theme(palette='tab10', font='Times New Roman', font_scale=1.5, rc=None)

        # Setting dark background for plots
        plt.style.use('dark_background')

        # Density plot for Air temperature
        st.write("### Density Plot of Air Temperature:")
        fig = sns.displot(data=df, x="Air temperature [K]", kde=True, bins=100, color="red", facecolor="cyan", height=5, aspect=3.5)
        st.pyplot(fig)

         # Density plot for Process temperature
        st.write("### Density Plot of Process Temperature:")
        fig = sns.displot(data=df, x="Process temperature [K]", kde=True, bins=100, color="red", facecolor="lime", height=5, aspect=3.5)
        st.pyplot(fig)

        # Sorting machine quality types (Low, Medium, High)
        st.write("### Sorting Machine Quality Types")
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        # Countplot
        sns.countplot(x='Type', data=df, ax=axes[0])
        axes[0].bar_label(axes[0].containers[0])
        axes[0].set_title("Type", fontsize=20, color='Red', font='Times New Roman')
        # Pie chart
        df['Type'].value_counts().plot.pie(explode=[0.1, 0.1, 0.1], autopct='%1.2f%%', shadow=True, ax=axes[1])
        axes[1].set_title("Type", fontsize=20, color='Red', font='Times New Roman')
        # Adjust layout
        plt.tight_layout()
        # Show the plots
        st.pyplot(fig)

    elif selected_page == "Predict":
        display_prediction_form(df, model, label_encoder)

if __name__ == "__main__":
    main()