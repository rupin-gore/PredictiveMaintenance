import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    # Title of the web app
    st.title("Predictive Maintenance App")

    # Load the dataset
    df = load_dataset()

    # Preprocess the data
    df = preprocess_data(df)

    # Train the model
    model = train_model(df)

    # Create dashboard layout
    st.sidebar.title("Dashboard")

    # Add components to the sidebar
    st.sidebar.subheader("Navigation")
    selected_page = st.sidebar.radio("", ["Overview", "Dataset Details", "Data Visualization", "Predict"])

    # Page content
    if selected_page == "Overview":
        st.subheader("Overview")
        st.write("Welcome to the Predictive Maintenance App!")
        st.write("This dashboard provides an overview of the dataset and trained model.")

    elif selected_page == "Dataset Details":
        st.subheader("Dataset Details")
        st.write("### Dataset Preview:")
        st.write(df.head())

        with st.expander("Details"):
            st.write("#### Dataset Shape:")
            st.write(df.shape)

            st.write("#### Dataset Description:")
            st.write(df.describe())

    elif selected_page == "Data Visualization":
        st.subheader("Data Visualization")
        st.write("### Select Visualization Type:")

        # Add visualization options here

    elif selected_page == "Predict":
        st.subheader("Predictive Maintenance")

        # Input form for user to enter metrics
        st.write("### Enter Input Metrics:")
        input_metrics = {}
        for column in df.columns:
            if column != 'Failure Type':  # Exclude target column
                input_metrics[column] = st.number_input(f"{column}", value=0)

        # Make prediction
        if st.button("Predict"):
            input_data = pd.DataFrame([input_metrics])
            prediction = model.predict(input_data)

            # Decode predicted failure type
            custom_label_encoder = CustomLabelEncoder()
            custom_label_encoder.fit(df['Failure Type'])
            prediction_label = custom_label_encoder.inverse_transform(prediction)

            st.write("### Prediction:")
            st.write(prediction_label)

if __name__ == "__main__":
    main()

