# app.py
import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.title("Predictive Maintenance App")

    # Load your dataset
    data_path = "machininfo.csv"
    df = load_data(data_path)

    if df is not None:
        # Display the dataset
        st.write("### Dataset Preview")
        st.write(df.head())

        # Generate a profile report (optional)
        report = ProfileReport(df, minimal=True)
        st.write(report)

        # Add your predictive maintenance code here
        # For example, you can use scikit-learn to train a model

        # Display predictions or other relevant information

if __name__ == "__main__":
    main()