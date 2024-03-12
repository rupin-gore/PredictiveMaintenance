# predictive_maintenance_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Predictive Maintenance App")

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load your dataset
data_path = "machininfo.csv"
df = load_data(data_path)

if df is not None:
    # Display the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.style.highlight_max(axis=0))
    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())


    # Temperature Conversion Function
def convert_temperature(df):
    df["Air temperature [°C]"] = df["Air temperature [K]"] - 272.15
    df["Process temperature [°C]"] = df["Process temperature [K]"] - 272.15
    df["Temperature difference [°C]"] = df["Process temperature [°C]"] - df["Air temperature [°C]"]
    df.drop(columns=["Air temperature [K]", "Process temperature [K]"], inplace=True)
    return df

# Convert Temperature
converted_df = convert_temperature(df)

# Display Converted DataFrame
st.subheader("Converted Temperature from K to C in DataFrame")
st.write(converted_df)

# Display Histograms
st.subheader("Column Trends - Histograms")

# Create subplots for all histograms
fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(8, 6 * len(df.columns)))

# Plot histograms using matplotlib and seaborn
for i, column in enumerate(df.columns):
    sns.histplot(df[column], ax=axes[i], kde=True)
    axes[i].set_title(f"Histogram for {column}")

# Adjust layout and spacing
plt.tight_layout()

# Display the entire figure using st.pyplot
st.pyplot(fig)
if __name__ == "__main__":
    main()