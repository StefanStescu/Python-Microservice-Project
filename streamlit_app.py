import streamlit as st  # type: ignore
from typing import List
import requests
import os
from PIL import Image
import base64

# Set up the Streamlit page configuration
# This sets the title and layout of the Streamlit app
st.set_page_config(page_title="Sleep Quality Dashboard", layout="centered")

st.title("😴 Sleep Quality Analyzer")
st.markdown("This interface runs analyses on your local dataset via the FastAPI backend.")

# Config
# Define the filename and API URL
# This is the dataset that will be analyzed by the FastAPI backend
# HARD-CODED: The filename of the dataset to be analyzed as the scripts would not run otherwise on another csv file
FILENAME = "Sleep_health_and_lifestyle_dataset.csv"
API_URL = "http://127.0.0.1:8000/analyze"

# Info
# Display information about the dataset
# This provides context to the user about the dataset being used
st.info(f"Using dataset: `{FILENAME}`")

# Analyze button
# This button triggers the analysis of the dataset by sending it to the FastAPI backend
# If the button is clicked, it will upload the dataset and display the results
# Divider
# This adds a horizontal line to separate sections of the Streamlit app
st.markdown("---")
if st.button("📊 Run Analyze"):
    try:
        with open(FILENAME, "rb") as f:
            files = {"file": (FILENAME, f)}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Analysis completed")

            st.markdown(f"**Filename:** `{result['filename']}`")
            st.markdown(f"**Valid Records:** `{result['valid_records']}`")
            st.markdown(f"**Shape:** {result['shape']}")
            st.markdown("**Columns:**")
            st.write(result["columns"])

            st.subheader("🔎 Column Metadata")
            for col, dtype in result["column_info"].items():
                col_type = "Numerical" if "float" in dtype or "int" in dtype else "Categorical"
                role = "Dependent" if "Stress" in col else "Independent"
                st.write(f"**{col}** → {col_type}, {role} variable (dtype: `{dtype}`)")
        else:
            st.error(f"❌ API Error: {response.status_code} - {response.text}")


    except FileNotFoundError:
        st.error(f"❌ Could not find `{FILENAME}`.")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")

# Divider
# This adds a horizontal line to separate sections of the Streamlit app
st.markdown("---")

# Plot button
# This button triggers the plotting functionality in the CLI module
# If the button is clicked, it will run the plotting command from the CLI module
if st.button("📊 Show Plots"):
    import subprocess
    subprocess.run(["python", "cli.py", "plot", "--file", FILENAME, "--save"])
    st.success("✅ Plots generated and saved in current directory.")

# Divider
# This adds a horizontal line to separate sections of the Streamlit app
st.markdown("---")

# Histograms buttons
# This section allows the user to view histograms of specific columns
# It checks if the histogram images exist and displays them if they do
if st.button("📸 View All Plots"):
    plot_folder = "plots"
    if os.path.exists(plot_folder):
        for filename in sorted(os.listdir(plot_folder)):
            if filename.endswith(".png"):
                path = os.path.join(plot_folder, filename)
                st.image(Image.open(path), caption=filename)
    else:
        st.warning("⚠️ Plot folder not found.")


# Divider
# This adds a horizontal line to separate sections of the Streamlit app
st.markdown("---")

# Regression
# This button triggers the regression analysis endpoint in the FastAPI backend
# If the button is clicked, it will run the regression analysis and display the result
if st.button("📈 Run Regression"):
    response = requests.get("http://127.0.0.1:8000/regression")
    if response.status_code == 200:
        result = response.json()
        st.success("✅ Regression completed")

        st.subheader("📉 Simple Linear Regression")
        st.code(result["simple_summary"])

        st.subheader("📈 Multiple Linear Regression")
        st.code(result["multiple_summary"])

        st.subheader("🌀 Quadratic Regression")
        st.code(result["quadratic_summary"])

        st.subheader("📊 ANOVA Comparison")
        st.code(result["anova"])

    else:
        st.error("❌ Failed to run regression.")

# Divider
# This adds a horizontal line to separate sections of the Streamlit app
st.markdown("---")

# Diagnostics
# This button triggers the diagnostics endpoint in the FastAPI backend
# If the button is clicked, it will run diagnostics on the dataset and display the result
if st.button("🧪 Run Diagnostics"):
    response = requests.get("http://127.0.0.1:8000/diagnostics")
    if response.status_code == 200:
        result = response.json()
        st.success("✅ Diagnostics completed")

        st.subheader("White Test")
        st.json(result["White Test"])

        st.subheader("Breusch-Pagan Test")
        st.json(result["Breusch-Pagan"])

        st.subheader("Durbin-Watson")
        st.write(result["Durbin-Watson"])

        st.subheader("Jarque-Bera")
        st.json(result["Jarque-Bera"])
    else:
        st.error("❌ Failed to run diagnostics.")

# Divider
# This adds a horizontal line to separate sections of the Streamlit app
st.markdown("---")