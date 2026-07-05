#!/bin/bash
# ---------------------------------------------------------
# Systemic Tau v4.1.5-Pro - Startup Script for Mac/Linux
# ---------------------------------------------------------

# Navigate to the directory where this script is located
cd "$(dirname "$0")"

echo "Initializing Systemic Tau v4.1.5-Pro..."

# Check if pip dependencies are needed (optional, assuming they are installed)
# pip install -r requirements.txt

# Run the Streamlit app
echo "Launching Streamlit server..."
streamlit run app.py
