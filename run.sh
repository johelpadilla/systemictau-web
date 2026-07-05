#!/bin/bash
echo "Installing requirements..."
pip3 install -r requirements.txt
echo "Starting Complete Systemic Tau Streamlit App..."
streamlit run app.py
