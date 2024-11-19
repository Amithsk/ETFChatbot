import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector

# Load FinancialBERT model and tokenizer with pipeline
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = AutoModelForMaskedLM.from_pretrained('yiyanghkust/finbert-tone')

# Pipeline for masked language modeling
nlp_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Function to generate a response
def generate_response(user_input):
    # Masked language modeling to generate predictions
    user_input += " [MASK]"  # Add a mask token for prediction
    result = nlp_pipeline(user_input)
    return result[0]['sequence']  # Return the top prediction

# Function to fetch ETF data (example query from a MySQL database)
def get_etf_data(etf_name):
    # Connect to the MySQL database
    conn = mysql.connector.connect(
        host='your_host',
        user='your_user',
        password='your_password',
        database='your_database'
    )
    query = f"SELECT date, close_price FROM etf_data WHERE etf_name = '{etf_name}' ORDER BY date"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Function to plot ETF data
def plot_etf_data(df, etf_name):
    fig, ax = plt.subplots()
    ax.plot(df['date'], df['close_price'], label=f'{etf_name} Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'{etf_name} ETF Price History')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Streamlit UI
st.title("ETF Chatbot and Visualizer")

# User input for chatbot
user_input = st.text_input("Ask about ETFs or financial data:", "")

if user_input:
    response = generate_response(user_input)
    st.write(f"Chatbot: {response}")

# User input for ETF data visualization
etf_name = st.text_input("Enter ETF name (e.g., GOLDBEES):", "")

if etf_name:
    try:
        # Fetch ETF data and plot it
        etf_data = get_etf_data(etf_name)
        if not etf_data.empty:
            plot_etf_data(etf_data, etf_name)
        else:
            st.write(f"No data available for {etf_name}")
    except Exception as e:
        st.write(f"Error fetching data: {e}")
