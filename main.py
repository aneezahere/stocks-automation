import streamlit as st
from utils import init_embeddings_model, init_pinecone_client, search_stocks, generate_comparison_analysis
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Stock Research Automation", layout="wide")

# Styling for dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    .stTextInput > div {
        background-color: #333;
        color: white;
        border: none;
        padding: 8px;
        border-radius: 5px;
    }
    .stButton > button {
        background-color: white;
        color: black;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 14px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ddd;
    }
    .result-box {
        background-color: #333;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-box h3 {
        color: white;
        margin-bottom: 10px;
    }
    .result-box p {
        color: #ccc;
        margin: 5px 0;
    }
    footer {display: none;}
    header {display: none;}
    </style>
""", unsafe_allow_html=True)

# Initialize services
model = init_embeddings_model()
index = init_pinecone_client()

# Header
st.markdown('<h1 style="text-align: center; color: white;">Stock Research Automation</h1>', unsafe_allow_html=True)

# Search box
query = st.text_input("Enter your search query", placeholder="Example: companies that build data centers")

# Submit button
if st.button("Submit"):
    if query:
        st.markdown("<p style='color: white;'>Searching, please wait...</p>", unsafe_allow_html=True)
        results = search_stocks(index, model, query)

        if results:
            # Display Results
            st.markdown("<h2 style='color: white;'>Results</h2>", unsafe_allow_html=True)
            for result in results:
                st.markdown(f"""
                <div class="result-box">
                    <h3>{result['Name']} ({result['Ticker']})</h3>
                    <p><strong>Business Summary:</strong> {result['Business Summary']}</p>
                    <p><strong>City:</strong> {result['City']}</p>
                    <p><strong>State:</strong> {result['State']}</p>
                    <p><strong>Country:</strong> {result['Country']}</p>
                    <p><strong>Industry:</strong> {result['Industry']}</p>
                    <p><strong>Sector:</strong> {result['Sector']}</p>
                </div>
                """, unsafe_allow_html=True)

            # Generate AI-powered comparison and summary
            st.subheader("Summary")
            with st.spinner("Generating insights..."):
                analysis = generate_comparison_analysis(query, results)
                st.markdown(analysis)
        else:
            st.markdown("<p style='color: white;'>No matching companies found.</p>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a search query.")
