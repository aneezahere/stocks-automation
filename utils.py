from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import openai
import os

def init_embeddings_model():
    """Initialize the sentence transformer model"""
    try:
        print("Initializing embeddings model...")
        return SentenceTransformer('all-mpnet-base-v2')
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

def init_pinecone_client():
    """Initialize Pinecone client"""
    try:
        print("Initializing Pinecone client...")
        pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        return pc.Index("stocks")
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return None

def search_stocks(index, model, query: str, top_k: int = 5) -> List[Dict]:
    """Search for stocks based on natural language query"""
    try:
        if not index or not model:
            raise ValueError("Index or model is not initialized.")

        # Get query embeddings
        query_embedding = model.encode(query).tolist()
        print("Query embeddings generated successfully.")

        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace="stock-descriptions",
            include_metadata=True
        )

        formatted_results = []
        if hasattr(results, 'matches'):
            for match in results.matches:
                metadata = match.metadata
                if metadata:
                    formatted_results.append({
                        'Business Summary': metadata.get('Business Summary', 'N/A'),
                        'City': metadata.get('City', 'N/A'),
                        'Country': metadata.get('Country', 'N/A'),
                        'Industry': metadata.get('Industry', 'N/A'),
                        'Name': metadata.get('Name', 'N/A'),
                        'Sector': metadata.get('Sector', 'N/A'),
                        'State': metadata.get('State', 'N/A'),
                        'Ticker': metadata.get('Ticker', 'N/A')
                    })
        return formatted_results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_comparison_analysis(query: str, companies: list) -> str:
    """Generate a comparative analysis for the identified companies"""
    try:
        # Build a detailed prompt
        prompt = f"""
        Based on the user's query: "{query}", we have identified the following companies:
        {', '.join([company['Name'] for company in companies])}.

        For each of these companies, fetch the following metrics:
        - Market Capitalization
        - Revenue Growth
        - Industry Sector
        - Any other key performance indicators (KPIs)

        Compare these companies with similar companies listed on the New York Stock Exchange (NYSE) in the same sector. Provide:
        1. A business summary for each company.
        2. A comparison of financial metrics with NYSE benchmarks.
        3. A detailed investment insight and recommendation, highlighting potential opportunities and risks.

        Ensure your response is concise and includes relevant data points.
        """

        # Query OpenAI for analysis
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a stock market analysis assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error generating analysis: {e}"
