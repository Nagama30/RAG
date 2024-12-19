import os
import google.generativeai as genai
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import SimpleDirectoryReader
from llama_index.core.settings import Settings

# Set your Gemini API key
genai.configure(api_key=os.environ["API_KEY"])

# Override Global Settings to Disable OpenAI Defaults
#Settings.llm = None  # Disable OpenAI LLM globally

# Load Document
def load_documents(input_files):
    return SimpleDirectoryReader(input_files=input_files).load_data()

def naive_rag_demo():
    # File paths to the documents
    input_files = ["./eBook-How-to-Build-a-Career-in-AI.pdf"]

    # Load documents
    documents = load_documents(input_files)

    # Initialize Gemini LLM and Embeddings
    gemini_llm = Gemini(model="models/gemini-1.5-pro-latest", temperature=0.1)
    embed_model = GeminiEmbedding()

    Settings.llm = gemini_llm

    # Create Vector Database Index with Explicit Embedding Model and LLM
    index = VectorStoreIndex.from_documents(
        documents, 
        embed_model=embed_model,
        llm=gemini_llm  # Pass Gemini LLM explicitly
    )

    # Query Engine with Explicit Gemini LLM
    query_engine = index.as_query_engine(llm=gemini_llm)
    response = query_engine.query("What is key to build career in AI?")

    print("Response:", response)

# Run the demo
if __name__ == "__main__":
    print("Running Naive RAG...")
    naive_rag_demo()
