# Advanced RAG Implementation
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core import Document
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
import os
import google.generativeai as genai
from llama_index.core.settings import Settings

# Set your Gemini API key
genai.configure(api_key = os.environ["API_KEY"])

# Load Document
def load_documents(input_files):
    return SimpleDirectoryReader(input_files=input_files).load_data()
    

def advanced_rag_demo():
    # File paths to the documents
    input_files = ["./eBook-How-to-Build-a-Career-in-AI.pdf"]

    # Load documents
    documents = load_documents(input_files)

    # Merge document text
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    # Initialize LLM and Embeddings
    gemini_llm = Gemini(model="models/gemini-1.5-pro-latest", temperature=0.1)
    embed_model = GeminiEmbedding()

    Settings.llm = gemini_llm
    
    # Sentence Window Index Builder
    def build_sentence_window_index(documents, llm, embed_model, sentence_window_size=3):
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=sentence_window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        sentence_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, llm=gemini_llm, node_parser=node_parser,)
        return sentence_index

    # Build Sentence Window Index
    index = build_sentence_window_index([document], llm=gemini_llm, embed_model=embed_model)

    # Query Engine with Postprocessing
    def get_sentence_window_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(top_n=rerank_top_n, model="BAAI/bge-reranker-base")

        sentence_window_engine = sentence_index.as_query_engine(
            similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
        )
        return sentence_window_engine

    query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)
    response = query_engine.query("What are the keys to building a career in AI?")

    print("Advanced Response:", response)

# Run both demos
if __name__ == "__main__":
    print("\nRunning Advanced RAG...")
    advanced_rag_demo()
