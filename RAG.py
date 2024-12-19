#importing neccessary libraries
import google.generativeai as genai  
import pandas as pd  
import os  
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np  
import time  

# Set up Gemini API key by retrieving it from environment variables
genai.configure(api_key=os.environ["API_KEY"])

# Load the dataset (replace with your actual CSV file)
df = pd.read_csv("data_scientist_salaries.csv")

# Select only the relevant columns based on the assignment instructions
relevant_columns = [
    'Hobby', 'OpenSource', 'Country', 'Student', 'Employment', 
    'FormalEducation', 'UndergradMajor', 'CompanySize', 'DevType', 
    'YearsCoding', 'Salary', 'SalaryType', 'ConvertedSalary'
]

# Filter dataset to only include the relevant columns
df = df[relevant_columns]

# Handle missing data (drop rows with missing values in these columns)
df = df.dropna()

#print(df.head())  # to check the first few rows of the dataset
#print(df.columns)  # to check the columns in the dataset

# Function to get embeddings for a given text (for retrieval)
def get_embedding(text):
    # Send the text to Gemini API to obtain its embedding
    result = genai.embed_content(
        model="models/text-embedding-004",  # Embedding model
        content=text,
        task_type="retrieval_document",  # Task type for document retrieval
        title="Embedding of single string"  # Title for the embedding task
    )
    # Return the embedding from the API response
    return result['embedding']

# Function to retrieve relevant rows based on cosine similarity
def retrieve_relevant_data(query, top_k=3, batch_size=5):
    #print("Generating embedding for the query...")  
    # Get the embedding of the query using the get_embedding function
    query_embedding = get_embedding(query)
    #print("Query embedding generated.")  

    # Create batches for dataset embeddings
    dataset_embeddings = []
    for i in range(0, len(df), batch_size):
        # Select a batch of rows from the dataset
        batch = df.iloc[i:i + batch_size]
        # Get embeddings for each row in the batch and add to the list
        batch_embeddings = [get_embedding(" ".join(map(str, row))) for row in batch.values]
        dataset_embeddings.extend(batch_embeddings)  # Extend the list with batch embeddings
        time.sleep(1)  # Add a delay to avoid overwhelming the API

    #print("Dataset embeddings generated.") 

    # Compute cosine similarity between query embedding and dataset embeddings
    #print("Computing cosine similarities...")
    similarities = cosine_similarity([query_embedding], dataset_embeddings)
    
    #print(f"Similarities: {similarities[0]}")  
    #print("Done with similarities.") 
    
    # Get indices of the most similar rows by sorting similarities in descending order
    top_indices = np.argsort(similarities[0])[::-1][:top_k]
    #print(f"Top indices: {top_indices}")
    
    # Retrieve the top_k most similar rows from the dataset
    return df.iloc[top_indices]

# Function to generate a response using Gemini
def generate_response(retrieved_data, query):
    # Convert the retrieved rows to text (to be used as context for the prompt)
    context = "\n".join([str(row) for row in retrieved_data.values])
    
    #print("Generating response using Gemini...")
    # Call Gemini API to generate content based on the context and query
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
        f"Using the following data, answer the question: {query}\n\n{context}"
    )
    #print("Print response.")  
    #print(response)
    
    # Extract and return the generated response
    try:
        # Access the nested content within the response object to get the text
        content = response.candidates[0].content.parts[0].text
        return content
    except (KeyError, IndexError, AttributeError) as e:
        # Handle cases where the structure is unexpected
        print(f"Error extracting response: {e}")
        return "Error: Unable to generate response."


# Start a loop to keep asking for queries continuously
while True:
    # Step 1: Ask the user for their query
    query = input("Please enter your query (or type 'exit' to quit): ")
    
    # If the user types 'exit', break the loop
    if query.lower() == 'exit':
        print("Exiting the program.")
        break
    
    # Step 2: Retrieve relevant data based on the user's query
    retrieved_data = retrieve_relevant_data(query)

    # Step 3: Generate response using the retrieved data
    response = generate_response(retrieved_data, query)

    # Output the result
    print("\nResponse:")
    print(response)