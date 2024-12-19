# Retrieval-Augmented Generation

## Implementation of the Retrieval process 

The retrieval mechanism is an important part of the Retrieval-Augmented Generation (RAG) 
system, and it is responsible for finding the most relevant information from the dataset to 
answer a user's question. Here’s how It is implemented: 

1. Dataset Preparation: 
o The dataset (data_scientist_salaries.csv) was loaded using pandas, and only 
the columns that were necessary for the task were kept.

o Any rows with missing values were removed to ensure the data was clean 
and reliable. 

2. Query Embedding:
   
o When the user asks a question, the system first converts the query into a 
vector (embedding) using the Gemini text-embedding-004 model. This embedding captures the meaning of the query. 

3. Dataset Embedding:
   
o The dataset is processed in small batches for efficiency. Each row of the 
dataset is converted into a string, and an embedding is generated for each of 
these rows.

o A small delay (time.sleep(1)) is added between batches to avoid overloading 
the Gemini API. 

4. Cosine Similarity Calculation:
    
o The cosine similarity metric is used to measure how similar the query 
embedding is to each row’s embedding in the dataset. It calculates how 
close or far apart two vectors are, based on their direction. 

o After calculating the similarity, the system sorts the rows based on their 
similarity to the query and selects the top-k most relevant rows. 

5. Data Retrieval:
    
o The rows that are most similar to the query (top-k rows) are retrieved from the 
dataset. 

o These rows are then used as context to answer the query using the 
Generative AI model.

## Prompts Used for the LLM and Integration of Retrieved Data 

1. Prompt:
   
o The prompt given to the Generative AI model combines both the user’s query 
and the relevant data retrieved from the dataset. This allows the model to 
answer the question based on the provided context. 

o The structure of the prompt is as follows: 
Using the following data, answer the question: {query} {retrieved_data_context} 

Above: 
o {query} is the user's question. 

o {retrieved_data_context} is the text that contains the data the model will use to 
answer the query. 

2. Integration of Retrieved Data:
   
o The retrieved data is turned into a string where each row's information is 
formatted into readable text. This text is then included in the prompt, giving the 
model the necessary context to generate an answer. 

o For example, if the query is "What is the maximum salary?" and the retrieved 
rows include salary data, the model will use this data to come up with the 
correct answer. 

3. LLM Integration:
   
o The system sends the prompt to the Gemini API using the generate_content 
method. 

o The model processes the query and the data, then generates a response that 
answers the question based on the given context.

o The generated response is then extracted from the API’s output and shown to the 
user. Error handling ensures that the system can still function properly if the API 
responds unexpectedly.

![image](https://github.com/user-attachments/assets/69489524-e3ac-498f-bc61-3f27b4316f1b)

