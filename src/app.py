import pandas as pd
import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

df = pd.read_csv("hotel_bookings.csv" )

# Initialize FastAPI
app = FastAPI()

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare text data for embeddings
text_data = df.apply(lambda row: f"Booking ID {row.name}: {row.to_dict()}", axis=1).tolist()

# Generate embeddings
embeddings = model.encode(text_data, convert_to_numpy=True)

# Ensure correct FAISS indexing
d = embeddings.shape[1]  # Use correct embedding dimension
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Store embeddings and metadata
metadata = {i: text_data[i] for i in range(len(text_data))}

# Load GPT-Neo model for QA
qa_pipeline = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-1.3B",  # Smaller model for lower VRAM
    torch_dtype=torch.float32,
    device=0 if torch.cuda.is_available() else -1
)

class QueryModel(BaseModel):
    question: str

@app.post("/analytics")
def get_analytics():
    """Returns key analytics from the dataset."""
    analytics = {
        "total_bookings": len(df),
        "cancellation_rate": df['is_canceled'].mean() * 100,
        "average_price": df['adr'].mean(),
        "top_booking_countries": df['country'].value_counts().head(3).to_dict()
    }
    return analytics

@app.post("/ask")
def retrieve_and_answer(query: QueryModel):
    """Retrieve relevant data and generate answers using LLM."""
    start_time = time.time()
    query_embedding = model.encode([query.question], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=3)  # Retrieve top 3 results
    retrieved_docs = "\n".join([metadata[idx] for idx in indices[0]])

    # Generate response using GPT-Neo
    prompt = f"Context: {retrieved_docs}\nQuestion: {query.question}\nAnswer:"
    response = qa_pipeline(
        prompt, 
        max_new_tokens=100,  # Ensures generated answer length is limited
        truncation=True,  # Explicitly enable truncation
        do_sample=True, 
        temperature=0.7, 
        top_p=0.9
    )[0]["generated_text"]

    response_time = time.time() - start_time
    
    return {
        "question": query.question,
        "answer": response,
        "response_time": f"{response_time:.2f} seconds"
    }

@app.post("/evaluate")
def evaluate_model():
    """Evaluate the model's accuracy on predefined questions."""
    test_questions = {
        "Show me total revenue for July 2017.": "Expected Revenue Value",
        "Which locations had the highest booking cancellations?": "Expected Locations",
        "What is the average price of a hotel booking?": "Expected Price Value"
    }
    
    results = {}
    for question, expected_answer in test_questions.items():
        answer = retrieve_and_answer(QueryModel(question=question))
        results[question] = {
            "generated_answer": answer["answer"],
            "expected_answer": expected_answer
        }
    
    return results

print("FastAPI server ready with GPT-Neo!")


'''if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)'''

    

