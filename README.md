# LLM-Powered Booking Analytics & QA System

## Overview
This project implements an LLM-powered booking analytics and retrieval-augmented question-answering (RAG) system using **FastAPI, FAISS, and Mistral-7B**.

## Features
- **Data Processing & Storage**: Cleans and structures hotel booking data.
- **Analytics API**: Provides insights like revenue trends, cancellations, and booking distribution.
- **Retrieval-Augmented QA**: Uses FAISS for vector storage and GPT-Neo 1.3B for answering user queries.
- **Performance Evaluation**: Measures response time and evaluates answer accuracy.

## Setup Instructions
### 1. Install Dependencies
```bash
pip install fastapi uvicorn pandas numpy faiss-cpu sentence-transformers transformers torch
```

### 2. Run the FastAPI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. API Endpoints
#### a) Analytics Endpoint
```bash
POST /analytics
```
- Returns revenue trends, cancellation rate, and top booking locations.

#### b) Q&A Endpoint
```bash
POST /ask
```
**Example Payload:**
```json
{
  "question": "What is the average price of a hotel booking?"
}
```

#### c) Model Evaluation
```bash
POST /evaluate
```
- Compares generated answers with expected values.


## Future Improvements
- Enhance LLM with fine-tuned models.
- Optimize FAISS indexing for larger datasets.
- Deploy using Docker & cloud services.

---


