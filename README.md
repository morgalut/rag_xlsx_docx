# RAG Backend

A production-ready Retrieval-Augmented Generation (RAG) system with support for document ingestion, hybrid search, and intelligent query processing.

## Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose (for containerized deployment)
- MongoDB (local or containerized)

## Installation

Install the required dependencies:

```sh
pip install -r requirements.txt
```

## Configuration

The application uses environment variables for configuration. Update the `.env` file based on your deployment method:

### Local MongoDB

```sh
MONGO_URI=mongodb://localhost:27017
```

### Docker MongoDB (Default)

```sh
MONGO_URI=mongodb://mongo:27017
```

## Deployment

### Docker Deployment

Build and start the services using Docker Compose:

```sh
docker compose up -d --build
```

### Local Development

Navigate to the backend directory and start the server:

```sh
cd rag_backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage

### Health Check

Verify the service is running:

```sh
curl http://localhost:8000/health
```

### Document Ingestion

Ingest XLSX or DOCX files from the `data/docs/` directory. Update the path in the command below to match your environment:

```sh
cd /home/mor/Desktop/bank_poalem/rag_backend

curl -X POST http://localhost:8000/api/ingest/ \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"employees_handbook","path":"app/data/docs"}'
```

### Query Examples

#### Retriever Fusion Strategy

```sh
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Based on the RAG System Design & Evaluation document and the consolidated model performance data, summarize the recommended retriever fusion strategy and how it combines lexical and semantic search using Reciprocal Rank Fusion (RRF). Also explain how knowledge graphs are integrated into the retrieval process for hybrid search."
  }'
```

#### Memory-Augmented Generation

```sh
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize how memory-augmented generation is implemented in the RAG architecture. Explain the difference between session memory and long-term memory, and how embeddings with recency decay help prevent redundant or stale responses."
  }'
```

#### Latency Optimization

```sh
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Using information from the RAG design document, outline the main latency optimization techniques such as async fan-out retrieval, caching, vector index pre-warming, and token budget enforcement. How do these techniques impact overall throughput and user response time?"
  }'
```

#### Model Performance Comparison

```sh
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Based on the consolidated model performance data in rag_test_data_consolidated.xlsx, compare the latency and accuracy of the listed foundation models (OpenAI GPT-4 Turbo, Claude 3.5 Sonnet, Gemini 1.5 Pro). Identify which model offers the best tradeoff between accuracy and speed for retrieval-augmented generation workloads."
  }'
```

#### Evaluation Framework

```sh
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "From the RAG evaluation guidelines, detail the recommended evaluation framework including synthetic Q/A pairs, LLM judges, and human audits. Describe how metrics such as Context Recall, Faithfulness, Answer Quality, MRR@10, and nDCG@10 are used to assess model performance."
  }'
```
