# How run 
```sh 
pip install -r requirements.txt
```


## If you want run loacal change path in .env
```sh
MONGO_URI=mongodb://localhost:27017
```

## If you want run docker change path in .env
*(This is default)*
```sh
MONGO_URI=mongodb://mongodb:27017
```

# Build docker 
```sh
docker compose up -d --build
```



# Run server 
```sh
cd rag_backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

# Check health
```sh
curl http://localhost:8000/health
```


# Ingest an xlsx or docx you placed in data/docs/
```sh
cd /home/mor/Desktop/bank_poalem/rag_backend
```
```sh
curl -X POST http://localhost:8000/api/ingest/ \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"employees_handbook","path":"app/data/docs"}'

```


# Ask a question
```sh
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Based on the RAG System Design & Evaluation document and the consolidated model performance data, summarize the recommended retriever fusion strategy and how it combines lexical and semantic search using Reciprocal Rank Fusion (RRF). Also explain how knowledge graphs are integrated into the retrieval process for hybrid search."
  }'


```

----
```sh
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize how memory-augmented generation is implemented in the RAG architecture. Explain the difference between session memory and long-term memory, and how embeddings with recency decay help prevent redundant or stale responses."
  }'
```
----
```sh
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Using information from the RAG design document, outline the main latency optimization techniques such as async fan-out retrieval, caching, vector index pre-warming, and token budget enforcement. How do these techniques impact overall throughput and user response time?"
  }'

```
----
```sh
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Based on the consolidated model performance data in rag_test_data_consolidated.xlsx, compare the latency and accuracy of the listed foundation models (OpenAI GPT-4 Turbo, Claude 3.5 Sonnet, Gemini 1.5 Pro). Identify which model offers the best tradeoff between accuracy and speed for retrieval-augmented generation workloads."
  }'

```
---
```sh
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "From the RAG evaluation guidelines, detail the recommended evaluation framework including synthetic Q/A pairs, LLM judges, and human audits. Describe how metrics such as Context Recall, Faithfulness, Answer Quality, MRR@10, and nDCG@10 are used to assess model performance."
  }'
```