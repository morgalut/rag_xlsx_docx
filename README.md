# How run 
```sh 
pip install -r requirements.txt
```


# If you want run loacal change path in .env
```sh
MONGO_URI=mongodb://localhost:27017
```

# If you want run loacal change path in .env
```sh
OLLAMA_HOST=http://localhost:11434
```
# If you want run docker change path in .env
```sh
OLLAMA_HOST=http://ollama:11434
```



# Build docker 
```sh
docker compose up -d --build
```

# Pull model from ollama 
```sh
docker exec -it rag_backend-ollama-1 ollama pull llama3.2
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