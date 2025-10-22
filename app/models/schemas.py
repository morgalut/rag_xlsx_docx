from pydantic import BaseModel
from typing import Optional

class IngestRequest(BaseModel):
    doc_id: str
    path: str
    metadata: Optional[dict] = None

class QueryRequest(BaseModel):
    query: str
