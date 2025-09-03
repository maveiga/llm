from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class Document(BaseModel):
    id: Optional[str] = None
    title: str
    category: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}

class DocumentResponse(BaseModel):
    id: str
    title: str
    category: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    category_filter: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[DocumentResponse]
    total_results: int