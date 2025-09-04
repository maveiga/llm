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

# Modelos para o RAG
class QuestionRequest(BaseModel):
    question: str
    max_documents: int = 5
    category_filter: Optional[str] = None

class Source(BaseModel):
    id: int
    title: str
    category: str
    similarity_score: Optional[float] = None

class SearchResult(BaseModel):
    title: str
    category: str
    similarity_score: Optional[float] = None
    content_preview: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Source]
    context_used: int
    question: str
    has_context: bool
    search_results: Optional[List[SearchResult]] = None
    error: Optional[str] = None