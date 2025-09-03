import chromadb
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.models.document import Document, DocumentResponse

class VectorService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name
        )
        self.embedding_model = SentenceTransformer(settings.embedding_model)
    
    async def add_document(self, document: Document) -> str:
        embedding = self.embedding_model.encode([document.content])[0].tolist()
        
        doc_id = f"{document.category}_{hash(document.title + document.content)}"
        
        self.collection.add(
            documents=[document.content],
            embeddings=[embedding],
            metadatas=[{
                "title": document.title,
                "category": document.category,
                **document.metadata
            }],
            ids=[doc_id]
        )
        
        return doc_id
    
    async def search_documents(
        self, 
        query: str, 
        limit: int = 5,
        category_filter: Optional[str] = None
    ) -> List[DocumentResponse]:
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter
        )
        
        documents = []
        for i, doc_id in enumerate(results['ids'][0]):
            documents.append(DocumentResponse(
                id=doc_id,
                title=results['metadatas'][0][i]['title'],
                category=results['metadatas'][0][i]['category'],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                similarity_score=1 - results['distances'][0][i] if 'distances' in results else None
            ))
        
        return documents
    
    async def get_document_by_id(self, doc_id: str) -> Optional[DocumentResponse]:
        results = self.collection.get(ids=[doc_id])
        
        if not results['ids']:
            return None
        
        return DocumentResponse(
            id=results['ids'][0],
            title=results['metadatas'][0]['title'],
            category=results['metadatas'][0]['category'],
            content=results['documents'][0],
            metadata=results['metadatas'][0]
        )