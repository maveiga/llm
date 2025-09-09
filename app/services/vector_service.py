import chromadb
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.core.config import settings
from app.models.document import Document, DocumentResponse

class VectorService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name
        )
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.cross_encoder = CrossEncoder(settings.cross_encoder_model)
    
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
        if category_filter and category_filter != "string":
            where_filter = {"category": category_filter}
        
        initial_limit = min(limit * 3, 20)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_limit,
            where=where_filter
        )
        
        documents = []
        if results['ids'] and results['ids'][0]:
            query_doc_pairs = []
            temp_documents = []
            
            for i, doc_id in enumerate(results['ids'][0]):
                content = results['documents'][0][i]
                query_doc_pairs.append([query, content])
                temp_documents.append(DocumentResponse(
                    id=doc_id,
                    title=results['metadatas'][0][i]['title'],
                    category=results['metadatas'][0][i]['category'],
                    content=content,
                    metadata=results['metadatas'][0][i],
                    similarity_score=1 - results['distances'][0][i] if 'distances' in results else None
                ))
            
            if query_doc_pairs:
                cross_scores = self.cross_encoder.predict(query_doc_pairs)
                
                for i, score in enumerate(cross_scores):
                    temp_documents[i].similarity_score = float(score)
            
                temp_documents.sort(key=lambda x: x.similarity_score, reverse=True)
                documents = temp_documents[:limit]
        
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