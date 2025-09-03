from typing import List
from fastapi import HTTPException
from app.models.document import Document, DocumentResponse, SearchRequest, SearchResponse
from app.services.vector_service import VectorService

class DocumentController:
    def __init__(self):
        self.vector_service = VectorService()
    
    async def add_document(self, document: Document) -> dict:
        try:
            doc_id = await self.vector_service.add_document(document)
            return {"id": doc_id, "message": "Document added successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")
    
    async def search_documents(self, search_request: SearchRequest) -> SearchResponse:
        try:
            results = await self.vector_service.search_documents(
                query=search_request.query,
                limit=search_request.limit,
                category_filter=search_request.category_filter
            )
            
            return SearchResponse(
                query=search_request.query,
                results=results,
                total_results=len(results)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")
    
    async def get_document_by_id(self, doc_id: str) -> DocumentResponse:
        try:
            document = await self.vector_service.get_document_by_id(doc_id)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            return document
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")