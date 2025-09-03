from fastapi import APIRouter, HTTPException
from app.services.document_processor import DocumentProcessor
from app.services.vector_service import VectorService
from langchain.schema import Document as LangChainDocument
from app.models.document import Document

router = APIRouter()
processor = DocumentProcessor()
vector_service = VectorService()

@router.post("/admin/load-documents")
async def load_documents_from_directory(directory_path: str = "conteudo_ficticio"):
    try:
        documents = processor.load_documents_from_directory(directory_path)
        
        total_chunks = 0
        for document in documents:
            chunked_docs = processor.chunk_document(document)
            
            for chunk in chunked_docs:
                await vector_service.add_document(chunk)
                total_chunks += 1
        
        return {
            "message": f"Successfully loaded {total_chunks} document chunks from {len(documents)} files",
            "total_files": len(documents),
            "total_chunks": total_chunks
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading documents: {str(e)}")

@router.get("/admin/collection-info")
async def get_collection_info():
    try:
        collection = vector_service.collection
        count = collection.count()
        
        return {
            "collection_name": collection.name,
            "document_count": count
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}")