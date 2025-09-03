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
        chunked_docs = processor.chunk_documents(documents)
        for chunk in chunked_docs:
            await vector_service.add_document(chunk)
        
        return {
            "message": f"Successfully loaded {chunked_docs} document chunks from {len(documents)} files",
            "total_files": len(documents),
            "total_chunks": chunked_docs
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