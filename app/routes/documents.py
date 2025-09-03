from fastapi import APIRouter, HTTPException
from app.models.document import Document, DocumentResponse
from app.controllers.document_controller import DocumentController

router = APIRouter()
controller = DocumentController()

@router.post("/documents", response_model=dict)
async def add_document(document: Document):
    return await controller.add_document(document)

@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    return await controller.get_document_by_id(doc_id)