from fastapi import APIRouter
from app.models.document import SearchRequest, SearchResponse
from app.controllers.document_controller import DocumentController

router = APIRouter()
controller = DocumentController()

@router.post("/search", response_model=SearchResponse)
async def search_documents(search_request: SearchRequest):
    return await controller.search_documents(search_request)