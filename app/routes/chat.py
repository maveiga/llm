from fastapi import APIRouter, HTTPException
from app.models.document import QuestionRequest, QuestionResponse
from app.services.rag_service import RAGService

router = APIRouter()
rag_service = RAGService()

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint RAG: recebe uma pergunta e retorna resposta com citações
    
    - **question**: Pergunta do usuário
    - **max_documents**: Número máximo de documentos para usar como contexto (default: 5)
    - **category_filter**: Filtro opcional por categoria
    
    Retorna:
    - **answer**: Resposta gerada pelo LLM
    - **sources**: Lista de fontes citadas
    - **search_results**: Documentos encontrados na busca vetorial
    """
    try:
        response = await rag_service.ask_question(
            question=request.question,
            max_documents=request.max_documents,
            category_filter=request.category_filter
        )
        
        return QuestionResponse(**response)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro no pipeline RAG: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Verifica se o serviço RAG está funcionando"""
    try:
        health = await rag_service.health_check()
        return {
            "status": "healthy" if all(health.values()) else "degraded",
            "services": health
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Serviço indisponível: {str(e)}"
        )