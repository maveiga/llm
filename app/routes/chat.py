# CHAT ROUTES - Interface HTTP para Pipeline RAG
# Route é responsável apenas por HTTP: validação, serialização, tratamento de erros
# Toda lógica de negócio RAG fica no ChatController

from fastapi import APIRouter, HTTPException
from app.models.document import QuestionRequest, QuestionResponse
from app.controllers.chat_controller import ChatController, ChatBusinessException

router = APIRouter()
chat_controller = ChatController()

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    """
    ENDPOINT RAG: Processa pergunta via pipeline RAG completo
    
    RESPONSABILIDADES DA ROUTE:
    - Validação de entrada HTTP
    - Chamada para controller (lógica RAG)
    - Serialização da resposta
    - Tratamento de exceções HTTP
    
    Args:
        request: Pergunta do usuário com parâmetros de busca
    
    Returns:
        QuestionResponse com resposta gerada, fontes e métricas
        
    Raises:
        HTTPException: Para erros HTTP (400, 500)
    """
    try:
        response = await chat_controller.process_question(request)
    
        if response.get("business_status") == "error":
            error_details = response.get("error_details", {})
            error_type = error_details.get("error_type", "UNKNOWN")
            
            if error_type == "BUSINESS_ERROR":
                raise HTTPException(status_code=400, detail=error_details.get("message"))
            else:
                raise HTTPException(status_code=500, detail=error_details.get("message"))
        return QuestionResponse(**response)
        
    except ChatBusinessException as e:
        if e.error_code in ["EMPTY_QUESTION", "QUESTION_TOO_SHORT", "QUESTION_TOO_LONG"]:
            raise HTTPException(status_code=400, detail=e.message)
        elif e.error_code == "INVALID_MAX_DOCUMENTS":
            raise HTTPException(status_code=400, detail=e.message)
        else:
            raise HTTPException(status_code=400, detail=e.message)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno no pipeline RAG: {str(e)}"
        )

