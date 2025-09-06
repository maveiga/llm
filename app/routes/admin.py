# ADMIN ROUTES - Interface HTTP para Operações Administrativas
# Route é responsável apenas por HTTP: validação, autenticação, serialização
# Toda lógica de negócio fica no AdminController

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.controllers.admin_controller import AdminController, AdminBusinessException

router = APIRouter()
admin_controller = AdminController()

@router.post("/admin/load-documents")
async def load_documents_from_directory(
    directory_path: str = Query(
        default="conteudo_ficticio", 
        description="Caminho para diretório contendo arquivos .txt para indexar"
    )
) -> dict:
    """
    ENDPOINT ADMINISTRATIVO: Carrega documentos de diretório para o sistema RAG
    
    RESPONSABILIDADES DA ROUTE:
    - Validação de parâmetros HTTP
    - Chamada para controller (lógica de negócio)
    - Tratamento de exceções HTTP
    - Serialização da resposta
    
    Args:
        directory_path: Caminho para diretório com arquivos .txt
        
    Returns:
        Dict com resultado do carregamento e métricas
        
    Raises:
        HTTPException: Para erros HTTP (400, 404, 500)
    """
    try:
        # CHAMA CONTROLLER: toda lógica de negócio está lá
        result = await admin_controller.load_documents_from_directory(directory_path)
        
        # CONTROLLER RETORNOU ERRO DE NEGÓCIO
        if not result.get("success", True):
            # Se é erro de negócio conhecido, retorna 400 (Bad Request)
            if "DIRECTORY_NOT_FOUND" in result.get("error", ""):
                raise HTTPException(status_code=404, detail=result["message"])
            elif "NO_TXT_FILES" in result.get("error", ""):
                raise HTTPException(status_code=400, detail=result["message"])
            else:
                # Erro genérico de processamento
                raise HTTPException(status_code=500, detail=result["message"])
        
        # SUCESSO: retorna resultado do controller
        return result
        
    except AdminBusinessException as e:
        # Exceções de negócio específicas (400 Bad Request)
        if e.error_code == "DIRECTORY_NOT_FOUND":
            raise HTTPException(status_code=404, detail=e.message)
        elif e.error_code == "NO_TXT_FILES":
            raise HTTPException(status_code=400, detail=e.message)
        else:
            raise HTTPException(status_code=400, detail=e.message)
            
    except Exception as e:
        # Erros técnicos inesperados (500 Internal Server Error)
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno durante carregamento: {str(e)}"
        )


