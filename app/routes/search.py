# SEARCH ROUTES - Interface HTTP para Busca Vetorial
# Route é responsável apenas por HTTP: validação, serialização, tratamento de erros
# Toda lógica de busca e análise fica no SearchController

from fastapi import APIRouter, HTTPException
from app.models.document import SearchRequest, SearchResponse, DocumentResponse
from app.controllers.search_controller import SearchController, SearchBusinessException

router = APIRouter()
search_controller = SearchController()

@router.post("/search", response_model=SearchResponse)
async def search_documents(search_request: SearchRequest) -> SearchResponse:
    """
    ENDPOINT DE BUSCA: Executa busca vetorial direta com análise de qualidade
    
    RESPONSABILIDADES DA ROUTE:
    - Validação de parâmetros HTTP
    - Chamada para controller (lógica de busca)
    - Serialização da resposta
    - Tratamento de exceções HTTP
    
    Args:
        search_request: Parâmetros da busca vetorial
    
    Returns:
        SearchResponse enriquecida com métricas e análise de qualidade
        
    Raises:
        HTTPException: Para erros HTTP (400, 500)
    """
    try:
        # CHAMA CONTROLLER: toda lógica de busca e análise está lá
        search_results = await search_controller.execute_search(search_request)
        
        # VERIFICAR SE HOUVE ERRO DE NEGÓCIO
        if search_results.get("search_status") == "error":
            error_details = search_results.get("error_details", {})
            error_code = error_details.get("error_code", "UNKNOWN")
            
            if error_code in ["EMPTY_QUERY", "QUERY_TOO_SHORT", "QUERY_TOO_LONG"]:
                raise HTTPException(status_code=400, detail=error_details.get("message"))
            elif error_code == "INVALID_LIMIT":
                raise HTTPException(status_code=400, detail=error_details.get("message"))
            else:
                raise HTTPException(status_code=500, detail=error_details.get("message"))
        
        # SUCESSO: converter para modelo de resposta
        # Converter dicionários para objetos DocumentResponse
        raw_results = search_results.get("results", [])
        document_responses = []
        
        for result_dict in raw_results:
            document_responses.append(DocumentResponse(
                id=result_dict.get("id", ""),
                title=result_dict.get("title", ""),
                category=result_dict.get("category", ""),
                content=result_dict.get("content", ""),
                metadata=result_dict.get("metadata", {}),
                similarity_score=result_dict.get("similarity_score")
            ))
        
        # Criar SearchResponse com todos os campos extras
        response = SearchResponse(
            query=search_results.get("query", ""),
            results=document_responses,
            total_results=search_results.get("total_results", 0),
            search_metadata=search_results.get("search_metadata"),
            quality_analysis=search_results.get("quality_analysis"),
            business_insights=search_results.get("business_insights"),
            search_status=search_results.get("search_status", "success")
        )
        
        return response
        
    except SearchBusinessException as e:
        # Exceções de negócio específicas (400 Bad Request)
        if e.error_code in ["EMPTY_QUERY", "QUERY_TOO_SHORT", "QUERY_TOO_LONG"]:
            raise HTTPException(status_code=400, detail=e.message)
        elif e.error_code in ["INVALID_LIMIT", "EMPTY_CATEGORY_FILTER"]:
            raise HTTPException(status_code=400, detail=e.message)
        else:
            raise HTTPException(status_code=400, detail=e.message)
            
    except Exception as e:
        # Erros técnicos inesperados (500 Internal Server Error)
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno na busca vetorial: {str(e)}"
        )