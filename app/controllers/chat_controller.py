# CHAT CONTROLLER - Lógica de Negócio para Pipeline RAG
# Controller orquestra todo o fluxo RAG: busca → geração → persistência → observabilidade

from typing import Dict, Any, Optional
from app.services.rag_service import RAGService
from app.models.document import QuestionRequest
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatController:
    """
    Controller para operações de chat/RAG do sistema
    
    RESPONSABILIDADES:
    - Orquestração do pipeline RAG completo
    - Validações de negócio para perguntas
    - Lógica de health check do sistema
    - Métricas e logging de interações
    - Tratamento de casos especiais (sem contexto, erros)
    
    PADRÃO MVC:
    Route → Controller → RAGService → Services → Response
    """
    
    def __init__(self):
        self.rag_service = RAGService()
    
    async def process_question(
        self, 
        question_request: QuestionRequest,
        save_interaction: bool = True
    ) -> Dict[str, Any]:
        """
        LÓGICA DE NEGÓCIO: Processa pergunta completa via pipeline RAG
        
        PIPELINE ORQUESTRADO:
        1. Validações de negócio da pergunta
        2. Execução do pipeline RAG via service
        3. Enriquecimento da resposta com métricas
        4. Logging e auditoria
        5. Tratamento de casos especiais
        
        Args:
            question_request: Objeto com pergunta e parâmetros
            save_interaction: Se deve salvar para avaliação RAGAS
            
        """
        session_start = datetime.now()
        logger.info(f"Iniciando processamento RAG: '{question_request.question[:50]}...'")
        
        try:
            self._validate_question_request(question_request)
            
            rag_response = await self.rag_service.ask_question(
                question=question_request.question,
                max_documents=question_request.max_documents,
                category_filter=question_request.category_filter,
                save_interaction=save_interaction
            )
            
            return rag_response
            
        except ChatBusinessException as e:
            logger.error(f"Erro de negócio RAG: {e.message}")
            return self._create_error_response(e.message, "BUSINESS_ERROR", session_start)
            
        except Exception as e:
            logger.error(f"Erro técnico RAG: {str(e)}")
            return self._create_error_response(
                "Erro interno durante processamento da pergunta", 
                "TECHNICAL_ERROR", 
                session_start,
                technical_details=str(e)
            )
    
    def _validate_question_request(self, request: QuestionRequest) -> None:
        """Validações de negócio para requests de pergunta"""
    
        if not request.question or not request.question.strip():
            raise ChatBusinessException(
                "Pergunta não pode estar vazia",
                error_code="EMPTY_QUESTION"
            )
        
        if len(request.question.strip()) < 3:
            raise ChatBusinessException(
                "Pergunta muito curta - mínimo 3 caracteres",
                error_code="QUESTION_TOO_SHORT"
            )
        
        if len(request.question) > 2000:
            raise ChatBusinessException(
                "Pergunta muito longa - máximo 2000 caracteres",
                error_code="QUESTION_TOO_LONG"
            )
        
        if request.max_documents < 1 or request.max_documents > 20:
            raise ChatBusinessException(
                "max_documents deve estar entre 1 e 20",
                error_code="INVALID_MAX_DOCUMENTS"
            )

    
    def _get_no_context_recommendation(self) -> str:
        """Recomendação de negócio quando não há contexto relevante"""
        return (
            "Nenhum documento relevante foi encontrado. "
            "Tente reformular sua pergunta ou verifique se os documentos "
            "foram carregados corretamente via /admin/load-documents"
        )
    
    def _create_error_response(
        self, 
        message: str, 
        error_type: str, 
        session_start: datetime,
        technical_details: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cria resposta padronizada para erros"""
        
        session_duration = (datetime.now() - session_start).total_seconds()
        
        return {
            "answer": f"Desculpe, ocorreu um erro: {message}",
            "sources": [],
            "context_used": 0,
            "has_context": False,
            "business_status": "error",
            "error_details": {
                "error_type": error_type,
                "message": message,
                "technical_details": technical_details,
                "session_duration_seconds": round(session_duration, 3),
                "timestamp": session_start.isoformat()
            }
        }
    
class ChatBusinessException(Exception):
    """Exceção para erros de lógica de negócio em operações de chat/RAG"""
    
    def __init__(self, message: str, error_code: str = "CHAT_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)