from typing import Dict, Any, List, Optional
from app.services.ragas_service import ragas_service
from app.services.database_service import AsyncSessionLocal
from app.services.phoenix_service import phoenix_service
from app.models.rag_interaction import RAGInteractionDB, RAGASEvaluation, UserFeedback
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class EvaluationController:
    """
    Controller para operações de avaliação de qualidade RAG
    
    RESPONSABILIDADES:
    - Orquestração de avaliações RAGAS
    - Lógica de negócio para relatórios de qualidade
    - Gestão de feedback de usuários
    - Integração Phoenix + RAGAS
    - Validações de dados de avaliação
    
    """
    
    def __init__(self):
        pass
    
    async def execute_ragas_evaluation(
        self, 
        evaluation_request: RAGASEvaluation,
        background_execution: bool = False
    ) -> Dict[str, Any]:
        """
        LÓGICA DE NEGÓCIO: Executa avaliação RAGAS com validações e métricas
        
        PROCESSO DE NEGÓCIO:
        1. Validar parâmetros de avaliação
        2. Buscar interações para avaliar (se não especificadas)
        3. Executar avaliação RAGAS
        
        Args:
            evaluation_request: Parâmetros da avaliação
            background_execution: Se deve executar em background
            
        Returns:
            Dict com resultados detalhados da avaliação
        """
        evaluation_start = datetime.now()
        
        try:
            await self._validate_evaluation_request(evaluation_request)
        
            interaction_ids = await self._resolve_interaction_ids(evaluation_request)
            
            if not interaction_ids:
                return self._create_evaluation_error_response(
                    "Nenhuma interação encontrada para avaliar",
                    "NO_INTERACTIONS",
                    evaluation_start
                )
            
            ragas_results = await ragas_service.evaluate_interactions(interaction_ids)
            
            if "error" in ragas_results:
                return self._create_evaluation_error_response(
                    ragas_results["error"],
                    "RAGAS_EXECUTION_ERROR",
                    evaluation_start
                )

            return ragas_results
            
        except EvaluationBusinessException as e:
            logger.error(f"Erro de negócio RAGAS: {e.message}")
            return self._create_evaluation_error_response(e.message, e.error_code, evaluation_start)
            
        except Exception as e:
            logger.error(f"Erro técnico RAGAS: {str(e)}")
            return self._create_evaluation_error_response(
                "Erro interno durante avaliação",
                "TECHNICAL_ERROR", 
                evaluation_start,
                technical_details=str(e)
            )
    
    async def _validate_evaluation_request(self, request: RAGASEvaluation) -> None:
        """Validações de negócio para requests de avaliação"""

        if request.interaction_ids:
            if len(request.interaction_ids) > 100:
                raise EvaluationBusinessException(
                    "Máximo 100 interações por avaliação",
                    error_code="TOO_MANY_INTERACTIONS"
                )
            
            invalid_ids = [id for id in request.interaction_ids if not id or len(id) < 10]
            if invalid_ids:
                raise EvaluationBusinessException(
                    f"IDs inválidos encontrados: {invalid_ids[:5]}",
                    error_code="INVALID_INTERACTION_IDS"
                )
        
        logger.info(f"Request de avaliação validado")
    
    async def _resolve_interaction_ids(self, request: RAGASEvaluation) -> List[str]:
        """Resolve IDs de interações para avaliar"""
        
        if request.interaction_ids:
            return request.interaction_ids
        
        async with AsyncSessionLocal() as session:
            query = select(RAGInteractionDB).order_by(
                desc(RAGInteractionDB.timestamp)
            ).limit(50)
            
            result = await session.execute(query)
            interactions = result.scalars().all()
            
            interaction_ids = [i.id for i in interactions]
            logger.info(f"Resolvidos {len(interaction_ids)} IDs de interações recentes")
            
            return interaction_ids

    
    def _create_evaluation_error_response(
        self, 
        message: str, 
        error_code: str, 
        evaluation_start: datetime,
        technical_details: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cria resposta padronizada para erros de avaliação"""
        
        duration = (datetime.now() - evaluation_start).total_seconds()
        
        return {
            "evaluation_status": "error",
            "error_details": {
                "message": message,
                "error_code": error_code,
                "technical_details": technical_details,
                "evaluation_duration_seconds": round(duration, 3),
                "timestamp": evaluation_start.isoformat()
            },
            "total_interactions": 0,
            "average_scores": {},
            "individual_scores": []
        }
    
    async def get_advanced_metrics(
        self,
        limit: int = 50,
        include_individual_scores: bool = False
    ) -> Dict[str, Any]:
        """
        LÓGICA DE NEGÓCIO: Calcula métricas avançadas RAG
        
        Responsável por:
        - Buscar interações para análise
        - Calcular métricas avançadas via RAGAS service
        - Interpretar resultados
        - Fornecer recomendações de negócio
        """
        try:
            async with AsyncSessionLocal() as session:
                query = select(RAGInteractionDB).order_by(
                    desc(RAGInteractionDB.timestamp)
                ).limit(limit)
                
                result = await session.execute(query)
                interactions = result.scalars().all()
                
                if not interactions:
                    raise EvaluationBusinessException(
                        "Nenhuma interação encontrada para análise",
                        "NO_INTERACTIONS"
                    )
                
                individual_scores = []
                advanced_metrics = await ragas_service._calculate_advanced_metrics(
                    interactions, 
                    individual_scores
                )
                
                response = {
                    "metrics_overview": {
                        "total_interactions_analyzed": len(interactions),
                        "analysis_period": "Últimas interações",
                        "timestamp": datetime.now().isoformat()
                    },
                    "advanced_metrics": advanced_metrics,
                    "interpretation": self._interpret_advanced_metrics(advanced_metrics),
                }
                
                return response
                
        except EvaluationBusinessException:
            raise
        except Exception as e:
            raise EvaluationBusinessException(
                f"Erro ao calcular métricas avançadas: {str(e)}",
                "ADVANCED_METRICS_ERROR"
            )

    def _interpret_advanced_metrics(self, advanced_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Interpreta as métricas avançadas fornecendo insights de negócio"""
        interpretation = {}
        
        # Interpretação Recall@3
        recall_value = advanced_metrics["recall_at_3"]["value"]
        interpretation["recall_at_3"] = {
            "current_value": recall_value,
            "quality_level": self._interpret_recall_at_3(recall_value),
            "recommendation": self._get_recall_recommendation(recall_value)
        }
        
        # Interpretação Precisão Percebida
        precision_value = advanced_metrics["perceived_precision"]["value"]
        if precision_value is not None:
            interpretation["perceived_precision"] = {
                "current_value": precision_value,
                "quality_level": self._interpret_perceived_precision(precision_value),
                "user_satisfaction": advanced_metrics["perceived_precision"].get("feedback_distribution")
            }
        else:
            interpretation["perceived_precision"] = None
        
        return interpretation

    def _interpret_recall_at_3(self, value: float) -> str:
        """Interpreta o valor de Recall@3"""
        if value >= 0.8:
            return "Excelente - Sistema recupera documentos muito relevantes"
        elif value >= 0.6:
            return "Bom - Maioria dos documentos top-3 são relevantes"
        elif value >= 0.4:
            return "Regular - Necessário melhorar relevância dos resultados"
        else:
            return "Crítico - Poucos documentos relevantes nos top-3"

    def _interpret_perceived_precision(self, value: float) -> str:
        """Interpreta a Precisão Percebida pelos usuários"""
        if value is None:
            return "Sem dados de feedback"
        elif value >= 0.8:
            return "Excelente - Usuários muito satisfeitos"
        elif value >= 0.6:
            return "Bom - Usuários satisfeitos"
        elif value >= 0.4:
            return "Regular - Satisfação moderada"
        else:
            return "Crítico - Usuários insatisfeitos"


    def _get_recall_recommendation(self, value: float) -> str:
        """Recomendações para melhorar Recall@3"""
        if value < 0.6:
            return "Considere: ajustar parâmetros de busca, melhorar embeddings, revisar critérios de relevância"
        else:
            return "Sistema performando bem na recuperação de documentos relevantes"


# Exceção específica para erros de negócio de avaliação
class EvaluationBusinessException(Exception):
    """Exceção para erros de lógica de negócio em operações de avaliação"""
    
    def __init__(self, message: str, error_code: str = "EVALUATION_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

    