# EVALUATION CONTROLLER - Lógica de Negócio para Avaliação RAGAS
# Controller orquestra avaliações de qualidade, relatórios e integração Phoenix

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
    - Análise de tendências e recomendações
    - Validações de dados de avaliação
    
    PADRÃO MVC:
    Route → Controller → Services (RAGAS, Database, Phoenix) → Response
    """
    
    def __init__(self):
        # Controller não instancia services diretamente - usa os globais
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
        4. Calcular métricas de negócio adicionais
        5. Gerar recomendações baseadas nos resultados
        6. Logging e auditoria
        
        Args:
            evaluation_request: Parâmetros da avaliação
            background_execution: Se deve executar em background
            
        Returns:
            Dict com resultados detalhados da avaliação
        """
        evaluation_start = datetime.now()
        logger.info(f"🔬 Iniciando avaliação RAGAS: {len(evaluation_request.interaction_ids or [])} interações")
        
        try:
            # ETAPA 1: VALIDAÇÕES DE NEGÓCIO
            await self._validate_evaluation_request(evaluation_request)
            
            # ETAPA 2: RESOLVER IDs DE INTERAÇÕES (se não fornecidos)
            interaction_ids = await self._resolve_interaction_ids(evaluation_request)
            
            if not interaction_ids:
                return self._create_evaluation_error_response(
                    "Nenhuma interação encontrada para avaliar",
                    "NO_INTERACTIONS",
                    evaluation_start
                )
            
            # ETAPA 3: EXECUTAR AVALIAÇÃO RAGAS
            ragas_results = await ragas_service.evaluate_interactions(interaction_ids)
            
            if "error" in ragas_results:
                return self._create_evaluation_error_response(
                    ragas_results["error"],
                    "RAGAS_EXECUTION_ERROR",
                    evaluation_start
                )
            
            # ETAPA 4: ENRIQUECER COM MÉTRICAS DE NEGÓCIO
            enriched_results = await self._enrich_evaluation_results(
                ragas_results, 
                interaction_ids,
                evaluation_start
            )
            
            # ETAPA 5: GERAR RECOMENDAÇÕES DE NEGÓCIO
            recommendations = self._generate_quality_recommendations(enriched_results)
            enriched_results["business_recommendations"] = recommendations
            
            # ETAPA 6: LOGGING E AUDITORIA
            self._log_evaluation_metrics(enriched_results, evaluation_start)
            
            logger.info(f"✅ Avaliação RAGAS concluída: {len(interaction_ids)} interações avaliadas")
            return enriched_results
            
        except EvaluationBusinessException as e:
            logger.error(f"❌ Erro de negócio RAGAS: {e.message}")
            return self._create_evaluation_error_response(e.message, e.error_code, evaluation_start)
            
        except Exception as e:
            logger.error(f"❌ Erro técnico RAGAS: {str(e)}")
            return self._create_evaluation_error_response(
                "Erro interno durante avaliação",
                "TECHNICAL_ERROR", 
                evaluation_start,
                technical_details=str(e)
            )
    
    async def _validate_evaluation_request(self, request: RAGASEvaluation) -> None:
        """Validações de negócio para requests de avaliação"""
        
        # Validar lista de IDs se fornecida
        if request.interaction_ids:
            if len(request.interaction_ids) > 100:
                raise EvaluationBusinessException(
                    "Máximo 100 interações por avaliação",
                    error_code="TOO_MANY_INTERACTIONS"
                )
            
            # Validar formato dos IDs
            invalid_ids = [id for id in request.interaction_ids if not id or len(id) < 10]
            if invalid_ids:
                raise EvaluationBusinessException(
                    f"IDs inválidos encontrados: {invalid_ids[:5]}",
                    error_code="INVALID_INTERACTION_IDS"
                )
        
        logger.info(f"✅ Request de avaliação validado")
    
    async def _resolve_interaction_ids(self, request: RAGASEvaluation) -> List[str]:
        """Resolve IDs de interações para avaliar"""
        
        if request.interaction_ids:
            # IDs específicos fornecidos
            return request.interaction_ids
        
        # Buscar interações recentes para avaliar
        async with AsyncSessionLocal() as session:
            query = select(RAGInteractionDB).order_by(
                desc(RAGInteractionDB.timestamp)
            ).limit(50)
            
            result = await session.execute(query)
            interactions = result.scalars().all()
            
            interaction_ids = [i.id for i in interactions]
            logger.info(f"📋 Resolvidos {len(interaction_ids)} IDs de interações recentes")
            
            return interaction_ids
    
    async def _enrich_evaluation_results(
        self, 
        ragas_results: Dict[str, Any],
        interaction_ids: List[str],
        evaluation_start: datetime
    ) -> Dict[str, Any]:
        """Enriquece resultados RAGAS com métricas de negócio"""
        
        evaluation_duration = (datetime.now() - evaluation_start).total_seconds()
        
        # Calcular métricas adicionais de negócio
        business_metrics = await self._calculate_business_metrics(ragas_results)
        
        # Comparar com avaliações históricas
        historical_comparison = await self._get_historical_comparison(ragas_results)
        
        enriched = {
            **ragas_results,  # Resultados originais do RAGAS
            "evaluation_metadata": {
                "evaluation_duration_seconds": round(evaluation_duration, 3),
                "timestamp": evaluation_start.isoformat(),
                "interaction_ids_evaluated": interaction_ids,
                "evaluation_version": "2.0"
            },
            "business_metrics": business_metrics,
            "historical_comparison": historical_comparison,
            "evaluation_status": "success"
        }
        
        return enriched
    
    async def _calculate_business_metrics(self, ragas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula métricas de negócio baseadas nos resultados RAGAS"""
        
        avg_scores = ragas_results.get("average_scores", {})
        individual_scores = ragas_results.get("individual_scores", [])
        
        # Classificar qualidade geral
        overall_score = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
        
        if overall_score >= 0.8:
            quality_grade = "Excellent"
        elif overall_score >= 0.7:
            quality_grade = "Good"
        elif overall_score >= 0.6:
            quality_grade = "Fair"
        else:
            quality_grade = "Poor"
        
        # Calcular distribuição de scores
        score_distribution = self._calculate_score_distribution(individual_scores)
        
        return {
            "overall_quality_score": round(overall_score, 3),
            "quality_grade": quality_grade,
            "score_distribution": score_distribution,
            "total_interactions_evaluated": len(individual_scores)
        }
    
    def _calculate_score_distribution(self, individual_scores: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calcula distribuição de scores para análise"""
        
        distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        
        for score_data in individual_scores:
            # Calcular score médio da interação
            scores = []
            for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
                if score_data.get(metric) is not None:
                    scores.append(score_data[metric])
            
            if scores:
                avg_score = sum(scores) / len(scores)
                
                if avg_score >= 0.8:
                    distribution["excellent"] += 1
                elif avg_score >= 0.7:
                    distribution["good"] += 1
                elif avg_score >= 0.6:
                    distribution["fair"] += 1
                else:
                    distribution["poor"] += 1
        
        return distribution
    
    async def _get_historical_comparison(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compara resultados atuais com histórico"""
        
        try:
            # Buscar avaliações dos últimos 30 dias
            cutoff_date = datetime.now() - timedelta(days=30)
            
            async with AsyncSessionLocal() as session:
                query = select(RAGInteractionDB).where(
                    RAGInteractionDB.timestamp >= cutoff_date,
                    RAGInteractionDB.ragas_scores.is_not(None)
                )
                
                result = await session.execute(query)
                historical_interactions = result.scalars().all()
            
            if not historical_interactions:
                return {"status": "no_historical_data", "message": "Sem dados históricos para comparação"}
            
            # Calcular médias históricas
            historical_averages = self._calculate_historical_averages(historical_interactions)
            current_averages = current_results.get("average_scores", {})
            
            # Comparar tendências
            trends = {}
            for metric in historical_averages:
                if metric in current_averages:
                    historical_avg = historical_averages[metric]
                    current_avg = current_averages[metric]
                    
                    difference = current_avg - historical_avg
                    if difference > 0.05:
                        trend = "improving"
                    elif difference < -0.05:
                        trend = "declining"
                    else:
                        trend = "stable"
                    
                    trends[metric] = {
                        "trend": trend,
                        "difference": round(difference, 3),
                        "current": round(current_avg, 3),
                        "historical": round(historical_avg, 3)
                    }
            
            return {
                "status": "comparison_available",
                "historical_period_days": 30,
                "historical_interactions_count": len(historical_interactions),
                "trends": trends
            }
            
        except Exception as e:
            return {"status": "comparison_error", "error": str(e)}
    
    def _calculate_historical_averages(self, interactions: List[RAGInteractionDB]) -> Dict[str, float]:
        """Calcula médias históricas das métricas RAGAS"""
        
        metric_sums = {"faithfulness": 0, "answer_relevancy": 0, "context_precision": 0}
        metric_counts = {"faithfulness": 0, "answer_relevancy": 0, "context_precision": 0}
        
        for interaction in interactions:
            if interaction.ragas_scores:
                for metric in metric_sums:
                    if metric in interaction.ragas_scores and interaction.ragas_scores[metric] is not None:
                        metric_sums[metric] += interaction.ragas_scores[metric]
                        metric_counts[metric] += 1
        
        averages = {}
        for metric in metric_sums:
            if metric_counts[metric] > 0:
                averages[metric] = metric_sums[metric] / metric_counts[metric]
        
        return averages
    
    def _generate_quality_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Gera recomendações de negócio baseadas nos resultados"""
        
        recommendations = []
        avg_scores = evaluation_results.get("average_scores", {})
        business_metrics = evaluation_results.get("business_metrics", {})
        
        # Recomendações baseadas em scores baixos
        if avg_scores.get("faithfulness", 1) < 0.7:
            recommendations.append(
                "🔍 FIDELIDADE BAIXA: Revisar qualidade dos documentos de base. "
                "Considerar limpeza de dados e melhoria do chunking."
            )
        
        if avg_scores.get("answer_relevancy", 1) < 0.8:
            recommendations.append(
                "🎯 RELEVÂNCIA BAIXA: Ajustar prompt do LLM para respostas mais focadas. "
                "Considerar fine-tuning ou mudança de modelo."
            )
        
        if avg_scores.get("context_precision", 1) < 0.6:
            recommendations.append(
                "📊 PRECISÃO BAIXA: Otimizar parâmetros de busca vetorial. "
                "Considerar re-embedding ou ajuste de similarity threshold."
            )
        
        # Recomendações baseadas em distribuição
        distribution = business_metrics.get("score_distribution", {})
        poor_percentage = distribution.get("poor", 0) / max(sum(distribution.values()), 1)
        
        if poor_percentage > 0.3:
            recommendations.append(
                "⚠️ MUITAS INTERAÇÕES RUINS: >30% das interações têm qualidade baixa. "
                "Revisão geral do sistema necessária."
            )
        
        # Recomendações baseadas em tendências
        trends = evaluation_results.get("historical_comparison", {}).get("trends", {})
        declining_metrics = [metric for metric, data in trends.items() 
                           if data.get("trend") == "declining"]
        
        if declining_metrics:
            recommendations.append(
                f"📉 TENDÊNCIA NEGATIVA: Métricas em declínio: {', '.join(declining_metrics)}. "
                "Investigar mudanças recentes no sistema."
            )
        
        # Se tudo estiver bem
        if not recommendations:
            recommendations.append("✅ QUALIDADE EXCELENTE: Sistema funcionando dentro dos padrões esperados.")
        
        return recommendations
    
    def _log_evaluation_metrics(self, results: Dict[str, Any], evaluation_start: datetime) -> None:
        """Logging estruturado das métricas de avaliação"""
        
        avg_scores = results.get("average_scores", {})
        business_metrics = results.get("business_metrics", {})
        
        logger.info(
            f"📊 RAGAS Evaluation Completed - "
            f"Quality Grade: {business_metrics.get('quality_grade', 'N/A')}, "
            f"Overall Score: {business_metrics.get('overall_quality_score', 0):.3f}, "
            f"Interactions: {results.get('total_interactions', 0)}, "
            f"Duration: {results.get('evaluation_metadata', {}).get('evaluation_duration_seconds', 0)}s"
        )
    
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

# Exceção específica para erros de negócio de avaliação
class EvaluationBusinessException(Exception):
    """Exceção para erros de lógica de negócio em operações de avaliação"""
    
    def __init__(self, message: str, error_code: str = "EVALUATION_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)