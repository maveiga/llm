# SEARCH CONTROLLER - Lógica de Negócio para Busca Vetorial
# Controller orquestra buscas diretas, filtros e análise de resultados

from typing import Dict, Any, List
from app.services.vector_service import VectorService
from app.models.document import SearchRequest, SearchResponse
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SearchBusinessException(Exception):
    """Exceção para erros de negócio específicos da busca"""
    def __init__(self, message: str, error_code: str):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

class SearchController:
    """
    Controller para operações de busca vetorial direta
    
    RESPONSABILIDADES:
    - Lógica de negócio para buscas vetoriais
    - Validações de parâmetros de busca
    - Enriquecimento de resultados com métricas
    - Análise de qualidade dos resultados
    - Filtros e ordenação avançada
    - Logging e auditoria de buscas
    
    PADRÃO MVC:
    Route → Controller → VectorService → Response
    """
    
    def __init__(self):
        self.vector_service = VectorService()
    
    async def execute_search(self, search_request: SearchRequest) -> Dict[str, Any]:
        """
        LÓGICA DE NEGÓCIO: Executa busca vetorial com validações e enriquecimento
        
        PROCESSO DE NEGÓCIO:
        1. Validar parâmetros de busca
        2. Executar busca vetorial via service
        3. Analisar qualidade dos resultados
        4. Enriquecer com métricas de negócio
        5. Aplicar filtros adicionais se necessário
        6. Logging e auditoria
        
        Args:
            search_request: Parâmetros da busca vetorial
            
        Returns:
            SearchResponse enriquecida com métricas de negócio
        """
        search_start = datetime.now()
        logger.info(f"🔍 Iniciando busca vetorial: '{search_request.query[:50]}...'")
        
        try:
            # ETAPA 1: VALIDAÇÕES DE NEGÓCIO
            self._validate_search_request(search_request)
            
            # ETAPA 2: EXECUTAR BUSCA VETORIAL
            search_results = await self.vector_service.search_documents(
                query=search_request.query,
                limit=search_request.limit,
                category_filter=search_request.category_filter
            )
            
            # ETAPA 3: ANALISAR QUALIDADE DOS RESULTADOS
            quality_analysis = self._analyze_search_quality(search_results, search_request)
            
            # ETAPA 4: ENRIQUECER COM MÉTRICAS DE NEGÓCIO
            enriched_response = self._enrich_search_response(
                search_results, 
                search_request, 
                quality_analysis,
                search_start
            )
            
            # ETAPA 5: APLICAR FILTROS AVANÇADOS (se necessário)
            if search_request.limit < len(search_results):
                enriched_response = self._apply_advanced_filtering(enriched_response, search_request)
            
            # ETAPA 6: LOGGING E AUDITORIA
            self._log_search_metrics(enriched_response, search_start)
            
            logger.info(f"✅ Busca concluída: {len(search_results)} documentos encontrados")
            return enriched_response
            
        except SearchBusinessException as e:
            logger.error(f"❌ Erro de negócio na busca: {e.message}")
            return self._create_search_error_response(e.message, e.error_code, search_start)
            
        except Exception as e:
            logger.error(f"❌ Erro técnico na busca: {str(e)}")
            return self._create_search_error_response(
                "Erro interno durante busca",
                "TECHNICAL_ERROR",
                search_start,
                technical_details=str(e)
            )
    
    def _validate_search_request(self, request: SearchRequest) -> None:
        """Validações de negócio para parâmetros de busca"""
        
        # Validar query não vazia
        if not request.query or not request.query.strip():
            raise SearchBusinessException(
                "Query de busca não pode estar vazia",
                error_code="EMPTY_QUERY"
            )
        
        # Validar tamanho da query
        if len(request.query.strip()) < 2:
            raise SearchBusinessException(
                "Query muito curta - mínimo 2 caracteres",
                error_code="QUERY_TOO_SHORT"
            )
        
        if len(request.query) > 1000:
            raise SearchBusinessException(
                "Query muito longa - máximo 1000 caracteres",
                error_code="QUERY_TOO_LONG"
            )
        
        # Validar limite de resultados
        if request.limit < 1 or request.limit > 50:
            raise SearchBusinessException(
                "Limit deve estar entre 1 e 50",
                error_code="INVALID_LIMIT"
            )
        
        # Validar filtro de categoria se fornecido
        if request.category_filter:
            if len(request.category_filter.strip()) == 0:
                raise SearchBusinessException(
                    "Filtro de categoria não pode estar vazio",
                    error_code="EMPTY_CATEGORY_FILTER"
                )
        
        logger.info(f"✅ Parâmetros de busca validados: query={len(request.query)} chars, limit={request.limit}")
    
    def _analyze_search_quality(
        self, 
        search_results: List[Any], 
        search_request: SearchRequest
    ) -> Dict[str, Any]:
        """Analisa qualidade dos resultados da busca"""
        
        if not search_results:
            return {
                "quality_grade": "no_results",
                "average_similarity": 0.0,
                "similarity_range": {"min": 0.0, "max": 0.0},
                "results_diversity": 0.0,
                "category_distribution": {}
            }
        
        # Calcular métricas de similaridade
        similarities = [doc.similarity_score for doc in search_results if hasattr(doc, 'similarity_score')]
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            similarity_range = {"min": min(similarities), "max": max(similarities)}
            
            # Classificar qualidade baseada na similaridade média
            if avg_similarity >= 0.8:
                quality_grade = "excellent"
            elif avg_similarity >= 0.6:
                quality_grade = "good"
            elif avg_similarity >= 0.4:
                quality_grade = "fair"
            else:
                quality_grade = "poor"
        else:
            avg_similarity = 0.0
            similarity_range = {"min": 0.0, "max": 0.0}
            quality_grade = "unknown"
        
        # Analisar diversidade de categorias
        categories = [doc.category for doc in search_results if hasattr(doc, 'category')]
        category_distribution = {}
        for category in categories:
            category_distribution[category] = category_distribution.get(category, 0) + 1
        
        results_diversity = len(category_distribution) / len(search_results) if search_results else 0
        
        return {
            "quality_grade": quality_grade,
            "average_similarity": round(avg_similarity, 3),
            "similarity_range": {
                "min": round(similarity_range["min"], 3),
                "max": round(similarity_range["max"], 3)
            },
            "results_diversity": round(results_diversity, 3),
            "category_distribution": category_distribution
        }
    
    def _enrich_search_response(
        self,
        search_results: List[Any],
        search_request: SearchRequest,
        quality_analysis: Dict[str, Any],
        search_start: datetime
    ) -> Dict[str, Any]:
        """Enriquece resposta da busca com métricas de negócio"""
        
        search_duration = (datetime.now() - search_start).total_seconds()
        
        # Converter resultados para formato padronizado
        formatted_results = []
        for i, doc in enumerate(search_results):
            # Gerar ID único se não existir
            doc_id = getattr(doc, 'id', None) or f"doc_{i}_{hash(str(doc))%10000}"
            
            formatted_results.append({
                "id": doc_id,  # Campo obrigatório para DocumentResponse
                "title": getattr(doc, 'title', 'Sem título'),
                "category": getattr(doc, 'category', 'sem_categoria'),
                "content": getattr(doc, 'content', ''),
                "similarity_score": getattr(doc, 'similarity_score', 0.0),
                "metadata": getattr(doc, 'metadata', {})
            })
        
        # Criar resposta enriquecida
        enriched_response = {
            "query": search_request.query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "search_metadata": {
                "search_duration_seconds": round(search_duration, 3),
                "timestamp": search_start.isoformat(),
                "parameters": {
                    "limit": search_request.limit,
                    "category_filter": search_request.category_filter,
                    "query_length": len(search_request.query)
                }
            },
            "quality_analysis": quality_analysis,
            "business_insights": self._generate_search_insights(quality_analysis, formatted_results),
            "search_status": "success"
        }
        
        return enriched_response
    
    def _generate_search_insights(
        self, 
        quality_analysis: Dict[str, Any], 
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Gera insights de negócio baseados nos resultados da busca"""
        
        insights = {
            "search_effectiveness": self._assess_search_effectiveness(quality_analysis),
            "content_coverage": self._analyze_content_coverage(results),
            "recommendations": []
        }
        
        # Gerar recomendações baseadas na qualidade
        quality_grade = quality_analysis.get("quality_grade", "unknown")
        
        if quality_grade == "poor":
            insights["recommendations"].append(
                "🔍 Qualidade baixa dos resultados. Considere reformular a query ou verificar se há documentos relevantes indexados."
            )
        elif quality_grade == "no_results":
            insights["recommendations"].append(
                "📭 Nenhum resultado encontrado. Tente termos mais gerais ou verifique se os documentos foram carregados."
            )
        elif quality_grade == "excellent":
            insights["recommendations"].append(
                "✅ Excelente qualidade dos resultados! Query bem formulada e documentos relevantes encontrados."
            )
        
        # Recomendações baseadas na diversidade
        diversity = quality_analysis.get("results_diversity", 0)
        if diversity < 0.3:
            insights["recommendations"].append(
                "📊 Baixa diversidade de categorias. Resultados muito concentrados em um tipo de documento."
            )
        
        return insights
    
    def _assess_search_effectiveness(self, quality_analysis: Dict[str, Any]) -> str:
        """Avalia efetividade geral da busca"""
        
        quality_grade = quality_analysis.get("quality_grade", "unknown")
        avg_similarity = quality_analysis.get("average_similarity", 0)
        diversity = quality_analysis.get("results_diversity", 0)
        
        # Algoritmo de efetividade considerando múltiplos fatores
        effectiveness_score = 0
        
        if quality_grade == "excellent":
            effectiveness_score += 0.4
        elif quality_grade == "good":
            effectiveness_score += 0.3
        elif quality_grade == "fair":
            effectiveness_score += 0.2
        
        if avg_similarity > 0.7:
            effectiveness_score += 0.3
        elif avg_similarity > 0.5:
            effectiveness_score += 0.2
        
        if diversity > 0.5:
            effectiveness_score += 0.3
        elif diversity > 0.3:
            effectiveness_score += 0.2
        
        if effectiveness_score >= 0.8:
            return "highly_effective"
        elif effectiveness_score >= 0.6:
            return "effective"
        elif effectiveness_score >= 0.4:
            return "moderately_effective"
        else:
            return "low_effectiveness"
    
    def _analyze_content_coverage(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa cobertura de conteúdo dos resultados"""
        
        if not results:
            return {"coverage_assessment": "no_content", "content_types": {}}
        
        # Analisar tipos de conteúdo baseado no tamanho
        content_types = {"short": 0, "medium": 0, "long": 0}
        total_content_chars = 0
        
        for result in results:
            content_length = len(result.get("content", ""))
            total_content_chars += content_length
            
            if content_length < 200:
                content_types["short"] += 1
            elif content_length < 800:
                content_types["medium"] += 1
            else:
                content_types["long"] += 1
        
        avg_content_length = total_content_chars / len(results)
        
        # Avaliar cobertura
        if avg_content_length > 500 and content_types["long"] > 0:
            coverage_assessment = "comprehensive"
        elif avg_content_length > 200:
            coverage_assessment = "adequate"
        else:
            coverage_assessment = "limited"
        
        return {
            "coverage_assessment": coverage_assessment,
            "content_types": content_types,
            "average_content_length": round(avg_content_length, 0),
            "total_content_chars": total_content_chars
        }
    
    def _apply_advanced_filtering(
        self, 
        response: Dict[str, Any], 
        request: SearchRequest
    ) -> Dict[str, Any]:
        """Aplica filtros avançados se necessário"""
        
        # Por enquanto, apenas log da funcionalidade
        logger.info("🔧 Filtros avançados: funcionalidade disponível para implementação futura")
        
        # Adicionar flag indicando que filtros avançados estão disponíveis
        response["advanced_filtering"] = {
            "available": True,
            "applied": False,
            "options": ["similarity_threshold", "content_length", "recency"]
        }
        
        return response
    
    def _log_search_metrics(self, response: Dict[str, Any], search_start: datetime) -> None:
        """Logging estruturado das métricas de busca"""
        
        metadata = response.get("search_metadata", {})
        quality = response.get("quality_analysis", {})
        insights = response.get("business_insights", {})
        
        logger.info(
            f"📊 Search Metrics - "
            f"Results: {response.get('total_results', 0)}, "
            f"Quality: {quality.get('quality_grade', 'N/A')}, "
            f"Avg Similarity: {quality.get('average_similarity', 0):.3f}, "
            f"Effectiveness: {insights.get('search_effectiveness', 'N/A')}, "
            f"Duration: {metadata.get('search_duration_seconds', 0)}s"
        )
    
    def _create_search_error_response(
        self, 
        message: str, 
        error_code: str, 
        search_start: datetime,
        technical_details: str = None
    ) -> Dict[str, Any]:
        """Cria resposta padronizada para erros de busca"""
        
        search_duration = (datetime.now() - search_start).total_seconds()
        
        return {
            "query": "",
            "results": [],
            "total_results": 0,
            "search_status": "error",
            "error_details": {
                "message": message,
                "error_code": error_code,
                "technical_details": technical_details,
                "search_duration_seconds": round(search_duration, 3),
                "timestamp": search_start.isoformat()
            }
        }

# Exceção específica para erros de negócio de busca
class SearchBusinessException(Exception):
    """Exceção para erros de lógica de negócio em operações de busca"""
    
    def __init__(self, message: str, error_code: str = "SEARCH_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)