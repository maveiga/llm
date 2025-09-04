from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
from app.models.rag_interaction import (
    RAGInteractionResponse, 
    UserFeedback, 
    RAGASEvaluation
)
from app.services.ragas_service import ragas_service
from app.services.database_service import AsyncSessionLocal
from app.services.phoenix_service import phoenix_service
from app.models.rag_interaction import RAGInteractionDB
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.post("/ragas/evaluate")
async def run_ragas_evaluation(
    evaluation_request: RAGASEvaluation,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Executa avaliação RAGAS em um conjunto de interações
    
    Args:
        evaluation_request: Parâmetros da avaliação
        background_tasks: Para executar a avaliação em background
    
    Returns:
        Dict com resultados da avaliação RAGAS
    """
    try:
        # Se não foram fornecidos IDs específicos, pegar as últimas interações
        if not evaluation_request.interaction_ids:
            async with AsyncSessionLocal() as session:
                query = select(RAGInteractionDB).order_by(
                    desc(RAGInteractionDB.timestamp)
                ).limit(50)
                
                result = await session.execute(query)
                interactions = result.scalars().all()
                
                if not interactions:
                    raise HTTPException(
                        status_code=404, 
                        detail="Nenhuma interação encontrada para avaliar"
                    )
                
                evaluation_request.interaction_ids = [i.id for i in interactions]
        
        # Executar avaliação
        results = await ragas_service.evaluate_interactions(
            interaction_ids=evaluation_request.interaction_ids
        )
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return {
            "message": "Avaliação RAGAS concluída com sucesso",
            "evaluation_results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro durante avaliação RAGAS: {str(e)}"
        )

@router.get("/ragas/report")
async def get_quality_report(days: int = 30) -> Dict[str, Any]:
    """
    Gera relatório de qualidade das interações RAG
    
    Args:
        days: Número de dias para incluir no relatório
    
    Returns:
        Dict com estatísticas e métricas de qualidade
    """
    try:
        report = await ragas_service.get_quality_report(days=days)
        
        if "error" in report:
            raise HTTPException(status_code=404, detail=report["error"])
        
        return report
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao gerar relatório de qualidade: {str(e)}"
        )

@router.get("/interactions")
async def list_interactions(
    limit: int = 50,
    offset: int = 0,
    with_ragas_scores: bool = False,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Lista interações RAG armazenadas
    
    Args:
        limit: Número máximo de interações para retornar
        offset: Número de interações para pular
        with_ragas_scores: Filtrar apenas interações com scores RAGAS
        db: Sessão do banco de dados
    
    Returns:
        Dict com lista de interações e metadados
    """
    try:
        query = select(RAGInteractionDB).order_by(
            desc(RAGInteractionDB.timestamp)
        ).offset(offset).limit(limit)
        
        if with_ragas_scores:
            query = query.where(RAGInteractionDB.ragas_scores.is_not(None))
        
        result = await db.execute(query)
        interactions = result.scalars().all()
        
        # Contar total
        count_query = select(RAGInteractionDB)
        if with_ragas_scores:
            count_query = count_query.where(RAGInteractionDB.ragas_scores.is_not(None))
        
        total_result = await db.execute(count_query)
        total_count = len(total_result.scalars().all())
        
        return {
            "interactions": [
                {
                    "id": i.id,
                    "timestamp": i.timestamp,
                    "question": i.question[:100] + "..." if len(i.question) > 100 else i.question,
                    "answer_preview": i.answer[:200] + "..." if len(i.answer) > 200 else i.answer,
                    "context_count": len(i.contexts) if i.contexts else 0,
                    "sources_count": len(i.sources) if i.sources else 0,
                    "response_time": i.response_time,
                    "has_ragas_scores": i.ragas_scores is not None,
                    "ragas_scores": i.ragas_scores,
                    "user_feedback": i.user_feedback
                }
                for i in interactions
            ],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_count,
                "has_more": offset + limit < total_count
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao listar interações: {str(e)}"
        )

@router.get("/interactions/{interaction_id}")
async def get_interaction_detail(
    interaction_id: str,
    db: AsyncSession = Depends(get_db)
) -> RAGInteractionResponse:
    """
    Obtém detalhes completos de uma interação específica
    
    Args:
        interaction_id: ID da interação
        db: Sessão do banco de dados
    
    Returns:
        Detalhes completos da interação
    """
    try:
        query = select(RAGInteractionDB).where(
            RAGInteractionDB.id == interaction_id
        )
        
        result = await db.execute(query)
        interaction = result.scalar_one_or_none()
        
        if not interaction:
            raise HTTPException(
                status_code=404, 
                detail="Interação não encontrada"
            )
        
        return RAGInteractionResponse.from_orm(interaction)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao buscar interação: {str(e)}"
        )

@router.post("/interactions/{interaction_id}/feedback")
async def add_user_feedback(
    interaction_id: str,
    feedback: UserFeedback,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Adiciona feedback do usuário para uma interação
    
    Args:
        interaction_id: ID da interação
        feedback: Feedback do usuário (rating e comentário opcional)
        db: Sessão do banco de dados
    
    Returns:
        Mensagem de confirmação
    """
    try:
        query = select(RAGInteractionDB).where(
            RAGInteractionDB.id == interaction_id
        )
        
        result = await db.execute(query)
        interaction = result.scalar_one_or_none()
        
        if not interaction:
            raise HTTPException(
                status_code=404, 
                detail="Interação não encontrada"
            )
        
        # Validar rating
        if feedback.rating < 1 or feedback.rating > 5:
            raise HTTPException(
                status_code=400, 
                detail="Rating deve ser entre 1 e 5"
            )
        
        interaction.user_feedback = feedback.rating
        await db.commit()
        
        return {
            "message": "Feedback adicionado com sucesso",
            "interaction_id": interaction_id,
            "rating": feedback.rating
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao adicionar feedback: {str(e)}"
        )

@router.get("/stats/overview")
async def get_evaluation_stats(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Obtém estatísticas gerais das avaliações
    
    Returns:
        Dict com estatísticas gerais
    """
    try:
        # Buscar todas as interações
        all_interactions_query = select(RAGInteractionDB)
        all_result = await db.execute(all_interactions_query)
        all_interactions = all_result.scalars().all()
        
        # Buscar interações com scores RAGAS
        ragas_interactions_query = select(RAGInteractionDB).where(
            RAGInteractionDB.ragas_scores.is_not(None)
        )
        ragas_result = await db.execute(ragas_interactions_query)
        ragas_interactions = ragas_result.scalars().all()
        
        # Buscar interações com feedback
        feedback_interactions_query = select(RAGInteractionDB).where(
            RAGInteractionDB.user_feedback.is_not(None)
        )
        feedback_result = await db.execute(feedback_interactions_query)
        feedback_interactions = feedback_result.scalars().all()
        
        # Calcular estatísticas
        avg_response_time = 0
        if all_interactions:
            response_times = [i.response_time for i in all_interactions if i.response_time]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
        
        avg_user_rating = 0
        if feedback_interactions:
            ratings = [i.user_feedback for i in feedback_interactions if i.user_feedback]
            if ratings:
                avg_user_rating = sum(ratings) / len(ratings)
        
        return {
            "total_interactions": len(all_interactions),
            "interactions_with_ragas": len(ragas_interactions),
            "interactions_with_feedback": len(feedback_interactions),
            "average_response_time": round(avg_response_time, 3),
            "average_user_rating": round(avg_user_rating, 2),
            "coverage": {
                "ragas_coverage": round(len(ragas_interactions) / len(all_interactions) * 100, 1) if all_interactions else 0,
                "feedback_coverage": round(len(feedback_interactions) / len(all_interactions) * 100, 1) if all_interactions else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao obter estatísticas: {str(e)}"
        )

# Endpoints específicos do Phoenix
@router.get("/phoenix/status")
async def get_phoenix_status() -> Dict[str, Any]:
    """
    Verifica status do Phoenix e retorna informações de conexão
    
    Returns:
        Dict com status e URL do Phoenix
    """
    try:
        return {
            "phoenix_enabled": phoenix_service.is_enabled,
            "dashboard_url": phoenix_service.get_phoenix_url(),
            "project_name": phoenix_service.project_name,
            "instrumentation_active": phoenix_service.is_enabled
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao verificar status do Phoenix: {str(e)}"
        )

@router.get("/phoenix/traces")
async def get_phoenix_traces(limit: int = 100) -> Dict[str, Any]:
    """
    Obtém informações sobre traces disponíveis no Phoenix
    
    Args:
        limit: Número máximo de traces para retornar informações
    
    Returns:
        Dict com informações dos traces
    """
    try:
        traces_data = phoenix_service.get_traces_data(limit=limit)
        return traces_data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter traces do Phoenix: {str(e)}"
        )

@router.get("/phoenix/embeddings-analysis")
async def get_embeddings_analysis() -> Dict[str, Any]:
    """
    Obtém análise de embeddings do Phoenix
    
    Returns:
        Dict com informações da análise de embeddings
    """
    try:
        analysis = phoenix_service.generate_embeddings_analysis()
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na análise de embeddings: {str(e)}"
        )

@router.post("/phoenix/export-for-ragas")
async def export_phoenix_traces_for_ragas(
    interaction_ids: List[str] = None
) -> Dict[str, Any]:
    """
    Exporta traces do Phoenix para integração com RAGAS
    
    Args:
        interaction_ids: Lista de IDs de interações para exportar
    
    Returns:
        Dict com dados exportados
    """
    try:
        exported_data = phoenix_service.export_traces_for_ragas(interaction_ids)
        return exported_data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao exportar traces: {str(e)}"
        )

@router.get("/phoenix/dashboard")
async def get_phoenix_dashboard_info() -> Dict[str, Any]:
    """
    Retorna informações para acessar o dashboard do Phoenix
    
    Returns:
        Dict com informações do dashboard
    """
    try:
        if not phoenix_service.is_enabled:
            return {
                "error": "Phoenix não está habilitado",
                "instructions": [
                    "1. Certifique-se que as dependências do Phoenix estão instaladas",
                    "2. Reinicie o servidor da aplicação",
                    "3. Verifique os logs de inicialização"
                ]
            }
        
        return {
            "dashboard_url": phoenix_service.get_phoenix_url(),
            "status": "active",
            "features": [
                "Traces de consultas RAG em tempo real",
                "Visualização de embeddings e clusters",
                "Análise de performance e latência",
                "Detecção de anomalias e drift"
            ],
            "instructions": [
                "1. Acesse a URL do dashboard",
                "2. Faça algumas consultas RAG para gerar traces",
                "3. Explore as abas: Traces, Embeddings, Performance",
                "4. Use filtros para análises específicas"
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter informações do dashboard: {str(e)}"
        )

# Endpoint combinado Phoenix + RAGAS
@router.get("/combined-report")
async def get_combined_phoenix_ragas_report() -> Dict[str, Any]:
    """
    Gera relatório combinado integrando dados do Phoenix com métricas RAGAS
    
    Returns:
        Dict com análise combinada e recomendações
    """
    try:
        combined_report = await ragas_service.generate_phoenix_ragas_report()
        return combined_report
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar relatório combinado: {str(e)}"
        )