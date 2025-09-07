# EVALUATION ROUTES - Interface HTTP para Avaliação RAGAS
# Route é responsável apenas por HTTP: validação, serialização, tratamento de erros
# Toda lógica de avaliação e relatórios fica no EvaluationController

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
from app.models.rag_interaction import (
    RAGInteractionResponse, 
    UserFeedback, 
    RAGASEvaluation
)
from app.controllers.evaluation_controller import EvaluationController, EvaluationBusinessException
from app.services.ragas_service import ragas_service
from app.services.database_service import AsyncSessionLocal
from app.services.phoenix_service import phoenix_service
from app.models.rag_interaction import RAGInteractionDB
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
evaluation_controller = EvaluationController()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.post("/ragas/evaluate")
async def run_ragas_evaluation(
    background_tasks: BackgroundTasks,
    evaluation_request: RAGASEvaluation = Depends(lambda: RAGASEvaluation())
) -> Dict[str, Any]:
    """
    ENDPOINT RAGAS: Executa avaliação de qualidade em interações RAG
    
    RESPONSABILIDADES DA ROUTE:
    - Validação de parâmetros HTTP
    - Chamada para controller (lógica RAGAS)  
    - Tratamento de background tasks
    - Serialização da resposta
    - Tratamento de exceções HTTP
    
    Args:
        evaluation_request: Parâmetros da avaliação RAGAS
        background_tasks: Para execução em background (futuro)
    
    Returns:
        Dict com resultados detalhados da avaliação RAGAS
        
    Raises:
        HTTPException: Para erros HTTP (400, 404, 500)
    """
    try:
        # CHAMA CONTROLLER: toda lógica RAGAS está lá
        evaluation_results = await evaluation_controller.execute_ragas_evaluation(
            evaluation_request,
            background_execution=False  # Por enquanto, execução síncrona
        )
        
        # VERIFICAR SE HOUVE ERRO DE NEGÓCIO
        if evaluation_results.get("evaluation_status") == "error":
            error_details = evaluation_results.get("error_details", {})
            error_code = error_details.get("error_code", "UNKNOWN")
            
            if error_code == "NO_INTERACTIONS":
                raise HTTPException(status_code=404, detail=error_details.get("message"))
            elif error_code in ["TOO_MANY_INTERACTIONS", "INVALID_INTERACTION_IDS"]:
                raise HTTPException(status_code=400, detail=error_details.get("message"))
            else:
                raise HTTPException(status_code=500, detail=error_details.get("message"))
        
        # SUCESSO: resposta enriquecida com métricas de negócio
        print(evaluation_results)
        return {
            "message": "Avaliação RAGAS concluída com sucesso",
            "evaluation_results": evaluation_results
        }
        
    except EvaluationBusinessException as e:
        # Exceções de negócio específicas
        if e.error_code == "NO_INTERACTIONS":
            raise HTTPException(status_code=404, detail=e.message)
        elif e.error_code in ["TOO_MANY_INTERACTIONS", "INVALID_INTERACTION_IDS"]:
            raise HTTPException(status_code=400, detail=e.message)
        else:
            raise HTTPException(status_code=400, detail=e.message)
            
    except Exception as e:
        # Erros técnicos inesperados
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno durante avaliação RAGAS: {str(e)}"
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
