from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from sqlalchemy import select
from app.models.rag_interaction import RAGInteractionDB
from app.services.database_service import AsyncSessionLocal
from app.services.phoenix_service import phoenix_service
import asyncio

class RAGASService:
    def __init__(self):
        # Métricas do RAGAS que vamos usar
        self.metrics = [
            faithfulness,      # Resposta é fiel ao contexto?
            answer_relevancy,  # Resposta é relevante à pergunta?
            context_precision, # Contextos relevantes estão bem rankeados?
            # context_recall   # Requer ground_truth - ativar quando tiver
        ]

    async def evaluate_interactions(
        self, 
        interaction_ids: List[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Avalia um conjunto de interações usando RAGAS
        
        Args:
            interaction_ids: Lista de IDs específicos para avaliar
            limit: Número máximo de interações para avaliar (se interaction_ids não fornecido)
            
        Returns:
            Dict com scores médios e individuais
        """
        
        # Buscar interações no banco de dados
        async with AsyncSessionLocal() as session:
            if interaction_ids:
                query = select(RAGInteractionDB).where(
                    RAGInteractionDB.id.in_(interaction_ids)
                )
            else:
                query = select(RAGInteractionDB).limit(limit)
            
            result = await session.execute(query)
            interactions = result.scalars().all()

        if not interactions:
            return {"error": "Nenhuma interação encontrada para avaliar"}

        # Preparar dados para o RAGAS
        data = []
        for interaction in interactions:
            data.append({
                'question': interaction.question,
                'answer': interaction.answer,
                'contexts': interaction.contexts,
                'ground_truth': None  # Pode ser adicionado no futuro
            })

        # Converter para Dataset do RAGAS
        dataset = Dataset.from_pandas(pd.DataFrame(data))

        try:
            # Executar avaliação RAGAS
            result = evaluate(dataset, metrics=self.metrics)
            
            # Processar resultados
            evaluation_results = {
                'total_interactions': len(interactions),
                'average_scores': {
                    'faithfulness': float(result['faithfulness']),
                    'answer_relevancy': float(result['answer_relevancy']),
                    'context_precision': float(result['context_precision']),
                },
                'individual_scores': []
            }

            # Adicionar scores individuais se necessário
            for i, interaction in enumerate(interactions):
                individual_score = {
                    'interaction_id': interaction.id,
                    'question': interaction.question[:100] + "..." if len(interaction.question) > 100 else interaction.question,
                    'faithfulness': float(result['faithfulness'][i]) if i < len(result['faithfulness']) else None,
                    'answer_relevancy': float(result['answer_relevancy'][i]) if i < len(result['answer_relevancy']) else None,
                    'context_precision': float(result['context_precision'][i]) if i < len(result['context_precision']) else None,
                }
                evaluation_results['individual_scores'].append(individual_score)

            # Salvar scores no banco de dados
            await self._save_ragas_scores(interactions, evaluation_results['individual_scores'])

            # Integração com Phoenix: registrar avaliação RAGAS
            if phoenix_service.is_enabled:
                self._log_ragas_evaluation_to_phoenix(evaluation_results, interactions)

            return evaluation_results

        except Exception as e:
            return {
                'error': f"Erro durante avaliação RAGAS: {str(e)}",
                'total_interactions': len(interactions)
            }

    async def _save_ragas_scores(
        self, 
        interactions: List[RAGInteractionDB], 
        individual_scores: List[Dict]
    ):
        """Salva os scores RAGAS no banco de dados"""
        async with AsyncSessionLocal() as session:
            for interaction, scores in zip(interactions, individual_scores):
                # Remove campos desnecessários para salvar
                ragas_scores = {
                    'faithfulness': scores.get('faithfulness'),
                    'answer_relevancy': scores.get('answer_relevancy'),
                    'context_precision': scores.get('context_precision'),
                }
                
                interaction.ragas_scores = ragas_scores
                session.add(interaction)
            
            await session.commit()

    async def get_quality_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Gera um relatório de qualidade das interações
        
        Args:
            days: Número de dias para incluir no relatório
            
        Returns:
            Dict com estatísticas e tendências
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        async with AsyncSessionLocal() as session:
            query = select(RAGInteractionDB).where(
                RAGInteractionDB.timestamp >= cutoff_date,
                RAGInteractionDB.ragas_scores.is_not(None)
            )
            
            result = await session.execute(query)
            interactions = result.scalars().all()

        if not interactions:
            return {"error": f"Nenhuma interação avaliada nos últimos {days} dias"}

        # Calcular estatísticas
        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []
        
        for interaction in interactions:
            if interaction.ragas_scores:
                if interaction.ragas_scores.get('faithfulness'):
                    faithfulness_scores.append(interaction.ragas_scores['faithfulness'])
                if interaction.ragas_scores.get('answer_relevancy'):
                    relevancy_scores.append(interaction.ragas_scores['answer_relevancy'])
                if interaction.ragas_scores.get('context_precision'):
                    precision_scores.append(interaction.ragas_scores['context_precision'])

        def calculate_stats(scores):
            if not scores:
                return {'mean': 0, 'min': 0, 'max': 0, 'count': 0}
            return {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'count': len(scores)
            }

        return {
            'period': f"Últimos {days} dias",
            'total_interactions': len(interactions),
            'metrics': {
                'faithfulness': calculate_stats(faithfulness_scores),
                'answer_relevancy': calculate_stats(relevancy_scores),
                'context_precision': calculate_stats(precision_scores),
            },
            'overall_quality': {
                'average_score': (
                    calculate_stats(faithfulness_scores)['mean'] +
                    calculate_stats(relevancy_scores)['mean'] +
                    calculate_stats(precision_scores)['mean']
                ) / 3 if faithfulness_scores and relevancy_scores and precision_scores else 0
            }
        }

    def _log_ragas_evaluation_to_phoenix(
        self, 
        evaluation_results: Dict[str, Any], 
        interactions: List[RAGInteractionDB]
    ):
        """Registra resultados da avaliação RAGAS no Phoenix"""
        try:
            # Log da avaliação completa no Phoenix
            phoenix_metadata = {
                "evaluation_type": "RAGAS",
                "total_interactions": evaluation_results.get('total_interactions', 0),
                "average_scores": evaluation_results.get('average_scores', {}),
                "timestamp": str(asyncio.get_event_loop().time()),
                "interactions_evaluated": [i.id for i in interactions]
            }
            
            # Phoenix captura automaticamente através da instrumentação
            # Este log adicional pode ser usado para análises customizadas
            print(f"🔥 RAGAS evaluation logged to Phoenix: {evaluation_results['total_interactions']} interactions")
            
        except Exception as e:
            print(f"Erro ao logar avaliação RAGAS no Phoenix: {str(e)}")

    async def generate_phoenix_ragas_report(self) -> Dict[str, Any]:
        """Gera relatório combinando dados do Phoenix com métricas RAGAS"""
        try:
            # Obter dados de traces do Phoenix
            phoenix_traces = phoenix_service.get_traces_data(limit=100)
            
            # Obter relatório de qualidade RAGAS
            ragas_report = await self.get_quality_report(days=30)
            
            # Combinar dados
            combined_report = {
                "report_type": "Phoenix + RAGAS Combined Analysis",
                "phoenix_data": {
                    "enabled": phoenix_service.is_enabled,
                    "dashboard_url": phoenix_service.get_phoenix_url(),
                    "traces_info": phoenix_traces
                },
                "ragas_metrics": ragas_report,
                "integration_status": {
                    "phoenix_active": phoenix_service.is_enabled,
                    "ragas_data_available": "error" not in ragas_report,
                    "combined_analysis": "available" if phoenix_service.is_enabled and "error" not in ragas_report else "limited"
                },
                "recommendations": self._generate_recommendations(ragas_report, phoenix_traces)
            }
            
            return combined_report
            
        except Exception as e:
            return {
                "error": f"Erro ao gerar relatório combinado: {str(e)}",
                "phoenix_status": phoenix_service.is_enabled,
                "fallback": "Use relatórios individuais do Phoenix e RAGAS"
            }

    def _generate_recommendations(
        self, 
        ragas_report: Dict[str, Any], 
        phoenix_traces: Dict[str, Any]
    ) -> List[str]:
        """Gera recomendações baseadas nos dados combinados"""
        recommendations = []
        
        try:
            # Análise baseada em métricas RAGAS
            if "metrics" in ragas_report:
                faithfulness = ragas_report["metrics"].get("faithfulness", {})
                if faithfulness.get("mean", 0) < 0.7:
                    recommendations.append("🔍 Considere melhorar a qualidade dos documentos de contexto")
                
                relevancy = ragas_report["metrics"].get("answer_relevancy", {})
                if relevancy.get("mean", 0) < 0.8:
                    recommendations.append("🎯 Revise o prompt do LLM para respostas mais relevantes")
                
                precision = ragas_report["metrics"].get("context_precision", {})
                if precision.get("mean", 0) < 0.6:
                    recommendations.append("📊 Ajuste os parâmetros de busca vetorial")
            
            # Análise baseada no Phoenix
            if phoenix_service.is_enabled:
                recommendations.append("📈 Use o dashboard Phoenix para análise detalhada de traces")
                recommendations.append("🔬 Analise clusters de embeddings para identificar padrões")
            
            if not recommendations:
                recommendations.append("✅ Sistema funcionando bem! Continue monitorando as métricas")
            
        except Exception as e:
            recommendations.append(f"⚠️ Erro ao gerar recomendações: {str(e)}")
        
        return recommendations

# Instância global do serviço RAGAS
ragas_service = RAGASService()