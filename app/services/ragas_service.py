# RAGAS SERVICE - Serviço de Avaliação de Qualidade RAG
# RAGAS (RAG Assessment) usa LLMs para avaliar se as respostas RAG são boas
# É como um "professor" que dá notas para o sistema RAG

import os
import asyncio
import nest_asyncio  # Para resolver conflitos de event loop

# IMPORTANTE: Configurar asyncio ANTES de qualquer outra importação
# Isso previne conflitos de event loop com uvloop no Docker
os.environ["UVLOOP_DISABLED"] = "1"
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# CONFIGURAR NEST_ASYNCIO para evitar conflitos com uvloop
# Isso permite que RAGAS funcione corretamente com FastAPI
nest_asyncio.apply()

from typing import List, Dict, Any
import pandas as pd  # Manipulação de dados
from datasets import Dataset  # Formato que RAGAS entende
from ragas import evaluate  # Função principal de avaliação
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall  # Métricas de qualidade
from sqlalchemy import select  # Consultas ao banco de dados
from app.models.rag_interaction import RAGInteractionDB  # Modelo das interações salvas
from app.services.database_service import AsyncSessionLocal  # Conexão com banco
from app.services.phoenix_service import phoenix_service  # Integração com observabilidade

class RAGASService:
    """
    Serviço de avaliação de qualidade RAG usando RAGAS
    
    RAGAS funciona como um "professor" que avalia se o sistema RAG está funcionando bem:
    1. Pega interações salvas (pergunta, resposta, contextos)
    2. Usa LLM (GPT) para avaliar cada métrica
    3. Dá notas de 0 a 1 (quanto maior, melhor)
    4. Salva as notas no banco para acompanhar evolução
    """
    
    def __init__(self):
        # MÉTRICAS DO RAGAS - O que vamos medir:
        self.metrics = [
            faithfulness,      # FIDELIDADE: A resposta está baseada apenas no contexto? (previne alucinações)
            answer_relevancy,  # RELEVÂNCIA: A resposta realmente responde a pergunta?
            # context_precision, # PRECISÃO: Os contextos mais importantes apareceram primeiro? (DESATIVADO - precisa ground_truth)
            # context_recall   # RECALL: Todos os contextos relevantes foram recuperados? (DESATIVADO - precisa ground_truth)
        ]

    async def evaluate_interactions(
        self, 
        interaction_ids: List[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        FUNÇÃO PRINCIPAL: Avalia um conjunto de interações usando RAGAS
        
        PROCESSO RAGAS (passo a passo):
        1. Busca interações no banco (pergunta, resposta, contextos)
        2. Converte para formato que RAGAS entende
        3. Usa LLM (GPT) para avaliar cada métrica
        4. Retorna scores de 0 a 1 (quanto maior, melhor)
        5. Salva resultados no banco para histórico
        
        Args:
            interaction_ids: Lista de IDs específicos para avaliar
            limit: Número máximo de interações para avaliar (se interaction_ids não fornecido)
            
        Returns:
            Dict com scores médios e individuais
        """
        
        # ETAPA 1: BUSCAR INTERAÇÕES NO BANCO DE DADOS
        # Pega as conversas salvas que queremos avaliar
        async with AsyncSessionLocal() as session:
            if interaction_ids:
                # Se passou IDs específicos, busca só esses
                query = select(RAGInteractionDB).where(
                    RAGInteractionDB.id.in_(interaction_ids)
                )
            else:
                # Senão, pega as últimas X interações
                query = select(RAGInteractionDB).limit(limit)
            
            result = await session.execute(query)
            interactions = result.scalars().all()  # Lista de interações do banco

        if not interactions:
            return {"error": "Nenhuma interação encontrada para avaliar"}

        # ETAPA 2: PREPARAR DADOS PARA O RAGAS
        # RAGAS precisa dos dados num formato específico
        data = []
        for interaction in interactions:
            data.append({
                'question': interaction.question,    # Pergunta do usuário
                'answer': interaction.answer,        # Resposta que o sistema deu
                'contexts': interaction.contexts,    # Documentos que foram usados como contexto
            })

        # ETAPA 3: CONVERTER PARA DATASET DO RAGAS
        # RAGAS trabalha com objetos Dataset, não listas Python normais
        dataset = Dataset.from_pandas(pd.DataFrame(data))

        try:
            # ETAPA 4: EXECUTAR AVALIAÇÃO RAGAS
            # Aqui é onde a mágica acontece: LLM avalia cada interação
            result = evaluate(dataset, metrics=self.metrics)
            
            # ETAPA 5: PROCESSAR RESULTADOS
            # Organiza os dados de forma mais friendly
            evaluation_results = {
                'total_interactions': len(interactions),
                'average_scores': {
                    'faithfulness': float(result['faithfulness']),        # Média de fidelidade (0-1)
                    'answer_relevancy': float(result['answer_relevancy']), # Média de relevância (0-1)
                },
                'individual_scores': []  # Scores individuais de cada interação
            }

            # ETAPA 6: EXTRAIR SCORES INDIVIDUAIS
            # Para cada interação, pega sua nota individual
            for i, interaction in enumerate(interactions):
                individual_score = {
                    'interaction_id': interaction.id,
                    # Preview da pergunta (trunca se muito longa)
                    'question': interaction.question[:100] + "..." if len(interaction.question) > 100 else interaction.question,
                    # Scores individuais para cada métrica (0-1, quanto maior melhor)
                    'faithfulness': float(result['faithfulness'][i]) if i < len(result['faithfulness']) else None,
                    'answer_relevancy': float(result['answer_relevancy'][i]) if i < len(result['answer_relevancy']) else None,
                }
                evaluation_results['individual_scores'].append(individual_score)

            # ETAPA 7: SALVAR SCORES NO BANCO DE DADOS
            # Salva os resultados para acompanhar evolução ao longo do tempo
            await self._save_ragas_scores(interactions, evaluation_results['individual_scores'])

            # ETAPA 8: INTEGRAÇÃO COM PHOENIX
            # Se Phoenix estiver ativo, registra a avaliação para observabilidade
            if phoenix_service.is_enabled:
                self._log_ragas_evaluation_to_phoenix(evaluation_results, interactions)

            return evaluation_results

        except Exception as e:
            # Se algo der errado (falta de API key, erro de rede, etc.)
            return {
                'error': f"Erro durante avaliação RAGAS: {str(e)}",
                'total_interactions': len(interactions)
            }

    async def _save_ragas_scores(
        self, 
        interactions: List[RAGInteractionDB], 
        individual_scores: List[Dict]
    ):
        """Salva os scores RAGAS no banco de dados para histórico
        
        Isso permite:
        - Acompanhar evolução da qualidade ao longo do tempo
        - Comparar performance antes/depois de mudanças
        - Gerar relatórios históricos
        """
        async with AsyncSessionLocal() as session:
            for interaction, scores in zip(interactions, individual_scores):
                # Extrai apenas as métricas, sem metadados extras
                ragas_scores = {
                    'faithfulness': scores.get('faithfulness'),      # Fidelidade (0-1)
                    'answer_relevancy': scores.get('answer_relevancy'), # Relevância (0-1)
                }
                
                # Atualiza o registro da interação com os scores
                interaction.ragas_scores = ragas_scores
                session.add(interaction)
            
            # Salva tudo no banco
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

        for interaction in interactions:
            if interaction.ragas_scores:
                if interaction.ragas_scores.get('faithfulness'):
                    faithfulness_scores.append(interaction.ragas_scores['faithfulness'])
                if interaction.ragas_scores.get('answer_relevancy'):
                    relevancy_scores.append(interaction.ragas_scores['answer_relevancy'])

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
            },
            'overall_quality': {
                'average_score': (
                    calculate_stats(faithfulness_scores)['mean'] +
                    calculate_stats(relevancy_scores)['mean']
                ) / 2 if faithfulness_scores and relevancy_scores else 0
            }
        }

    def _log_ragas_evaluation_to_phoenix(
        self, 
        evaluation_results: Dict[str, Any], 
        interactions: List[RAGInteractionDB]
    ):
        """Registra resultados da avaliação RAGAS no Phoenix para observabilidade
        
        Isso permite ver no dashboard Phoenix:
        - Quando avaliações RAGAS foram executadas
        - Quais foram os scores médios
        - Trends de qualidade ao longo do tempo
        """
        try:
            # Prepara metadados para Phoenix
            phoenix_metadata = {
                "evaluation_type": "RAGAS",  # Tipo de avaliação
                "total_interactions": evaluation_results.get('total_interactions', 0),  # Quantas foram avaliadas
                "average_scores": evaluation_results.get('average_scores', {}),  # Scores médios
                "timestamp": str(asyncio.get_event_loop().time()),  # Quando aconteceu
                "interactions_evaluated": [i.id for i in interactions]  # Quais interações
            }
            
            # Phoenix captura automaticamente através da instrumentação
            # Este log adicional aparece no dashboard para análises customizadas
            print(f"🔥 RAGAS evaluation logged to Phoenix: {evaluation_results['total_interactions']} interactions")
            print(f"📊 Scores médios: {evaluation_results.get('average_scores', {})}")
            
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
        """Gera recomendações baseadas nos dados combinados RAGAS + Phoenix
        
        Funciona como um "consultor automático" que analisa as métricas
        e sugere melhorias específicas para o sistema RAG
        """
        recommendations = []
        
        try:
            # ANÁLISE BASEADA EM MÉTRICAS RAGAS
            if "metrics" in ragas_report:
                # Verifica FIDELIDADE (faithfulness)
                faithfulness = ragas_report["metrics"].get("faithfulness", {})
                if faithfulness.get("mean", 0) < 0.7:
                    recommendations.append("🔍 FIDELIDADE BAIXA: Melhore a qualidade dos documentos de contexto")
                
                # Verifica RELEVÂNCIA (answer_relevancy) 
                relevancy = ragas_report["metrics"].get("answer_relevancy", {})
                if relevancy.get("mean", 0) < 0.8:
                    recommendations.append("🎯 RELEVÂNCIA BAIXA: Revise o prompt do LLM para respostas mais relevantes")
                
                # Verifica PRECISÃO (context_precision)
                # precision = ragas_report["metrics"].get("context_precision", {})
                # if precision.get("mean", 0) < 0.6:
                #     recommendations.append("📊 PRECISÃO BAIXA: Ajuste os parâmetros de busca vetorial")
            
            # ANÁLISE BASEADA NO PHOENIX
            if phoenix_service.is_enabled:
                recommendations.append("📈 Use o dashboard Phoenix para análise detalhada de traces")
                recommendations.append("🔬 Analise clusters de embeddings para identificar padrões")
            
            # Se está tudo bem
            if not recommendations:
                recommendations.append("✅ Sistema funcionando bem! Continue monitorando as métricas")
            
        except Exception as e:
            recommendations.append(f"⚠️ Erro ao gerar recomendações: {str(e)}")
        
        return recommendations

# Instância global do serviço RAGAS
ragas_service = RAGASService()