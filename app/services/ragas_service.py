import os
import asyncio
import nest_asyncio

#REMOVER, SE PARAR DE DAR CONFLITO NO DOCKER COM evaluate
# IMPORTANTE: Configurar asyncio ANTES de qualquer outra importação
# Isso previne conflitos de event loop com uvloop no Docker
os.environ["UVLOOP_DISABLED"] = "1"
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
nest_asyncio.apply()

from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from sqlalchemy import select
from app.models.rag_interaction import RAGInteractionDB
from app.services.database_service import AsyncSessionLocal
from app.services.phoenix_service import phoenix_service
from app.core.config import settings
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        self._setup_openai_config()
        
        self.metrics = [
            faithfulness, 
            answer_relevancy,
            #context_precision, # PRECISÃO: Os contextos mais importantes apareceram primeiro?
            #context_recall   # RECALL: Todos os contextos relevantes foram recuperados?
        ]
    
    def _setup_openai_config(self):
        openai.api_key = settings.openai_api_key
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

      
    async def evaluate_interactions(
        self,
        interaction_ids: List[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        FUNÇÃO PRINCIPAL: Avalia a qualidade das conversas RAG usando inteligência artificial
        
        COMO FUNCIONA O PROCESSO (8 etapas):
        1.Busca conversas salvas no banco de dados
        2.Converte os dados para um formato que o RAGAS entende
        3.Cria um dataset padronizado para análise
        4.Usa GPT para avaliar cada conversa em várias métricas
        5.Calcula médias e organiza os resultados
        6.Extrai notas individuais de cada conversa
        7.Salva todas as notas no banco para histórico
        8.Registra no Phoenix (sistema de monitoramento)
        
        PARÂMETROS:
            interaction_ids: Se você quiser avaliar conversas específicas, passe os IDs aqui
                           Se não passar nada, vai avaliar as conversas mais recentes
            limit: Quantas conversas avaliar no máximo (padrão: 100)
        
        """
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


        data = []
        for interaction in interactions:
            data.append({
                'question': interaction.question,
                'answer': interaction.answer,
                'contexts': interaction.contexts,
            })

        dataset = Dataset.from_pandas(pd.DataFrame(data))

        try:
            print("RAGAS:LENDO E AVALIANDO INTERAÇÕES COM IA...")
            result = evaluate(dataset, metrics=self.metrics)
            
            if not result:
                raise ValueError("RAGAS retornou resultado vazio")
            
            faithfulness_scores = []
            answer_relevancy_scores = []
            
            try:
                if hasattr(result, 'to_pandas'):
                    df = result.to_pandas()
                    if 'faithfulness' in df.columns:
                        faithfulness_scores = df['faithfulness'].tolist()

                    else:
                        print("RAGAS: Coluna 'faithfulness' não encontrada")
                        
                    if 'answer_relevancy' in df.columns:
                        answer_relevancy_scores = df['answer_relevancy'].tolist()

                    else:
                        print("RAGAS: Coluna 'answer_relevancy' não encontrada")
                        
                elif hasattr(result, 'scores'):
                    scores = result.scores
                    if 'faithfulness' in scores:
                        faithfulness_scores = scores['faithfulness']
                    if 'answer_relevancy' in scores:
                        answer_relevancy_scores = scores['answer_relevancy']
                
                else:
                    print("RAGAS: Tentando métodos diretos do EvaluationResult")
                    for attr in dir(result):
                        if attr == 'faithfulness':
                            faithfulness_scores = getattr(result, attr, [])
                        elif attr == 'answer_relevancy':
                            answer_relevancy_scores = getattr(result, attr, [])
                            
            except Exception as e:
                print(f"AVISO RAGAS: Erro ao extrair scores: {str(e)}")

            faithfulness_avg = 0
            if faithfulness_scores and len(faithfulness_scores) > 0:
                valid_faithfulness = [score for score in faithfulness_scores if score is not None and isinstance(score, (int, float))]
                faithfulness_avg = sum(valid_faithfulness) / len(valid_faithfulness) if valid_faithfulness else 0
                print(f"RAGAS: Média de fidelidade: {faithfulness_avg:.3f} (baseado em {len(valid_faithfulness)} scores válidos)")
            
            relevancy_avg = 0  
            if answer_relevancy_scores and len(answer_relevancy_scores) > 0:
                valid_relevancy = [score for score in answer_relevancy_scores if score is not None and isinstance(score, (int, float))]
                relevancy_avg = sum(valid_relevancy) / len(valid_relevancy) if valid_relevancy else 0
                print(f"RAGAS: Média de relevância: {relevancy_avg:.3f} (baseado em {len(valid_relevancy)} scores válidos)")
            
            evaluation_results = {
                'total_interactions': len(interactions),
                'average_scores': {
                    'faithfulness': round(faithfulness_avg, 3),
                    'answer_relevancy': round(relevancy_avg, 3),
                },
                'individual_scores': []
            }
            
            for i, interaction in enumerate(interactions):
                try:
                    faithfulness_score = None
                    if i < len(faithfulness_scores):
                        score = faithfulness_scores[i]
                        if score is not None and isinstance(score, (int, float)):
                            faithfulness_score = round(float(score), 3)
                    
                    relevancy_score = None  
                    if i < len(answer_relevancy_scores):
                        score = answer_relevancy_scores[i]
                        if score is not None and isinstance(score, (int, float)):
                            relevancy_score = round(float(score), 3)
                    
                    individual_score = {
                        'interaction_id': interaction.id,
                        'question': interaction.question[:100] + "..." if len(interaction.question) > 100 else interaction.question,
                        'faithfulness': faithfulness_score,
                        'answer_relevancy': relevancy_score,
                    }
                    evaluation_results['individual_scores'].append(individual_score)
                    
                except Exception as e:
                    print(f"RAGAS: Erro ao processar score individual {i}: {str(e)}")
                    evaluation_results['individual_scores'].append({
                        'interaction_id': interaction.id,
                        'question': interaction.question[:100] + "..." if len(interaction.question) > 100 else interaction.question,
                        'faithfulness': None,
                        'answer_relevancy': None,
                    })
            
            print(f"RAGAS: {len(evaluation_results['individual_scores'])} scores individuais processados")

            await self._save_ragas_scores(interactions, evaluation_results['individual_scores'])

            advanced_metrics = await self._calculate_advanced_metrics(interactions, evaluation_results['individual_scores'])
            evaluation_results['advanced_metrics'] = advanced_metrics
            
            return evaluation_results

        except Exception as e:
            print(f"RAGAS: Erro durante avaliação: {str(e)}")
            print(f"RAGAS: Tipo do erro: {type(e).__name__}")
            error_message = str(e) if str(e) != "0" else "Erro desconhecido na avaliação RAGAS"
            
            return {
                'error': f"Erro durante avaliação RAGAS: {error_message}",
                'error_type': type(e).__name__,
                'total_interactions': len(interactions),
                'debug_info': {
                    'dataset_size': len(data) if 'data' in locals() else 0,
                    'metrics_count': len(self.metrics),
                    'openai_configured': bool(settings.openai_api_key)
                }
            }

    async def _save_ragas_scores(
        self, 
        interactions: List[RAGInteractionDB], 
        individual_scores: List[Dict]
    ):
        """
        Salva as notas RAGAS no banco de dados
        
        Esta função é importante para:
        - Acompanhar se a qualidade do sistema está melhorando ou piorando ao longo do tempo
        - Comparar como o sistema estava antes e depois de mudanças (novos prompts, modelos, etc)
        - Gerar relatórios históricos mostrando tendências
        - Identificar quais conversas tiveram problemas específicos
        
        COMO FUNCIONA:
        1. Pega cada conversa e sua nota correspondente
        2. Extrai só as métricas importantes (fidelidade e relevância)
        3. Salva essas notas no campo ragas_scores da conversa no banco
        4. Confirma a operação (commit)
        
        PARÂMETROS:
            interactions: Lista das conversas que foram avaliadas
            individual_scores: Lista das notas individuais de cada conversa
        """
        async with AsyncSessionLocal() as session:
            for interaction, scores in zip(interactions, individual_scores):
                ragas_scores = {
                    'faithfulness': scores.get('faithfulness'),
                    'answer_relevancy': scores.get('answer_relevancy'), 
                }
                
                interaction.ragas_scores = ragas_scores
                session.add(interaction)
            await session.commit()



    async def _calculate_advanced_metrics(
        self, 
        interactions: List[RAGInteractionDB], 
        individual_scores: List[Dict]
    ) -> Dict[str, Any]:
        """
        Calcula métricas avançadas não cobertas pelo RAGAS padrão:
        
        1. RECALL@3: Avalia se documentos relevantes estão nos top-3 resultados
        2. PRECISÃO PERCEBIDA: Baseada no feedback do usuário (ratings 1-5)
        """
        
        if not interactions:
            print("AVISO: Nenhuma interação fornecida para calcular métricas avançadas")
            return self._get_empty_advanced_metrics()
        
        # 1. RECALL@3 - Documentos relevantes nos top-3
        recall_at_3_scores = []

        for i, interaction in enumerate(interactions):
            try:
                if hasattr(interaction, 'sources') and interaction.sources:
                    print(f"Interação {i+1} - {len(interaction.sources)} sources disponíveis")
                    
                    sources_are_dicts = all(isinstance(source, dict) for source in interaction.sources)
                    
                    if sources_are_dicts:
                        top_3_sources = interaction.sources[:3]
                        relevant_in_top3 = sum(1 for source in top_3_sources if source.get('similarity_score', 0) > 0.3)
                        total_relevant = sum(1 for source in interaction.sources if source.get('similarity_score', 0) > 0.3)
                        
                        print(f"Interação {i+1} - Relevant in top3: {relevant_in_top3}, Total relevant: {total_relevant}")
                        
                        if total_relevant > 0:
                            recall_at_3 = relevant_in_top3 / total_relevant
                            recall_at_3_scores.append(recall_at_3)
                            print(f"Interação {i+1} - Recall@3: {recall_at_3:.3f}")
                        else:
                            recall_at_3_scores.append(0.0)
                            print(f"Interação {i+1} - Nenhum documento relevante")
                    else:
                        if len(interaction.sources) >= 3:
                            recall_at_3_scores.append(1.0)
                        else:
                            recall_at_3_scores.append(len(interaction.sources) / 3.0)
                        print(f"Interação {i+1} - Sources não são dicts, score: {recall_at_3_scores[-1]}")
                        
                elif hasattr(interaction, 'contexts') and interaction.contexts:
                    print(f"Interação {i+1} - Usando contexts como fallback")
                    if len(interaction.contexts) >= 3:
                        recall_at_3_scores.append(1.0)
                    else:
                        recall_at_3_scores.append(len(interaction.contexts) / 3.0)
                else:
                    recall_at_3_scores.append(0.0)
                    print(f"Interação {i+1} - Sem sources ou contexts")
                    
            except Exception as e:
                print(f"Erro ao calcular Recall@3 para interação {i+1}: {e}")
                recall_at_3_scores.append(0.0)
                continue
        
        avg_recall_at_3 = sum(recall_at_3_scores) / len(recall_at_3_scores) if recall_at_3_scores else 0
        
        # 2. PRECISÃO PERCEBIDA - Baseada em feedback do usuário
        user_feedback_scores = []
        for interaction in interactions:
            try:
                if hasattr(interaction, 'user_feedback') and interaction.user_feedback and isinstance(interaction.user_feedback, (int, float)):
                    feedback_value = float(interaction.user_feedback)
                    if 1 <= feedback_value <= 5:
                        perceived_precision = (feedback_value - 1) / 4
                        user_feedback_scores.append(perceived_precision)

                        #FEEDBACK-MIN
                        #------------------------
                        #MAX-MIN
            except Exception as e:
                print(f"AVISO: Erro ao processar feedback do usuário: {e}")
                continue
        
        avg_perceived_precision = sum(user_feedback_scores) / len(user_feedback_scores) if user_feedback_scores else None
        
        advanced_metrics = {
            "recall_at_3": {
                "value": round(avg_recall_at_3, 3),
                "description": "Proporção de documentos relevantes nos top-3 resultados",
                "interactions_analyzed": len(recall_at_3_scores),
                "individual_scores": [round(score, 3) for score in recall_at_3_scores] 
            },
            "perceived_precision": {
                "value": round(avg_perceived_precision, 3) if avg_perceived_precision is not None else None,
                "description": "Precisão baseada no feedback dos usuários (escala 0-1)",
                "interactions_analyzed": len(user_feedback_scores),

            }
        }
        
        print(f"Métricas avançadas calculadas:")
        print(f"   - Recall@3: {advanced_metrics['recall_at_3']['value']:.3f} ({len(recall_at_3_scores)} interações)")
        print(f"   - Precisão Percebida: {advanced_metrics['perceived_precision']['value'] or 'N/A'} ({len(user_feedback_scores)} feedbacks)")
        
        return advanced_metrics
    


    def _get_empty_advanced_metrics(self) -> Dict[str, Any]:
        """Retorna estrutura vazia de métricas avançadas quando não há dados suficientes"""
        return {
            "recall_at_3": {
                "value": 0.0,
                "description": "Proporção de documentos relevantes nos top-3 resultados",
                "interactions_analyzed": 0,
                "individual_scores": []
            },
            "perceived_precision": {
                "value": None,
                "description": "Precisão baseada no feedback dos usuários (escala 0-1)",
                "interactions_analyzed": 0,
                "feedback_distribution": None
            }
        }


ragas_service = RAGASService()