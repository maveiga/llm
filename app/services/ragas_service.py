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
import pandas as pd  # Biblioteca para manipulação de dados em tabelas
from datasets import Dataset  # Formato de dados que a biblioteca RAGAS consegue entender
from ragas import evaluate  # Função principal que executa todas as avaliações RAGAS
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall  # Métricas que medem qualidade RAG
from sqlalchemy import select  # Comandos para buscar dados no banco de dados
from app.models.rag_interaction import RAGInteractionDB  # Modelo que representa as conversas salvas no banco
from app.services.database_service import AsyncSessionLocal  # Serviço para conectar com o banco de dados
from app.services.phoenix_service import phoenix_service  # Serviço de observabilidade e monitoramento
from app.core.config import settings  # Configurações gerais da aplicação (API keys, etc)
import openai  # Biblioteca oficial da OpenAI para usar GPT

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
        # CONFIGURAR OPENAI API KEY PARA RAGAS
        self._setup_openai_config()
        
        # MÉTRICAS DO RAGAS - O que vamos medir:
        self.metrics = [
            faithfulness,      # FIDELIDADE: A resposta está baseada apenas no contexto? (previne alucinações)
            answer_relevancy,  # RELEVÂNCIA: A resposta realmente responde a pergunta?
            # context_precision, # PRECISÃO: Os contextos mais importantes apareceram primeiro? (DESATIVADO - precisa ground_truth)
            # context_recall   # RECALL: Todos os contextos relevantes foram recuperados? (DESATIVADO - precisa ground_truth)
            # usar quando tiver a resposta correta no banco
        ]
    
    def _setup_openai_config(self):

        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
            print("Chave da OpenAI configurada com sucesso para o RAGAS")
        else:
            print("Chave da OpenAI não encontrada - RAGAS não conseguirá funcionar")

    async def evaluate_interactions(
        self,
        interaction_ids: List[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        FUNÇÃO PRINCIPAL: Avalia a qualidade das conversas RAG usando inteligência artificial
        
        Esta é a função mais importante do serviço. Ela funciona como um "professor virtual"
        que analisa as conversas entre usuários e o sistema RAG, dando notas de qualidade.
        
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
            
        RETORNA:
            Um dicionário com:
            - total_interactions: quantas conversas foram avaliadas
            - average_scores: nota média geral do sistema
            - individual_scores: nota individual de cada conversa
            
        EXEMPLO DE USO:
            # Avaliar as 50 conversas mais recentes
            resultado = await ragas_service.evaluate_interactions(limit=50)
            
            # Avaliar conversas específicas
            resultado = await ragas_service.evaluate_interactions(['id1', 'id2', 'id3'])
        """
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

        # ETAPA 2:  PREPARAR DADOS NO FORMATO RAGAS
        data = []
        for interaction in interactions:
            data.append({
                'question': interaction.question,    # A pergunta que o usuário fez
                'answer': interaction.answer,        # A resposta que nosso sistema deu
                'contexts': interaction.contexts,    # Os documentos que usamos como base
            })

        # ETAPA 3:CRIAR DATASET PADRONIZADO

        dataset = Dataset.from_pandas(pd.DataFrame(data))

        try:
            # ETAPA 4: EXECUTAR AVALIAÇÃO COM INTELIGÊNCIA ARTIFICIAL O RAGAS usa modelos de IA para "ler" e "entender" se as respostas são boas
            print(f"🔄 RAGAS: Executando avaliação com {len(self.metrics)} métricas...")
            print(f"🔄 RAGAS: Dataset shape: {dataset.shape if hasattr(dataset, 'shape') else 'N/A'}")
            
            result = evaluate(dataset, metrics=self.metrics)
            
            print(f"✅ RAGAS: Avaliação concluída. Tipo do resultado: {type(result)}")
            print(f"📋 RAGAS: Atributos do resultado: {[attr for attr in dir(result) if not attr.startswith('_')][:10]}")
            
            # ETAPA 5: PROCESSAR RESULTADOS - COMPATÍVEL COM EvaluationResult
            if not result:
                raise ValueError("RAGAS retornou resultado vazio")
            
            # Extrair scores do EvaluationResult
            faithfulness_scores = []
            answer_relevancy_scores = []
            
            try:
                # Converter para DataFrame se possível
                if hasattr(result, 'to_pandas'):
                    df = result.to_pandas()
                    print(f"📊 RAGAS: DataFrame criado com shape: {df.shape}")
                    print(f"📊 RAGAS: Colunas disponíveis: {list(df.columns)}")
                    
                    # Extrair scores das colunas
                    if 'faithfulness' in df.columns:
                        faithfulness_scores = df['faithfulness'].tolist()
                        print(f"📊 RAGAS: Faithfulness scores extraídos: {len(faithfulness_scores)} valores")
                        print(f"📊 RAGAS: Primeiro faithfulness score (tipo): {type(faithfulness_scores[0]) if faithfulness_scores else 'Lista vazia'}")
                    else:
                        print("RAGAS: Coluna 'faithfulness' não encontrada")
                        
                    if 'answer_relevancy' in df.columns:
                        answer_relevancy_scores = df['answer_relevancy'].tolist()
                        print(f"📊 RAGAS: Answer relevancy scores extraídos: {len(answer_relevancy_scores)} valores")
                        print(f"📊 RAGAS: Primeiro answer_relevancy score (tipo): {type(answer_relevancy_scores[0]) if answer_relevancy_scores else 'Lista vazia'}")
                    else:
                        print("RAGAS: Coluna 'answer_relevancy' não encontrada")
                        
                # Método alternativo: tentar acessar diretamente os atributos
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
                print(f"⚠️  RAGAS: Erro ao extrair scores: {str(e)}")

            # Calcular médias de forma mais segura

            faithfulness_avg = 0
            if faithfulness_scores and len(faithfulness_scores) > 0:
                valid_faithfulness = [score for score in faithfulness_scores if score is not None and isinstance(score, (int, float))]
                faithfulness_avg = sum(valid_faithfulness) / len(valid_faithfulness) if valid_faithfulness else 0
                print(f"📊 RAGAS: Média de fidelidade: {faithfulness_avg:.3f} (baseado em {len(valid_faithfulness)} scores válidos)")
            
            relevancy_avg = 0  
            if answer_relevancy_scores and len(answer_relevancy_scores) > 0:
                valid_relevancy = [score for score in answer_relevancy_scores if score is not None and isinstance(score, (int, float))]
                relevancy_avg = sum(valid_relevancy) / len(valid_relevancy) if valid_relevancy else 0
                print(f"📊 RAGAS: Média de relevância: {relevancy_avg:.3f} (baseado em {len(valid_relevancy)} scores válidos)")
            
            evaluation_results = {
                'total_interactions': len(interactions),
                'average_scores': {
                    'faithfulness': round(faithfulness_avg, 3),
                    'answer_relevancy': round(relevancy_avg, 3),
                },
                'individual_scores': []
            }

            # ETAPA 6: EXTRAIR SCORES INDIVIDUAIS - VERSÃO SIMPLIFICADA
            
            for i, interaction in enumerate(interactions):
                try:
                    # Scores individuais com verificação de bounds
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
                    print(f" RAGAS: Erro ao processar score individual {i}: {str(e)}")
                    evaluation_results['individual_scores'].append({
                        'interaction_id': interaction.id,
                        'question': interaction.question[:100] + "..." if len(interaction.question) > 100 else interaction.question,
                        'faithfulness': None,
                        'answer_relevancy': None,
                    })
            
            print(f"RAGAS: {len(evaluation_results['individual_scores'])} scores individuais processados")

            # ETAPA 7: SALVAR SCORES NO BANCO DE DADOS
            # Salva os resultados para acompanhar evolução ao longo do tempo
            await self._save_ragas_scores(interactions, evaluation_results['individual_scores'])

            # ETAPA 8: PHOENIX REMOVIDO
            # Phoenix captura dados automaticamente via instrumentação OpenTelemetry

            return evaluation_results

        except Exception as e:
            # TRATAMENTO DE ERRO MELHORADO
            print(f"RAGAS: Erro durante avaliação: {str(e)}")
            print(f"RAGAS: Tipo do erro: {type(e).__name__}")
            
            # Log adicional para debug
            if hasattr(e, '__traceback__'):
                import traceback
                print(f"RAGAS: Traceback: {traceback.format_exc()}")
            
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
        Salva as notas RAGAS no banco de dados para criar um histórico
        
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
                # Cria um dicionário só com as notas principais (sem outros dados desnecessários)
                ragas_scores = {
                    'faithfulness': scores.get('faithfulness'),         # Nota de fidelidade (0 a 1, quanto maior melhor)
                    'answer_relevancy': scores.get('answer_relevancy'), # Nota de relevância (0 a 1, quanto maior melhor)
                }
                
                interaction.ragas_scores = ragas_scores
                session.add(interaction)
            await session.commit()

    async def get_quality_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Gera um relatório completo da qualidade do sistema RAG
        
        Esta função analisa todas as conversas que já foram avaliadas pelo RAGAS
        e cria um relatório executivo com estatísticas úteis para entender
        como o sistema está performando.
        
        O QUE O RELATÓRIO MOSTRA:
        - Quantas conversas foram analisadas no período
        - Nota média geral do sistema (fidelidade + relevância)
        - Melhor e pior nota registrada
        - Score geral de qualidade (média das duas métricas)
        
        PARÂMETROS:
            days: Quantos dias para trás incluir no relatório (padrão: 30 dias)
            
        RETORNA:
            Dicionário com estatísticas completas ou mensagem de erro se não houver dados
            
        EXEMPLO DE USO:
            # Relatório dos últimos 7 dias
            relatorio = await ragas_service.get_quality_report(days=7)
            
            # Relatório do último mês (padrão)
            relatorio = await ragas_service.get_quality_report()
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



# Instância global do serviço RAGAS
ragas_service = RAGASService()