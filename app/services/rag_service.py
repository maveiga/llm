# RAG SERVICE - Orquestrador Principal do Sistema RAG
# Este é o "maestro" que coordena todo o pipeline RAG:
# 1. Busca documentos relevantes (VectorService)
# 2. Gera resposta com LLM (LLMService)
# 3. Salva interação no banco (DatabaseService)
# 4. Monitora com Phoenix (PhoenixService)
# 5. Disponibiliza dados para RAGAS avaliar depois

from typing import List, Dict, Any, Optional
import time  # Medir tempo de resposta
import uuid  # Gerar IDs únicos para interações
from app.services.vector_service import VectorService  # Busca vetorial de documentos
from app.services.llm_service import LLMService  # Geração de respostas com GPT
from app.models.document import DocumentResponse  # Modelo de documento
from app.models.rag_interaction import RAGInteractionDB, RAGInteractionCreate  # Modelos de interação
from app.services.database_service import AsyncSessionLocal  # Conexão com banco de dados
from app.services.phoenix_service import phoenix_service  # Observabilidade
import logging

class RAGService:
    """
    SERVIÇO PRINCIPAL DO SISTEMA RAG
    
    Este é o "maestro" que orquestra todo o pipeline RAG:
    
    PIPELINE COMPLETO:
    Pergunta do Usuário
           ↓
    1. VectorService: Busca documentos similares
           ↓
    2. LLMService: Gera resposta baseada nos documentos
           ↓
    3. DatabaseService: Salva interação completa
           ↓
    4. PhoenixService: Monitora para observabilidade
           ↓
    5. RAGAS: Pode avaliar qualidade depois
           ↓
    Resposta + Fontes para o Usuário
    
    INTEGRAÇÕES:
    - LangChain: Via LLMService para geração de resposta
    - Phoenix: Monitoramento automático de todo o pipeline
    - RAGAS: Usa dados salvos para avaliar qualidade
    """
    
    def __init__(self):
        # Inicializa os serviços que compõem o pipeline RAG
        self.vector_service = VectorService()  # Busca vetorial (ChromaDB + embeddings)
        self.llm_service = LLMService()        # Geração de resposta (LangChain + OpenAI)
    
    async def ask_question(
        self, 
        question: str, 
        max_documents: int = 5,
        category_filter: Optional[str] = None,
        save_interaction: bool = True
    ) -> Dict[str, Any]:
        """
        FUNÇÃO PRINCIPAL: Pipeline RAG Completo
        
        FLUXO PASSO A PASSO:
        1. 🔍 BUSCA: Vector search encontra documentos similares
        2. 🤖 GERAÇÃO: LLM cria resposta baseada nos documentos
        3. 💾 PERSISTÊNCIA: Salva interação no banco (para RAGAS avaliar)
        4. 🔥 OBSERVABILIDADE: Phoenix monitora toda a operação
        5. 📤 RESPOSTA: Retorna resposta + fontes para o usuário
        
        INTEGRAÇÕES ATIVAS:
        - LangChain: Geração de resposta estruturada
        - Phoenix: Monitoramento automático via instrumentação
        - RAGAS: Dados salvos ficam disponíveis para avaliação
        
        Args:
            question: Pergunta do usuário
            max_documents: Máximo de documentos para usar como contexto (default=5)
            category_filter: Filtro opcional por categoria de documento
            save_interaction: Se deve salvar a interação no banco (default=True para RAGAS)
            
        Returns:
            Dict com resposta gerada, fontes citadas, metadados e tempo de resposta
        """
        # ⏱️ CRONOMETRO: Medir tempo total de resposta
        start_time = time.time()
        
        # 🔍 ETAPA 1: BUSCAR DOCUMENTOS RELEVANTES
        # Vector search: converte pergunta em embedding e busca documentos similares
        print(f"🔍 Buscando documentos relevantes para: '{question[:50]}...'")
        relevant_docs = await self.vector_service.search_documents(
            query=question,                # Pergunta vira embedding
            limit=max_documents,           # Top-K documentos mais similares
            category_filter=category_filter # Filtro opcional por categoria
        )
        print(f"📝 Encontrados {len(relevant_docs)} documentos relevantes")
        
        # ⚠️ CASO ESPECIAL: NÃO ENCONTROU DOCUMENTOS RELEVANTES
        if not relevant_docs:
            print("⚠️ Nenhum documento relevante encontrado")
            response = {
                "answer": "Desculpe, não encontrei documentos relevantes para responder sua pergunta.",
                "sources": [],
                "context_used": 0,
                "question": question,
                "has_context": False
            }
            
            # Salvar interação mesmo sem contexto (importante para métricas)
            if save_interaction:
                await self._save_interaction(
                    question=question,
                    answer=response["answer"],
                    contexts=[],             # Lista vazia
                    sources=[],              # Sem fontes
                    response_time=time.time() - start_time
                )
            
            return response
        
        # 🔄 ETAPA 2: CONVERTER DOCUMENTOS PARA FORMATO DO LLM
        # Prepara dados em dois formatos:
        # - context_documents: formato rico para LLM (com metadados)
        # - contexts_for_db: apenas texto para salvar no banco (RAGAS usará)
        context_documents = []  # Para LLMService
        contexts_for_db = []    # Para banco de dados
        
        print(f"🔄 Preparando {len(relevant_docs)} documentos para o LLM:")
        for doc in relevant_docs:
            # Formato rico para o LLM
            context_documents.append({
                "title": doc.title,
                "category": doc.category,
                "content": doc.content,
                "similarity_score": doc.similarity_score,  # Quão similar à pergunta (0-1)
                "metadata": doc.metadata
            })
            # Apenas o conteúdo para salvar no banco (RAGAS precisa só do texto)
            contexts_for_db.append(doc.content)
            print(f"   • {doc.title} (similaridade: {doc.similarity_score:.3f})")
        
        # 🤖 ETAPA 3: GERAR RESPOSTA COM LLM (LANGCHAIN + OPENAI)
        # Aqui é onde a "mágica" acontece: GPT lê os documentos e gera a resposta
        print(f"🤖 Gerando resposta com LLM...")
        llm_response = await self.llm_service.generate_answer(
            question=question,              # Pergunta original
            context_documents=context_documents  # Documentos como contexto
        )
        print(f"✅ Resposta gerada: '{llm_response.get('answer', '')[:100]}...'")
        
        # 🎆 ETAPA 4: ENRIQUECER RESPOSTA COM INFORMAÇÕES ADICIONAIS
        # Prepara informações extras para mostrar ao usuário
        search_results = [
            {
                "title": doc.title,
                "category": doc.category,
                "similarity_score": doc.similarity_score,
                # Preview do conteúdo (primeiros 200 caracteres)
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            }
            for doc in relevant_docs
        ]
        
        # Combina resposta do LLM com informações extras
        response = {
            **llm_response,                    # Resposta + fontes do LLM
            "has_context": True,              # Flag indicando que tem contexto
            "search_results": search_results  # Detalhes dos documentos encontrados
        }
        
        # 💾 ETAPA 5: SALVAR INTERAÇÃO NO BANCO DE DADOS
        # Essencial para RAGAS avaliar qualidade depois
        response_time = time.time() - start_time
        print(f"⏱️ Tempo total de resposta: {response_time:.2f}s")
        
        if save_interaction:
            print(f"💾 Salvando interação no banco de dados...")
            interaction_id = await self._save_interaction(
                question=question,                      # Pergunta original
                answer=response["answer"],              # Resposta gerada
                contexts=contexts_for_db,               # Textos usados como contexto
                sources=response.get("sources", []),    # Fontes citadas
                response_time=response_time             # Performance
            )
            response["interaction_id"] = interaction_id
            print(f"✅ Interação salva com ID: {interaction_id}")
            
            # 🔥 ETAPA 6: LOG NO PHOENIX PARA OBSERVABILIDADE
            # Phoenix monitora automaticamente via instrumentação, mas podemos adicionar logs extras
            if phoenix_service.is_enabled:
                print(f"🔥 Enviando dados para Phoenix dashboard...")
                phoenix_service.log_rag_interaction(
                    question=question,
                    answer=response["answer"],
                    contexts=contexts_for_db,
                    sources=response.get("sources", []),
                    response_time=response_time,
                    metadata={
                        "interaction_id": interaction_id,        # ID único da interação
                        "max_documents": max_documents,          # Parâmetros da busca
                        "category_filter": category_filter,     # Filtros aplicados
                        "has_context": response.get("has_context", False),  # Se encontrou contexto
                        "context_used": response.get("context_used", 0)     # Quantos docs usou
                    }
                )
                print(f"✅ Dados enviados para Phoenix - dashboard em: {phoenix_service.get_phoenix_url()}")
        
        print(f"🎉 Pipeline RAG concluído com sucesso!")
        return response
    
    async def _save_interaction(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        sources: List[Dict],
        response_time: float
    ) -> str:
        """Salva a interação RAG no banco de dados
        
        IMPORTANTE: Esses dados são essenciais para:
        - RAGAS avaliar qualidade das respostas
        - Acompanhar performance ao longo do tempo
        - Gerar relatórios de uso
        - Identificar padrões e problemas
        
        DADOS SALVOS:
        - question: Pergunta original do usuário
        - answer: Resposta gerada pelo LLM
        - contexts: Textos dos documentos usados como contexto
        - sources: Metadados das fontes citadas
        - response_time: Tempo que demorou para responder
        
        Returns:
            ID único da interação salva
        """
        # Gerar ID único para a interação
        interaction_id = str(uuid.uuid4())
        
        # Salvar no banco de dados
        async with AsyncSessionLocal() as session:
            interaction = RAGInteractionDB(
                id=interaction_id,
                question=question,          # RAGAS usará isso para avaliar
                answer=answer,              # RAGAS avaliará se é boa resposta
                contexts=contexts,          # RAGAS avaliará se contexto é relevante
                sources=sources,            # Metadados das fontes
                response_time=response_time # Métrica de performance
            )
            
            session.add(interaction)
            await session.commit()
            
        return interaction_id