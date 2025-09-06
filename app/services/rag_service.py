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
    
    Orquestra todo o pipeline RAG:
    
    PIPELINE COMPLETO:
    Pergunta do Usuário

    1. VectorService: Busca documentos similares

    2. LLMService: Gera resposta baseada nos documentos

    3. DatabaseService: Salva interação completa

    4. PhoenixService: Monitora para observabilidade

    5. RAGAS: Pode avaliar qualidade depois

    Resposta + Fontes para o Usuário
    
    INTEGRAÇÕES:
    - LangChain: Via LLMService para geração de resposta
    - Phoenix: Monitoramento automático de todo o pipeline
    - RAGAS: Usa dados salvos para avaliar qualidade
    """
    
    def __init__(self):
        self.vector_service = VectorService()  # Busca vetorial (ChromaDB + embeddings)
        self.llm_service = LLMService()        # Geração de resposta (LangChain + OpenAI)
    
    async def ask_question(
        self, 
        question: str, 
        max_documents: int = 5,
        category_filter: Optional[str] = None,
        save_interaction: bool = True
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        # Vector search: converte pergunta em embedding e busca documentos similares
        relevant_docs = await self.vector_service.search_documents(
            query=question,                # Pergunta vira embedding
            limit=max_documents,           # Top-K documentos mais similares
            category_filter=category_filter # Filtro opcional por categoria
        )
        
        if not relevant_docs:
            response = {
                "answer": "Desculpe, não encontrei documentos relevantes para responder sua pergunta.",
                "sources": [],
                "context_used": 0,
                "question": question,
                "has_context": False
            }
            
            if save_interaction:
                await self._save_interaction(
                    question=question,
                    answer=response["answer"],
                    contexts=[],             
                    sources=[],              
                    response_time=time.time() - start_time
                )
            
            return response

        # CONVERTER DOCUMENTOS PARA FORMATO DO LLM
        context_documents = []  # Para LLMService
        contexts_for_db = []    # Para banco de dados
        
        for doc in relevant_docs:
            context_documents.append({
                "title": doc.title,
                "category": doc.category,
                "content": doc.content,
                "similarity_score": doc.similarity_score,
                "metadata": doc.metadata
            })

            contexts_for_db.append(doc.content)

        #GERAR RESPOSTA COM LLM (LANGCHAIN + OPENAI)
        llm_response = await self.llm_service.generate_answer(
            question=question,              # Pergunta original
            context_documents=context_documents  # Documentos como contexto
        )

        # ETAPA 4: ENRIQUECER RESPOSTA COM INFORMAÇÕES ADICIONAIS
        search_results = [
            {
                "title": doc.title,
                "category": doc.category,
                "similarity_score": doc.similarity_score,
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
        
        # SALVAR INTERAÇÃO NO BANCO DE DADOS
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
            
            # Phoenix monitora automaticamente via instrumentação do OpenTelemetry
            if phoenix_service.is_enabled:
                print(f"Phoenix ativo ")
        
        return response
    
    async def _save_interaction(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        sources: List[Dict],
        response_time: float
    ) -> str:

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