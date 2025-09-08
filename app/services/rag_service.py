from typing import List, Dict, Any, Optional
import time
import uuid
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.models.document import DocumentResponse
from app.models.rag_interaction import RAGInteractionDB, RAGInteractionCreate
from app.services.database_service import AsyncSessionLocal
from app.services.phoenix_service import phoenix_service
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
        self.vector_service = VectorService() 
        self.llm_service = LLMService()
    
    async def ask_question(
        self, 
        question: str, 
        max_documents: int = 5,
        category_filter: Optional[str] = None,
        save_interaction: bool = True
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        print(f"convertendo pergunta em embedding e buscando documentos similares")
        relevant_docs = await self.vector_service.search_documents(
            query=question,
            limit=max_documents,
            category_filter=category_filter
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

        context_documents = []
        contexts_for_db = []    #banco de dados
        
        for doc in relevant_docs:
            context_documents.append({
                "title": doc.title,
                "category": doc.category,
                "content": doc.content,
                "similarity_score": doc.similarity_score,
                "metadata": doc.metadata
            })

            contexts_for_db.append(doc.content)


        llm_response = await self.llm_service.generate_answer(
            question=question,
            context_documents=context_documents
        )

        search_results = [
            {
                "title": doc.title,
                "category": doc.category,
                "similarity_score": doc.similarity_score,
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            }
            for doc in relevant_docs
        ]

        response = {
            **llm_response,
            "has_context": True,
            "search_results": search_results
        }
        
        response_time = time.time() - start_time

        
        if save_interaction:
            interaction_id = await self._save_interaction(
                question=question,
                answer=response["answer"],
                contexts=contexts_for_db,
                sources=response.get("sources", []),
                response_time=response_time
            )
            response["interaction_id"] = interaction_id
            
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
        interaction_id = str(uuid.uuid4())
        
        async with AsyncSessionLocal() as session:
            interaction = RAGInteractionDB(
                id=interaction_id,
                question=question,
                answer=answer,
                contexts=contexts,
                sources=sources,
                response_time=response_time
            )
            
            session.add(interaction)
            await session.commit()
            
        return interaction_id