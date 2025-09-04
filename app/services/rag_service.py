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
        """
        Pipeline RAG completo:
        1. Busca documentos relevantes
        2. Gera resposta com LLM
        3. Salva interação no banco de dados
        4. Retorna resposta + citações
        
        Args:
            question: Pergunta do usuário
            max_documents: Máximo de documentos para usar como contexto
            category_filter: Filtro opcional por categoria
            save_interaction: Se deve salvar a interação no banco de dados
            
        Returns:
            Dict com resposta, fontes e metadados
        """
        start_time = time.time()
        
        # Etapa 1: Buscar documentos relevantes no vector store
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
            
            # Salvar interação mesmo sem contexto
            if save_interaction:
                await self._save_interaction(
                    question=question,
                    answer=response["answer"],
                    contexts=[],
                    sources=[],
                    response_time=time.time() - start_time
                )
            
            return response
        
        # Etapa 2: Converter documentos para formato do LLM
        context_documents = []
        contexts_for_db = []
        
        for doc in relevant_docs:
            context_documents.append({
                "title": doc.title,
                "category": doc.category,
                "content": doc.content,
                "similarity_score": doc.similarity_score,
                "metadata": doc.metadata
            })
            contexts_for_db.append(doc.content)
        
        # Etapa 3: Gerar resposta com LLM
        llm_response = await self.llm_service.generate_answer(
            question=question,
            context_documents=context_documents
        )
        
        # Etapa 4: Enriquecer resposta com informações adicionais
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
        
        # Etapa 5: Salvar interação no banco de dados
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
            
            # Etapa 6: Log da interação no Phoenix para observabilidade
            if phoenix_service.is_enabled:
                phoenix_service.log_rag_interaction(
                    question=question,
                    answer=response["answer"],
                    contexts=contexts_for_db,
                    sources=response.get("sources", []),
                    response_time=response_time,
                    metadata={
                        "interaction_id": interaction_id,
                        "max_documents": max_documents,
                        "category_filter": category_filter,
                        "has_context": response.get("has_context", False),
                        "context_used": response.get("context_used", 0)
                    }
                )
        
        return response
    
    async def health_check(self) -> Dict[str, bool]:
        """Verifica se todos os serviços estão funcionando"""
        return {
            "vector_service": True,  # ChromaDB normalmente não falha
            "llm_service": await self.llm_service.check_connection(),
            "phoenix_service": phoenix_service.is_enabled,
            "phoenix_url": phoenix_service.get_phoenix_url()
        }
    
    async def _save_interaction(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        sources: List[Dict],
        response_time: float
    ) -> str:
        """Salva a interação RAG no banco de dados"""
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