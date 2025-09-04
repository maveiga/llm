import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.core.config import settings

class LLMService:
    def __init__(self):
        # Inicializa o modelo GPT-3.5
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=settings.openai_api_key
        )
    
    async def generate_answer(
        self, 
        question: str, 
        context_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Gera uma resposta baseada na pergunta e documentos de contexto
        
        Args:
            question: Pergunta do usuário
            context_documents: Lista de documentos relevantes encontrados
            
        Returns:
            Dict com resposta e informações de citação
        """
        
        # Prepara o contexto dos documentos
        context_text = ""
        sources = []
        
        for i, doc in enumerate(context_documents, 1):
            context_text += f"[Documento {i}]\n"
            context_text += f"Título: {doc.get('title', 'N/A')}\n"
            context_text += f"Categoria: {doc.get('category', 'N/A')}\n"
            context_text += f"Conteúdo: {doc.get('content', '')}\n\n"
            
            sources.append({
                "id": i,
                "title": doc.get('title', 'N/A'),
                "category": doc.get('category', 'N/A'),
                "similarity_score": doc.get('similarity_score')
            })
        
        # Prompt do sistema
        system_prompt = """Você é um assistente especializado que responde perguntas baseado exclusivamente nos documentos fornecidos.

Instruções:
1. Use APENAS as informações dos documentos fornecidos para responder
2. Se a pergunta não puder ser respondida com os documentos, diga que não há informações suficientes
3. Sempre cite as fontes usando [Documento X] no final de cada afirmação
4. Seja preciso e objetivo
5. Mantenha um tom profissional

Documentos disponíveis:
{context}"""

        # Prompt do usuário
        user_prompt = f"Pergunta: {question}\n\nPor favor, responda baseado nos documentos fornecidos."

        # Prepara as mensagens
        messages = [
            SystemMessage(content=system_prompt.format(context=context_text)),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            # Gera a resposta
            response = await self.llm.ainvoke(messages)
            answer = response.content
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": len(context_documents),
                "question": question
            }
            
        except Exception as e:
            return {
                "answer": f"Erro ao gerar resposta: {str(e)}",
                "sources": sources,
                "context_used": len(context_documents),
                "question": question,
                "error": str(e)
            }
    
    async def check_connection(self) -> bool:
        """Verifica se a conexão com a API OpenAI está funcionando"""
        try:
            response = await self.llm.ainvoke([HumanMessage(content="Hello")])
            return True
        except Exception:
            return False