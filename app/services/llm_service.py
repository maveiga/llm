# LLM SERVICE - Serviço de Geração de Respostas com LangChain + OpenAI
# Este é o "cérebro" do sistema RAG que gera as respostas finais
# Usa LangChain para facilitar a comunicação com modelos de linguagem

import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI  # Interface LangChain para OpenAI GPT
from langchain.schema import HumanMessage, SystemMessage  # Tipos de mensagem do LangChain
from app.core.config import settings  # Configurações (API keys, etc.)

class LLMService:
    """
    Serviço que usa LangChain + OpenAI para gerar respostas RAG
    
    PROCESSO:
    1. Recebe pergunta + documentos relevantes
    2. Cria um prompt estruturado
    3. Envia para GPT via LangChain
    4. Retorna resposta + citações
    
    LangChain facilita:
    - Comunicação com diferentes LLMs
    - Estruturação de mensagens
    - Tratamento de erros
    - Instrumentação automática (Phoenix monitora)
    """
    
    def __init__(self):
        # Inicializa o modelo GPT-3.5 via LangChain
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",           # Modelo específico (pode ser GPT-4, Claude, etc.)
            temperature=0.7,                # Criatividade (0=robótico, 1=criativo)
            api_key=settings.openai_api_key # Chave da API OpenAI
        )
    
    async def generate_answer(
        self, 
        question: str, 
        context_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        FUNÇÃO PRINCIPAL: Gera resposta RAG usando LangChain + OpenAI
        
        PROCESSO DETALHADO:
        1. Pega documentos relevantes encontrados pelo vector search
        2. Formata eles num contexto estruturado
        3. Cria prompt system (instruções para o AI)
        4. Cria prompt user (pergunta do usuário)
        5. Envia tudo para GPT via LangChain
        6. Processa resposta e adiciona citações
        
        Args:
            question: Pergunta do usuário
            context_documents: Lista de documentos relevantes encontrados pelo RAG
            
        Returns:
            Dict com resposta gerada, fontes citadas e metadados
        """
        
        # Converte documentos encontrados num texto estruturado que GPT entende
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
        
        system_prompt = """Você é um assistente especializado que responde perguntas baseado exclusivamente nos documentos fornecidos.

        REGRAS IMPORTANTES:
        1. Use APENAS as informações dos documentos fornecidos para responder
        2. Se a pergunta não puder ser respondida com os documentos, diga que não há informações suficientes
        3. Sempre cite as fontes usando [Documento X] no final de cada afirmação
        4. Se possivel cite fontes como titulo e categoria no formato: (titulo - categoria) no final de cada afirmação
        5. Seja preciso e objetivo - evite invenções ou alucinações
        6. Mantenha um tom profissional

        DOCUMENTOS DISPONÍVEIS PARA CONSULTA:
        {context}"""

        user_prompt = f"Pergunta: {question}\n\nPor favor, responda baseado exclusivamente nos documentos fornecidos acima."

        messages = [
            SystemMessage(content=system_prompt.format(context=context_text)),
            HumanMessage(content=user_prompt)                                   
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            answer = response.content
            
            return {
                "answer": answer,                        # Resposta gerada pelo GPT
                "sources": sources,                      # Fontes citadas
                "context_used": len(context_documents),  # Quantos documentos foram usados
                "question": question                     # Pergunta original
            }
            
        except Exception as e:
            return {
                "answer": f"Erro ao gerar resposta: {str(e)}",
                "sources": sources,
                "context_used": len(context_documents),
                "question": question,
                "error": str(e)
            }
