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
        
        # ETAPA 1: PREPARAR CONTEXTO DOS DOCUMENTOS
        # Converte documentos encontrados num texto estruturado que GPT entende
        context_text = ""
        sources = []  # Lista de fontes para citação
        
        for i, doc in enumerate(context_documents, 1):
            # Formata cada documento de forma estruturada
            context_text += f"[Documento {i}]\n"
            context_text += f"Título: {doc.get('title', 'N/A')}\n"
            context_text += f"Categoria: {doc.get('category', 'N/A')}\n"
            context_text += f"Conteúdo: {doc.get('content', '')}\n\n"
            
            # Prepara informações das fontes para citação
            sources.append({
                "id": i,
                "title": doc.get('title', 'N/A'),
                "category": doc.get('category', 'N/A'),
                "similarity_score": doc.get('similarity_score')  # Quão similar à pergunta (0-1)
            })
        
        # ETAPA 2: CRIAR PROMPT DO SISTEMA (INSTRUÇÕES PARA O AI)
        # Este prompt ensina o GPT como ele deve se comportar
        system_prompt = """Você é um assistente especializado que responde perguntas baseado exclusivamente nos documentos fornecidos.

REGRAS IMPORTANTES:
1. Use APENAS as informações dos documentos fornecidos para responder
2. Se a pergunta não puder ser respondida com os documentos, diga que não há informações suficientes
3. Sempre cite as fontes usando [Documento X] no final de cada afirmação
4. Seja preciso e objetivo - evite invenções ou alucinações
5. Mantenha um tom profissional

DOCUMENTOS DISPONÍVEIS PARA CONSULTA:
{context}"""

        # ETAPA 3: CRIAR PROMPT DO USUÁRIO (PERGUNTA REAL)
        user_prompt = f"Pergunta: {question}\n\nPor favor, responda baseado exclusivamente nos documentos fornecidos acima."

        # ETAPA 4: PREPARAR MENSAGENS NO FORMATO LANGCHAIN
        # LangChain usa tipos específicos de mensagem para estruturar a conversa
        messages = [
            SystemMessage(content=system_prompt.format(context=context_text)),  # Instruções + contexto
            HumanMessage(content=user_prompt)                                   # Pergunta do usuário
        ]
        
        try:
            # ETAPA 5: ENVIAR PARA GPT VIA LANGCHAIN E AGUARDAR RESPOSTA
            # ainvoke = chamada assíncrona para o modelo de linguagem
            response = await self.llm.ainvoke(messages)
            answer = response.content  # Extrai o texto da resposta
            
            # ETAPA 6: ESTRUTURAR RESPOSTA FINAL
            return {
                "answer": answer,                        # Resposta gerada pelo GPT
                "sources": sources,                      # Fontes citadas
                "context_used": len(context_documents),  # Quantos documentos foram usados
                "question": question                     # Pergunta original
            }
            
        except Exception as e:
            # Se der erro (sem internet, API key inválida, etc.)
            return {
                "answer": f"Erro ao gerar resposta: {str(e)}",
                "sources": sources,
                "context_used": len(context_documents),
                "question": question,
                "error": str(e)
            }
    
    async def check_connection(self) -> bool:
        """Verifica se a conexão com a API OpenAI está funcionando
        
        Usado para health checks:
        - Verifica se API key é válida
        - Testa conectividade
        - Garante que modelo está disponível
        """
        try:
            # Faz uma chamada simples para testar
            response = await self.llm.ainvoke([HumanMessage(content="Hello")])
            return True  # Se chegou até aqui, está funcionando
        except Exception:
            return False  # Algo deu errado (sem API key, sem internet, etc.)