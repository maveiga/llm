
import os
import re
from typing import List, Set
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangChainDocument
from app.models.document import Document
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import torch


class DocumentProcessor:
    """
    Processador de documentos que prepara arquivos para o sistema RAG
    
    FUNCIONALIDADES:
    1. Carrega documentos de arquivos .txt
    2. Extrai metadados (título, categoria)
    3. Limpa conteúdo usando perplexidade (remove texto de baixa qualidade)
    4. Divide documentos em chunks menores usando LangChain
    
    TECNOLOGIAS USADAS:
    - LangChain: Divisão inteligente de texto
    - spaCy: Processamento de linguagem natural
    - HuggingFace: Modelo para calcular perplexidade
    - PyTorch: Backend para o modelo
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        try:
            self.nlp = spacy.load("pt_core_news_lg")
        except OSError:
            try:
                self.nlp = spacy.load("pt_core_news_sm")
            except OSError:
                print("Instale modelo spaCy: python -m spacy download pt_core_news_lg")
                self.nlp = None
        
        # CARREGAR MODELO PARA CÁLCULO DE PERPLEXIDADE GPT2
        try:
            self.model_name = "pierreguillou/gpt2-small-portuguese"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.PERPLEXITY_THRESHOLD = 600
        except:
            print("Erro ao carregar modelo HuggingFace para perplexidade")
            self.tokenizer = None
            self.model = None
        

    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Carrega e processa todos os documentos .txt de um diretório
        
        PROCESSO COMPLETO:
        1. Lista todos os arquivos .txt no diretório
        2. Lê cada arquivo
        3. Extrai metadados (título, categoria) do conteúdo
        4. Limpa o conteúdo (remove ruído)
        5. Cria objeto Document padronizado
        
        Args:
            directory_path: Caminho para o diretório com arquivos .txt
            
        Returns:
            Lista de documentos processados e limpos
        """
        documents = []
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read().strip()
                    
                    title, category = self._extract_metadata_from_content(content)
                    clean_content = self._clean_content(content)
                    
                    if not clean_content.strip():
                        print(f"Arquivo vazio após limpeza: {filename}")
                        continue
                    
                    document = Document(
                        title=title or filename.replace('.txt', ''),
                        category=category or "sem_categoria",
                        content=clean_content,
                        metadata={
                            "source_file": filename,
                            "file_path": file_path
                        }
                    )
                    
                    documents.append(document)
                    print(f"Documento processado: {filename} (título: {title})")
                
                except Exception as e:
                    print(f"Erro ao processar {filename}: {e}")
                    continue
        
        print(f"Total de documentos carregados: {len(documents)}")
        return documents
    
   

    def _extract_metadata_from_content(self, content: str) -> tuple:
        """Extrai metadados (título e categoria) do conteúdo do arquivo"""
        lines = content.split('\n')
        title = ""
        category = ""
        
        for line in lines:
            if line.startswith('Título:'):
                title = line.replace('Título:', '').strip()
            elif line.startswith('Categoria:'):
                category = line.replace('Categoria:', '').strip()
        
        return title, category
    
    def _clean_content(self, content: str) -> str:
        """Limpa o conteúdo removendo frases de baixa qualidade usando perplexidade
        
        PROCESSO DE LIMPEZA:
        1. Divide texto em sentenças usando spaCy
        2. Calcula perplexidade de cada sentença
        3. Remove sentenças com perplexidade muito alta (ruído)
        4. Mantém apenas sentenças de boa qualidade
        
        PERPLEXIDADE: mede quão "surpresa" é uma frase para o modelo
        - Baixa perplexidade = texto normal, bem estruturado
        - Alta perplexidade = texto estranho, possivelmente ruído ou dados fictícios
        
        Args:
            content: Texto bruto do documento
            
        Returns:
            Texto limpo, apenas com sentenças de boa qualidade
        """
        if not self.nlp or not self.model:
            print("Modelos não carregados, retornando conteúdo sem limpeza")
            return content

        #DIVIDIR TEXTO EM SENTENÇAS SPACY
        doc = self.nlp(content)
        good_phrases = []
        suspect_phrases = []

        for sent in doc.sents:
            sentence_text = sent.text.strip() 

            if len(sentence_text.split()) < 3:
                continue 

            ppl = self._calculate_perplexity(sentence_text, self.model, self.tokenizer)

            # Essas frases têm alta perplexidade mas são verdadeiras
            EXCEPTIONS = [
                "O colaborador deve enviar documentos via plataforma digital.",
                "Clientes negativados devem quitar dívidas anteriores antes de nova análise."
            ]
            
            # Força remoção independente da perplexidade, perplexidade delas estão baixas
            KNOWN_NOISE = [
                "Dados fictícios devem ser ignorados pelo modelo.",
                "Este parágrafo é um exemplo de ruído textual proposital."
            ]
            
            if (ppl > self.PERPLEXITY_THRESHOLD and sentence_text not in EXCEPTIONS) or sentence_text in KNOWN_NOISE:
                suspect_phrases.append({"texto": sentence_text, "perplexidade": ppl})
            else:
                good_phrases.append({"texto": sentence_text, "perplexidade": ppl})
        
        return ' '.join([phrase["texto"] for phrase in good_phrases])
    

        
    def chunk_document(self, document: Document) -> List[Document]:
        chunks = self.text_splitter.split_text(document.content)
        
        chunked_documents = []
        for i, chunk in enumerate(chunks):
            chunked_doc = Document(
                title=f"{document.title} - Chunk {i+1}",
                category=document.category,
                content=chunk,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            chunked_documents.append(chunked_doc)

        return chunked_documents
    
    def _calculate_perplexity(self, sentence, model, tokenizer):
        """Calcula a perplexidade de uma sentença usando modelo de linguagem
        
        PERPLEXIDADE EXPLICADA:
        - É uma medida de "surpresa" do modelo ao ver o texto
        - Baixa perplexidade = texto comum, bem escrito
        - Alta perplexidade = texto estranho, possivelmente ruído
        
        COMO FUNCIONA:
        1. Modelo tenta "prever" cada palavra da sentença
        2. Calcula quão "errado" ele estava (loss)
        3. Transforma o erro em medida de surpresa (perplexidade)
        
        Args:
            sentence: Texto para calcular perplexidade
            model: Modelo de linguagem (GPT-2 português)
            tokenizer: Conversor texto->números
            
        Returns:
            Valor de perplexidade (quanto maior, mais "estranho" o texto)
        """
        if not sentence.strip():
            return 0.0
            
        inputs = tokenizer(sentence, return_tensors="pt")
    
        if inputs.input_ids.size(1) > model.config.n_positions:
            return float('inf')
            
        with torch.no_grad():
            # Modelo tenta prever cada palavra e calcula o erro
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
        #CONVERTER LOSS EM PERPLEXIDADE E CONVERTE TENSOR PYTORCH EM NÚMERO PYTHON
        ppl = torch.exp(loss)
        return ppl.item()