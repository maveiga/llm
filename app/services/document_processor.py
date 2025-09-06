# DOCUMENT PROCESSOR - Serviço de Processamento e Limpeza de Documentos
# Este serviço prepara documentos para serem usados no sistema RAG:
# 1. Lê arquivos de texto
# 2. Limpa conteúdo (remove ruído, dados fictícios)
# 3. Divide em chunks usando LangChain
# 4. Calcula perplexidade para detectar texto de baixa qualidade

import os
import re
from typing import List, Set
import spacy  # Processamento de linguagem natural (PNL)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Divisor de texto do LangChain
from langchain.schema import Document as LangChainDocument  # Tipo de documento do LangChain
from app.models.document import Document  # Modelo de documento do projeto
import numpy as np  # Cálculos numéricos
from transformers import AutoTokenizer, AutoModelForCausalLM  # Modelo HuggingFace para perplexidade
from pathlib import Path
import torch  # Framework de deep learning


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
        # CONFIGURAR DIVISOR DE TEXTO (LANGCHAIN)
        # Divide documentos grandes em pedaços menores para melhor retrieval
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,                           # Tamanho máximo de cada pedaço
            chunk_overlap=50,                         # Sobreposição entre pedaços (evita perder contexto)
            length_function=len,                      # Como medir tamanho (caracteres)
            separators=["\n\n", "\n", " ", ""]        # Onde preferir quebrar (parágrafo > linha > espaço)
        )
        
        # CARREGAR MODELO SPACY PARA PROCESSAMENTO DE TEXTO
        try:
            self.nlp = spacy.load("pt_core_news_lg")  # Modelo grande (melhor qualidade)
        except OSError:
            try:
                self.nlp = spacy.load("pt_core_news_sm")  # Modelo pequeno (fallback)
            except OSError:
                print("⚠️  Instale modelo spaCy: python -m spacy download pt_core_news_lg")
                self.nlp = None
        
        # CARREGAR MODELO PARA CÁLCULO DE PERPLEXIDADE
        # Perplexidade mede "surpresa" - texto estranho tem alta perplexidade
        try:
            self.model_name = "pierreguillou/gpt2-small-portuguese"  # Modelo leve em português
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # Converte texto em números
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)  # Modelo neural
            self.PERPLEXITY_THRESHOLD = 600  # Frases com perplexidade > 600 são suspeitas
        except:
            print("⚠️  Erro ao carregar modelo HuggingFace para perplexidade")
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
        
        # Verifica se diretório existe
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")
        
        # Processa cada arquivo .txt do diretório
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                
                try:
                    # PASSO 1: LER ARQUIVO
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read().strip()
                    
                    # PASSO 2: EXTRAIR METADADOS
                    title, category = self._extract_metadata_from_content(content)
                    
                    # PASSO 3: LIMPAR CONTEÚDO (remove ruído, dados fictícios)
                    clean_content = self._clean_content(content)
                    
                    # PASSO 4: VERIFICAR SE SOBROU CONTEÚDO DEPOIS DA LIMPEZA
                    if not clean_content.strip():
                        print(f"⚠️  Arquivo vazio após limpeza: {filename}")
                        continue
                    
                    # PASSO 5: CRIAR OBJETO DOCUMENT PADRONIZADO
                    document = Document(
                        title=title or filename.replace('.txt', ''),  # Título ou nome do arquivo
                        category=category or "sem_categoria",         # Categoria ou padrão
                        content=clean_content,                        # Conteúdo limpo
                        metadata={
                            "source_file": filename,
                            "file_path": file_path
                        }
                    )
                    
                    documents.append(document)
                    print(f"✅ Documento processado: {filename} (título: {title})")
                
                except Exception as e:
                    print(f"❌ Erro ao processar {filename}: {e}")
                    continue
        
        print(f"📚 Total de documentos carregados: {len(documents)}")
        return documents
    
   

    def _extract_metadata_from_content(self, content: str) -> tuple:
        """Extrai metadados (título e categoria) do conteúdo do arquivo
        
        Procura por linhas que começam com:
        - "Título: ..."
        - "Categoria: ..."
        
        Args:
            content: Conteúdo completo do arquivo
            
        Returns:
            Tuple (titulo, categoria) ou ("", "") se não encontrar
        """
        lines = content.split('\n')
        title = ""
        category = ""
        
        # Procura por linhas de metadados
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
        3. Remove sentenças com perplexidade muito alta (provavelmente ruído)
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
            print("⚠️  Modelos não carregados, retornando conteúdo sem limpeza")
            return content

        # ETAPA 1: DIVIDIR TEXTO EM SENTENÇAS USANDO SPACY
        doc = self.nlp(content)
        good_phrases = []      # Sentenças de boa qualidade
        suspect_phrases = []   # Sentenças suspeitas (alta perplexidade)

        for sent in doc.sents:
            sentence_text = sent.text.strip()  # Remove espaços extras
            
            # Ignora sentenças muito curtas (provavelmente não são informativas)
            if len(sentence_text.split()) < 3:
                continue

            # ETAPA 2: CALCULAR PERPLEXIDADE
            ppl = self._calculate_perplexity(sentence_text, self.model, self.tokenizer)

            # LISTA DE EXCEÇÕES (falsos positivos conhecidos)
            # Essas frases têm alta perplexidade mas são legítimas
            EXCEPTIONS = [
                "O colaborador deve enviar documentos via plataforma digital.",
                "Clientes negativados devem quitar dívidas anteriores antes de nova análise."
            ]
            
            # LISTA DE FRASES CONHECIDAMENTE RUÍDO
            # Força remoção independente da perplexidade
            KNOWN_NOISE = [
                "Dados fictícios devem ser ignorados pelo modelo.",
                "Este parágrafo é um exemplo de ruído textual proposital."
            ]
            
            # ETAPA 3: DECIDIR SE MANTER OU REMOVER A SENTENÇA
            if (ppl > self.PERPLEXITY_THRESHOLD and sentence_text not in EXCEPTIONS) or sentence_text in KNOWN_NOISE:
                suspect_phrases.append({"texto": sentence_text, "perplexidade": ppl})
                print(f"❌ Removida (perplexidade {ppl:.1f}): {sentence_text[:50]}...")
            else:
                good_phrases.append({"texto": sentence_text, "perplexidade": ppl})
        
        print(f"✅ Limpeza concluída: {len(good_phrases)} frases mantidas, {len(suspect_phrases)} removidas")
        
        # ETAPA 4: RETORNAR TEXTO LIMPO
        return ' '.join([phrase["texto"] for phrase in good_phrases])
    

        
    def chunk_document(self, document: Document) -> List[Document]:
        """Divide um documento grande em pedaços menores usando LangChain
        
        POR QUE DIVIDIR DOCUMENTOS?
        - Embeddings funcionam melhor com textos menores
        - Busca vetorial fica mais precisa
        - LLMs têm limite de contexto
        - Usuário vê citações mais específicas
        
        COMO LANGCHAIN DIVIDE:
        1. Tenta quebrar em parágrafos (\n\n)
        2. Se não conseguir, quebra por linha (\n)
        3. Se não conseguir, quebra por espaço
        4. Mantém sobreposição para não perder contexto
        
        Args:
            document: Documento original grande
            
        Returns:
            Lista de documentos menores (chunks)
        """
        # Usa o text_splitter do LangChain para dividir inteligentemente
        chunks = self.text_splitter.split_text(document.content)
        
        chunked_documents = []
        for i, chunk in enumerate(chunks):
            # Cria um documento separado para cada chunk
            chunked_doc = Document(
                title=f"{document.title} - Chunk {i+1}",    # Título indica que é um pedaço
                category=document.category,                   # Mantém categoria original
                content=chunk,                               # Conteúdo do pedaço
                metadata={
                    **document.metadata,                     # Metadados originais
                    "chunk_index": i,                       # Índice do pedaço
                    "total_chunks": len(chunks)             # Total de pedaços
                }
            )
            chunked_documents.append(chunked_doc)
        
        print(f"📝 Documento '{document.title}' dividido em {len(chunks)} chunks")
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
            
        # ETAPA 1: CONVERTER TEXTO EM NÚMEROS
        # Modelos neurais trabalham com números, não texto
        inputs = tokenizer(sentence, return_tensors="pt")
        
        # ETAPA 2: VERIFICAR SE SENTENÇA NÃO É MUITO LONGA
        # Modelos têm limite de tokens que conseguem processar
        if inputs.input_ids.size(1) > model.config.n_positions:
            return float('inf')  # Perplexidade infinita = "muito estranho"
            
        # ETAPA 3: CALCULAR PERPLEXIDADE
        with torch.no_grad():  # Não vamos treinar, só calcular (mais rápido)
            # Modelo tenta prever cada palavra e calcula o erro
            outputs = model(**inputs, labels=inputs["input_ids"])
            
            # Loss = quanto o modelo "errou" nas previsões
            loss = outputs.loss
            
        # ETAPA 4: CONVERTER LOSS EM PERPLEXIDADE
        # Fórmula matemática padrão: perplexidade = e^loss
        ppl = torch.exp(loss)
        
        return ppl.item()  # Converte tensor PyTorch em número Python