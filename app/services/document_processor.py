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
        try:
            self.model_name = "pierreguillou/gpt2-small-portuguese" #leve e não precisa de GPU
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.PERPLEXITY_THRESHOLD = 600 #valor definido conforme fui testando o modelo
        except:
            print("Erro ao carregar modelos da Hugging Face. Verifique sua conexão com a internet e se o modelo existe.")
            self.tokenizer = None
            self.model = None
        

    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
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
                
                except Exception as e:
                    print(f"Erro ao processar {filename}: {e}")
                    continue
        
        return documents
    
   

    def _extract_metadata_from_content(self, content: str) -> tuple:
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

        doc = self.nlp(content)
        good_phrases = []
        suspect_phrases = []

        for sent in doc.sents:
                sentence_text = sent.text.strip() #pega a sentença como string e strip remove espaços em branco
                if len(sentence_text.split()) < 3: # Ignora sentenças muito curtas
                    continue

                # Calcula a perplexidade da sentença
                ppl = self._calculate_perplexity(sentence_text, self.model, self.tokenizer)

                EXCEPTIONS = [
                    "O colaborador deve enviar documentos via plataforma digital.",
                    "Clientes negativados devem quitar dívidas anteriores antes de nova análise."
                ]
                #Essas frases geraram falso positivo, ou seja, foram marcadas como suspeitas mas na verdade são boas.
                #Então vou ignorá-las na análise de perplexidade.
                #proximo passo é gravar isso numa base de dados para não perder.
                
                LOWER_PERPLEXITY= [
                    "Dados fictícios devem ser ignorados pelo modelo.",
                    "Este parágrafo é um exemplo de ruído textual proposital."
                ]
                if ppl > self.PERPLEXITY_THRESHOLD and sentence_text not in EXCEPTIONS or sentence_text in LOWER_PERPLEXITY:
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

        if not sentence.strip():
            return 0.0
            
        inputs = tokenizer(sentence, return_tensors="pt") #converte a sentença em números.
        
        # Previne erro com sentenças muito longas para o modelo
        if inputs.input_ids.size(1) > model.config.n_positions:
            return float('inf') # Retorna perplexidade infinita se for muito longa
            
        with torch.no_grad(): #só vamos calcular, não vamos aprender nada novo agora", o que torna o processo mais rápido.
            outputs = model(**inputs, labels=inputs["input_ids"]) #A sentença (já em números) é enviada para o "cérebro" (modelo).
            #O modelo tenta "prever" a próxima palavra a cada passo e calcula o quão errado ele estava.
            #  Essa medida de "erro" é chamada de loss (perda).
            #enfim, quanto mais errada for a previsão  maior sera perplexidade 
            loss = outputs.loss #"Pegue o valor de erro calculado pelo modelo quando ele tentou prever cada palavra da frase, e guarde esse número." 
            
        # A perplexidade é o exponencial da loss
        ppl = torch.exp(loss) #Uma fórmula matemática (a exponencial) é aplicada ao "erro" (loss)
        #para transformá-lo na nossa nota de "surpresa" (perplexidade, ppl).
        return ppl.item()