import os
import re
from typing import List, Set
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangChainDocument
from app.models.document import Document
import numpy as np

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
        
        # Padrões de ruído conhecidos
        self.noise_patterns = [
            "Este parágrafo é um exemplo de ruído textual",
            "Esta linha não tem relação com o conteúdo",
            "Dados fictícios devem ser ignorados",
            "Lorem ipsum",
            "banana azul",
            "Título:",
            "Categoria:",
            "Informação irrelevante:"
        ]
        
        # Palavras que frequentemente aparecem em texto de ruído
        self.noise_keywords = {
            'lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 'elit',
            'exemplo', 'ruído', 'textual', 'proposital', 'fictício', 'ignorar',
            'banana', 'azul', 'voadora', 'irrelevante'
        }
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        documents = []
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
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
    
    def _is_noise_sentence(self, sentence_text: str) -> bool:
        """
        Verifica se uma frase é ruído baseado em múltiplos critérios
        """
        sentence_lower = sentence_text.lower().strip()
        
        # 1. Verifica padrões conhecidos de ruído
        for pattern in self.noise_patterns:
            if pattern.lower() in sentence_lower:
                return True
        
        # 2. Verifica se contém muitas palavras de ruído
        words = set(re.findall(r'\b\w+\b', sentence_lower))
        noise_word_count = len(words.intersection(self.noise_keywords))
        
        # Se mais de 30% das palavras são palavras de ruído, considera ruído
        if len(words) > 0 and (noise_word_count / len(words)) > 0.3:
            return True
        
        # 3. Verifica frases muito curtas e sem sentido
        if len(words) < 3 and any(noise_word in words for noise_word in self.noise_keywords):
            return True
        
        # 4. Verifica padrões regex para sequências sem sentido
        nonsense_patterns = [
            r'\b[a-z]+\s+azul\b',  # qualquer palavra seguida de "azul"
            r'\bbanana\s+\w+\b',   # banana seguida de qualquer palavra
            r'\b[a-z]{1,3}\s+[a-z]{1,3}\s+[a-z]{1,3}\b'  # sequências de palavras muito curtas
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _is_semantically_coherent(self, sentence: spacy.tokens.span.Span, threshold: float = 0.25) -> bool:
        """
        Verifica a coerência semântica de uma frase com base na similaridade média dos vetores de palavras.
        """
        # Primeiro verifica se é ruído óbvio
        if self._is_noise_sentence(sentence.text):
            return False
        
        # Pega os vetores apenas de tokens que têm vetor e não são stop words ou pontuação.
        word_vectors = [
            token.vector for token in sentence 
            if token.has_vector and not token.is_stop and not token.is_punct and not token.is_space
        ]

        # Se a frase tiver poucas palavras significativas, consideramos coerente para evitar falsos positivos.
        if len(word_vectors) < 3:
            return True

        # Calcula o vetor médio (centroide) da frase
        sentence_vector = np.mean(word_vectors, axis=0)

        # Calcula a similaridade do cosseno de cada palavra com o vetor médio da frase
        similarities = [
            np.dot(word_vec, sentence_vector) / (np.linalg.norm(word_vec) * np.linalg.norm(sentence_vector))
            for word_vec in word_vectors
        ]
        
        # Calcula a similaridade média
        average_similarity = np.mean(similarities)
        
        # Se a similaridade média for maior que o limiar, a frase é considerada coerente.
        return average_similarity > threshold
    
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
        # 1. Limpeza inicial baseada em padrões
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not self._is_noise_sentence(line):
                cleaned_lines.append(line)

        text_to_process = ' '.join(cleaned_lines)
        
        if not self.nlp:
            print("Modelo spaCy não carregado. Usando limpeza básica.")
            return text_to_process

        doc = self.nlp(text_to_process)
        cleaned_sentences = []

        # 2. Filtragem por frase usando análise semântica
        for sentence in doc.sents:
            sentence_text = sentence.text.strip()
            
            # Pula frases muito curtas (menos de 10 caracteres)
            if len(sentence_text) < 10:
                continue
                
            # Verifica coerência semântica (já inclui verificação de ruído)
            if self._is_semantically_coherent(sentence, threshold=0.25):
                # Lematiza apenas as palavras significativas
                lemmatized_tokens = []
                for token in sentence:
                    if not token.is_stop and not token.is_punct and not token.is_space:
                        lemmatized_tokens.append(token.lemma_.lower())
                
                if lemmatized_tokens:  # Só adiciona se tiver tokens significativos
                    cleaned_sentences.append(' '.join(lemmatized_tokens))
        
        return ' '.join(cleaned_sentences)
    
    def add_noise_pattern(self, pattern: str):
        """Adiciona um novo padrão de ruído à lista"""
        if pattern not in self.noise_patterns:
            self.noise_patterns.append(pattern)
    
    def add_noise_keywords(self, keywords: List[str]):
        """Adiciona novas palavras-chave de ruído"""
        self.noise_keywords.update(keyword.lower() for keyword in keywords)
        
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
    
    def chunk_documents(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        return self.text_splitter.split_documents(documents)