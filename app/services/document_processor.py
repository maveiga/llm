import os
from typing import List
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangChainDocument
from app.models.document import Document


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len, #garante que maximo será 500 com os 50
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
        lines = content.split('\n')
        cleaned_lines = []
        
        skip_patterns = [
            "Este parágrafo é um exemplo de ruído textual",
            "Esta linha não tem relação com o conteúdo",
            "Dados fictícios devem ser ignorados",
            "Título:",
            "Categoria:"
        ]
        
        for line in lines:
            line = line.strip()
            if line and not any(pattern in line for pattern in skip_patterns):
                cleaned_lines.append(line)

        text_to_process = ' '.join(cleaned_lines)
        
        # Se o modelo spaCy não estiver disponível, retorna apenas a limpeza básica
        if not self.nlp:
            print("Modelo spaCy não carregado. Usando limpeza básica.")
            return text_to_process

        doc = self.nlp(text_to_process)
        cleaned_tokens_ok = []

        for token in doc:
            #is_stop = "o", "um", "de", "que" is_punct é acentos, passando nessa verificação pegamos o lema  by teteu 
            if not token.is_stop and not token.is_punct and not token.is_space:
                cleaned_tokens_ok.append(token.lemma_.lower())
        
        return ' '.join(cleaned_tokens_ok)
    

        
    # muito manual, vou remover
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