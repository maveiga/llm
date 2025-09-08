# -*- coding: utf-8 -*-
"""
Script para limpar completamente o ChromaDB
"""

import chromadb
import shutil
import os
from app.core.config import settings

def clear_chromadb():
    try:
        print("Limpando ChromaDB...")
        
        try:
            client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
            client.delete_collection(name=settings.chroma_collection_name)
            print(f"Coleção '{settings.chroma_collection_name}' deletada com sucesso!")
        except Exception as e:
            print(f"Erro ao deletar coleção: {e}")
        
        if os.path.exists(settings.chroma_persist_directory):
            shutil.rmtree(settings.chroma_persist_directory)
            print(f"Diretório '{settings.chroma_persist_directory}' removido completamente!")
        else:
            print(f"Diretório '{settings.chroma_persist_directory}' não existe")
        
        print("ChromaDB limpo com sucesso!")
        print("Você pode recarregar os documentos agora.")
        
    except Exception as e:
        print(f"Erro ao limpar ChromaDB: {e}")

if __name__ == "__main__":
    clear_chromadb()
