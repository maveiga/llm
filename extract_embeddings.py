# -*- coding: utf-8 -*-
"""
Script para extrair todo o texto armazenado no ChromaDB (embeddings)
e salvar em um único arquivo texto.
"""

import chromadb
import os
from datetime import datetime
from app.core.config import settings

def extract_embeddings():
    # Arquivo de saída
    output_file = "embeddings_combined.txt"
    
    try:
        # Conectar ao ChromaDB
        print("Conectando ao ChromaDB...")
        client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        collection = client.get_or_create_collection(name=settings.chroma_collection_name)
        
        # Obter contagem total
        total_count = collection.count()
        print(f"Total de documentos no banco: {total_count}")
        
        if total_count == 0:
            print("Nenhum documento encontrado no ChromaDB")
            return
        
        # Obter todos os documentos
        print("Extraindo documentos...")
        results = collection.get(
            include=['documents', 'metadatas']
        )
        
        # Criar arquivo de saída
        with open(output_file, 'w', encoding='utf-8') as output:
            # Cabeçalho
            output.write(f"=== EMBEDDINGS EXTRAÍDOS DO CHROMADB ===\n")
            output.write(f"Data de extração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            output.write(f"Total de documentos: {len(results['ids'])}\n")
            output.write(f"Coleção: {settings.chroma_collection_name}\n")
            output.write(f"Diretório ChromaDB: {settings.chroma_persist_directory}\n")
            output.write("="*60 + "\n\n")
            
            # Processar cada documento
            for i, doc_id in enumerate(results['ids'], 1):
                metadata = results['metadatas'][i-1] if results['metadatas'] else {}
                document = results['documents'][i-1] if results['documents'] else ""
                
                title = metadata.get('title', 'Sem título')
                category = metadata.get('category', 'Sem categoria')
                
                print(f"Processando ({i}/{len(results['ids'])}): {title}")
                
                # Escrever informações do documento
                output.write(f"\n--- DOCUMENTO {i} ---\n")
                output.write(f"ID: {doc_id}\n")
                output.write(f"Título: {title}\n")
                output.write(f"Categoria: {category}\n")
                output.write(f"Tamanho: {len(document)} caracteres\n")
                
                # Mostrar metadados extras
                extra_metadata = {k: v for k, v in metadata.items() 
                                if k not in ['title', 'category']}
                if extra_metadata:
                    output.write(f"Metadados extras: {extra_metadata}\n")
                
                output.write("-" * 50 + "\n")
                output.write(document)
                output.write("\n" + "="*60 + "\n")
        
        print(f"\nEmbeddings extraídos e salvos em: {output_file}")
        
        # Mostrar estatísticas
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"Tamanho do arquivo final: {size:,} bytes")
            
    except Exception as e:
        print(f"Erro ao extrair embeddings: {e}")
        print(f"Verifique se o ChromaDB existe em: {settings.chroma_persist_directory}")

if __name__ == "__main__":
    # Adicionar o diretório atual ao path para importar módulos da app
    import sys
    sys.path.append('.')
    
    extract_embeddings()