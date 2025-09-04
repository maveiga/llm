import chromadb
import os
import numpy as np
from datetime import datetime
from app.core.config import settings


def export_for_embedding_projector():
    try:
        # Criar pasta de saída
        output_dir = "embedding_export"
        os.makedirs(output_dir, exist_ok=True)

        embeddings_file = os.path.join(output_dir, "embeddings.tsv")
        metadata_file = os.path.join(output_dir, "metadata.tsv")

        print("Conectando ao ChromaDB...")
        client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        collection = client.get_or_create_collection(name=settings.chroma_collection_name)

        print(f"Exportando da coleção: {collection.name}")

        # Buscar dados (pode ser pesado se tiver muitos registros)
        data = collection.get(include=["embeddings", "documents", "metadatas"])
        embeddings = data.get("embeddings", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        # CORREÇÃO: Verificar se embeddings está vazio corretamente
        if embeddings is None or len(embeddings) == 0:
            print("Nenhum embedding encontrado na coleção.")
            return

        print(f"Total de embeddings: {len(embeddings)}")

        # Converter para array numpy se necessário
        if isinstance(embeddings, list):
            embeddings_array = np.array(embeddings)
        else:
            embeddings_array = embeddings

        # Salvar embeddings em TSV
        print(f"Salvando embeddings em {embeddings_file} ...")
        np.savetxt(embeddings_file, embeddings_array, delimiter="\t")

        # Salvar metadados em TSV
        print(f"Salvando metadata em {metadata_file} ...")
        with open(metadata_file, "w", encoding="utf-8") as f:
            # Escrever cabeçalho se houver metadados
            if metadatas and len(metadatas) > 0:
                # Extrair nomes das colunas do primeiro item de metadados
                first_meta = metadatas[0] if metadatas else {}
                headers = list(first_meta.keys()) if first_meta else []
                if headers:
                    f.write("\t".join(headers) + "\tdocument\n")
            
            # Escrever dados
            for i, doc in enumerate(documents):
                line_parts = []
                
                # Adicionar metadados se existirem
                if metadatas and i < len(metadatas):
                    meta = metadatas[i] or {}
                    for header in (headers if 'headers' in locals() else meta.keys()):
                        value = meta.get(header, "")
                        line_parts.append(str(value).replace("\t", " ").replace("\n", " "))
                
                # Adicionar documento
                doc_text = doc.replace("\n", " ").replace("\t", " ") if isinstance(doc, str) else str(doc)
                line_parts.append(doc_text)
                
                f.write("\t".join(line_parts) + "\n")

        print("Exportação concluída!")
        print(f"Arquivos prontos na pasta: {output_dir}")

    except Exception as e:
        # CORREÇÃO: Remover caracteres Unicode que causam erro de encoding
        print(f"Erro ao exportar embeddings: {str(e)}")


if __name__ == "__main__":
    export_for_embedding_projector()