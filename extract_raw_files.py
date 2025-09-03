# -*- coding: utf-8 -*-
"""
Script para ler todos os arquivos .txt do diretório conteudo_ficticio
e salvar em um único arquivo texto.
"""

import os
import glob
from datetime import datetime

def extract_raw_files():
    # Diretório de origem (relativo ao script)
    source_dir = "conteudo_ficticio"
    
    # Arquivo de saída
    output_file = "raw_files_combined.txt"
    
    # Verificar se o diretório existe
    if not os.path.exists(source_dir):
        print(f"Diretório não encontrado: {source_dir}")
        return
    
    # Buscar todos os arquivos .txt
    txt_files = glob.glob(os.path.join(source_dir, "*.txt"))
    
    if not txt_files:
        print("Nenhum arquivo .txt encontrado no diretório")
        return
    
    print(f"Encontrados {len(txt_files)} arquivos .txt")
    
    # Criar arquivo de saída
    with open(output_file, 'w', encoding='utf-8') as output:
        # Cabeçalho
        output.write(f"=== ARQUIVOS RAW COMBINADOS ===\n")
        output.write(f"Data de extração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output.write(f"Total de arquivos: {len(txt_files)}\n")
        output.write(f"Diretório origem: {source_dir}\n")
        output.write("="*50 + "\n\n")
        
        # Processar cada arquivo
        for i, file_path in enumerate(txt_files, 1):
            filename = os.path.basename(file_path)
            print(f"Processando ({i}/{len(txt_files)}): {filename}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Escrever separador e conteúdo
                output.write(f"\n--- ARQUIVO {i}: {filename} ---\n")
                output.write(f"Tamanho: {len(content)} caracteres\n")
                output.write("-" * 40 + "\n")
                output.write(content)
                output.write("\n" + "="*50 + "\n")
                
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
                output.write(f"\n--- ERRO NO ARQUIVO: {filename} ---\n")
                output.write(f"Erro: {str(e)}\n")
                output.write("="*50 + "\n")
    
    print(f"\nArquivos combinados salvos em: {output_file}")
    
    # Mostrar estatísticas
    if os.path.exists(output_file):
        size = os.path.getsize(output_file)
        print(f"Tamanho do arquivo final: {size:,} bytes")

if __name__ == "__main__":
    extract_raw_files()