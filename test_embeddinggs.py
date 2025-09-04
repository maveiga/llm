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

# --- CONFIGURAÇÃO INICIAL ---

# 1. Carrega o modelo do spaCy para processamento de sentenças
print("Carregando modelo spaCy...")
nlp = spacy.load("pt_core_news_sm")


# 2. Carrega o modelo e tokenizador da Hugging Face para perplexidade
# Usaremos um modelo treinado em português para melhores resultados.
# 'pierreguillou/gpt2-small-portuguese' é uma boa opção.
print("Carregando modelo da Hugging Face (pode levar um tempo na primeira vez)...")
model_name = "pierreguillou/gpt2-small-portuguese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Modelos carregados com sucesso!")

# 3. Define o limiar de perplexidade.
# Este valor é EXPERIMENTAL. Valores altos indicam frases "estranhas".
# Comece com um valor entre 1000 e 5000 e ajuste conforme os resultados.
LIMIAR_PERPLEXIDADE = 600 

# --- FUNÇÃO PARA DETECTAR FRASES SUSPEITAS COM LIMIAR DINÂMICO ---
def detectar_suspeitas(sentencas_com_ppl, percentil=90):
    """
    Recebe uma lista de tuplas (sentenca, ppl) e retorna as frases
    cuja perplexidade está acima do limiar dinâmico.
    
    Args:
        sentencas_com_ppl (list): lista de tuplas (texto, perplexidade)
        percentil (int): percentil usado para definir o limiar (default=90)
    
    Returns:
        list: lista de dicionários com frases suspeitas
    """
    if not sentencas_com_ppl:
        return []

    # Extrai só os valores de perplexidade
    ppls = [ppl for _, ppl in sentencas_com_ppl]

    # Calcula o limiar baseado no percentil
    limiar_dinamico = np.percentile(ppls, percentil)
    print(f"\n[LIMIAR DINÂMICO] Usando percentil {percentil}: {limiar_dinamico:.2f}\n")

    # Coleta frases acima do limiar
    frases_suspeitas = []
    for sentenca, ppl in sentencas_com_ppl:
        if ppl > limiar_dinamico:
            frases_suspeitas.append({"texto": sentenca, "perplexidade": ppl})

    return frases_suspeitas


# --- FUNÇÃO PARA CALCULAR PERPLEXIDADE ---

def calcular_perplexidade(sentenca, model, tokenizer):
    """Calcula a perplexidade de uma única sentença."""
    if not sentenca.strip():
        return 0.0
        
    inputs = tokenizer(sentenca, return_tensors="pt") #converte a sentença em números.
    
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

# --- PROCESSAMENTO DOS ARQUIVOS ---

# Define o diretório onde estão os arquivos
diretorio = Path("conteudo_ficticio")

# Encontra todos os arquivos .txt no diretório
arquivos_txt = list(diretorio.glob("*.txt"))

if not arquivos_txt:
    print("Nenhum arquivo .txt encontrado no diretório.")
else:
    print(f"\nEncontrados {len(arquivos_txt)} arquivo(s) .txt para processar.")
    print("-" * 60)
    
    for arquivo in arquivos_txt:
        print(f"\nProcessando arquivo: {arquivo.name}")
        print("-" * 40)
        
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                texto = f.read().strip()
            
            if not texto:
                print("Arquivo vazio. Pulando...")
                continue
            
            # Processa o texto com spaCy (COM ACENTOS!)
            doc = nlp(texto)
            
            frases_boas = []
            frases_suspeitas = []

            #sent.text não é o texto inteiro – cada sent é uma sentença detectada pelo SpaCy,
            # ou seja, um pedaço do texto delimitado por pontuação ou regras gramaticais.
            
            # Itera sobre as SENTENÇAS identificadas pelo spaCy
            for sent in doc.sents:
                sentenca_texto = sent.text.strip() #pega a sentença como string e strip remove espaços em branco
                if len(sentenca_texto.split()) < 3: # Ignora sentenças muito curtas
                    continue

                # Calcula a perplexidade da sentença
                ppl = calcular_perplexidade(sentenca_texto, model, tokenizer)
                
                # Compara com o limiar

                EXCECOES = [
                    "O colaborador deve enviar documentos via plataforma digital.",
                    "Clientes negativados devem quitar dívidas anteriores antes de nova análise."
                ]
                

                frases_suspeitas.append({"texto": sentenca_texto, "perplexidade": ppl})


            # Exibe os resultados para este arquivo
            print(f"Total de sentenças analisadas: {len(frases_boas) + len(frases_suspeitas)}")
            #---------------------------------------------------------------------------------------------------------#
            #sentencas_com_ppl = [(frase["texto"], frase["perplexidade"]) for frase in frases_boas + frases_suspeitas]
            #suspeitas_dinamicas = detectar_suspeitas(sentencas_com_ppl, percentil=90)
            #for frase in suspeitas_dinamicas:
                #print(f"[!] Suspeita: {frase['texto']} | PPL: {frase['perplexidade']:.2f}")

            


            if frases_suspeitas:
                print(f"\n--- [!] {len(frases_suspeitas)} SENTENÇA(S) SUSPEITA(S) ENCONTRADA(S) ---")
                for frase in frases_suspeitas:
                    print(f"  - Perplexidade: {frase['perplexidade']:.2f} | Frase: '{frase['texto']}'")
            else:
                print("\nNenhuma sentença suspeita encontrada neste arquivo.")
            
            # Você pode então decidir o que fazer. Por exemplo, salvar apenas as frases boas.
            texto_limpo = " ".join([frase['texto'] for frase in frases_boas])
            
            # Aqui você salvaria o texto_limpo em um novo arquivo, se quisesse
            # with open(f"limpo_{arquivo.name}", "w", encoding='utf-8') as f_out:
            #     f_out.write(texto_limpo)


            print(f"\nFinalizado: {arquivo.name}")
            print("-" * 40)
            
        except Exception as e:
            print(f"Erro ao processar o arquivo {arquivo.name}: {str(e)}")
            continue




print("\nProcessamento de todos os arquivos concluído!")