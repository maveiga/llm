# Sistema RAG com Avaliação e Observabilidade

Este projeto implementa um sistema completo de *Retrieval-Augmented Generation* (RAG) em Python, projetado para responder a perguntas em linguagem natural com base em uma base de conhecimento de documentos. A solução é encapsulada em uma API RESTful e inclui ferramentas robustas para avaliação de qualidade e monitoramento de performance.

## Funcionalidades Principais

-   **Pipeline RAG Completo**: Orquestração de busca vetorial, formatação de contexto e geração de respostas com LLMs.
-   **API RESTful com FastAPI**: Endpoints para interação de chat, administração e avaliação.
-   **Busca Vetorial**: Utiliza **ChromaDB** e **Sentence-Transformers** para indexação e busca semântica.
-   **Avaliação de Qualidade**: Integra o framework **Ragas** para calcular métricas como `faithfulness` e `answer_relevancy`.
-   **Observabilidade**: Utiliza **Phoenix/OpenTelemetry** para monitoramento e *tracing* do pipeline, permitindo a análise de latência e o fluxo de dados.
-   **Processamento de Documentos**: Inclui uma etapa de pré-processamento para limpar e dividir os documentos em *chunks*.
-   **Containerização**: Fornecido com uma configuração **Docker** para facilitar o deploy e a execução dos serviços.

## Arquitetura e Tecnologias

O sistema é construído sobre uma arquitetura modular que desacopla as principais responsabilidades:

-   **Interface da API**: **FastAPI**
-   **Orquestração do RAG**: **LangChain**
-   **Modelo de Linguagem (LLM)**: **OpenAI**
-   **Banco de Dados Vetorial**: **ChromaDB**
-   **Geração de Embeddings**: **Sentence-Transformers**
-   **Avaliação**: **Ragas**
-   **Observabilidade**: **Phoenix**

O fluxo de trabalho principal é orquestrado pelo `RAGService`, que coordena a busca no `VectorService` e a geração de respostas no `LLMService`.

## Instalação e Execução

Você pode executar o projeto de duas maneiras: com Docker (recomendado) ou manualmente em um ambiente Python.

### Opção 1: Com Docker (Recomendado)

Este método gerencia todos os serviços (API, ChromaDB) automaticamente.

**Pré-requisitos**:
-   Docker
-   Docker Compose

**Passos**:

1.  **Clone o repositório** e navegue até o diretório do projeto.
2.  **Configure sua chave da OpenAI**: Copie `.env.example` para `.env` e adicione sua chave no campo `OPENAI_API_KEY`.
3.  **Inicie os serviços**:
    ```bash
    docker-compose up --build
    ```
4.  **Acesse os serviços**:
    -   **API**: `http://localhost:8000`
    -   **Documentação (Swagger)**: `http://localhost:8000/docs`
    -   **Dashboard do Phoenix**: `http://localhost:6006`

### Opção 2: Execução Manual

Este método requer a instalação manual das dependências e a execução separada dos serviços.

**Pré-requisitos**:
-   Python 3.9+
-   Um ambiente virtual (e.g., `venv`)

**Passos**:

1.  **Inicie o ChromaDB**: Em um terminal, inicie o servidor do ChromaDB.
    ```bash
    chroma run --host localhost --port 8001 --path ./chroma_data
    ```

2.  **Instale as dependências**: Em outro terminal, configure o ambiente Python.
    ```bash
    # Crie e ative um ambiente virtual
    python -m venv venv
    source venv/bin/activate  # ou venv\Scripts\activate no Windows

    # Instale as dependências
    pip install -r requirements.txt

    # Baixe o modelo de linguagem para o spaCy
    python -m spacy download pt_core_news_lg
    ```

3.  **Configure as variáveis de ambiente**: Copie `.env.example` para `.env` e adicione sua chave da OpenAI.

4.  **Execute a aplicação**:
    ```bash
    python main.py
    ```
    A API estará disponível em `http://localhost:8000`.

## Como Usar

### 1. Carregar Documentos

Adicione seus arquivos `.txt` ao diretório `conteudo_ficticio` e execute o seguinte comando para processá-los e indexá-los:

```bash
curl -X POST http://localhost:8000/api/v1/admin/load-documents
```

### 2. Fazer uma Pergunta

Envie uma pergunta para o endpoint de chat para receber uma resposta baseada nos documentos carregados.

```bash
curl -X POST http://localhost:8000/api/v1/chat/question \
-H "Content-Type: application/json" \
-d '{
  "question": "Qual é a política de compliance?",
  "max_documents": 5
}'
```

### 3. Avaliar a Qualidade

Execute uma avaliação com Ragas para medir a qualidade das respostas geradas. Os resultados serão salvos em chat_assistant.db na coluna ragas_score

```bash
curl -X GET http://localhost:8000/api/v1/evaluation/ragas/evaluete
```

### 4. Monitorar com Phoenix

Após interagir com a API, acesse o dashboard do Phoenix em `http://localhost:6006` para visualizar os *traces* de execução, analisar latências e depurar o pipeline.

### 5. Avaliar as respostas
Execute uma avaliação das respostas entregue pelo modelo

```bash
curl -X GET http://localhost:8000/api/v1/evaluation/interactions/{interaction_id}/feedback
```



