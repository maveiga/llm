# RAG Document API

API para busca e recuperação de documentos usando embeddings vetoriais com FastAPI, ChromaDB e LangChain.

## Estrutura do Projeto

```
analista_dados_teste/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py          # Configurações da aplicação
│   ├── models/
│   │   ├── __init__.py
│   │   └── document.py        # Modelos Pydantic
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vector_service.py  # Serviço ChromaDB
│   │   └── document_processor.py  # Processamento com LangChain
│   ├── controllers/
│   │   ├── __init__.py
│   │   └── document_controller.py  # Lógica de negócio
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── documents.py       # Rotas de documentos
│   │   ├── search.py          # Rotas de busca
│   │   └── admin.py           # Rotas administrativas
│   └── __init__.py
├── conteudo_ficticio/         # Dados de teste
├── main.py                    # Ponto de entrada
├── requirements.txt           # Dependências Python
├── Dockerfile                 # Configuração Docker
├── docker-compose.yml         # Orquestração de contêineres
└── README.md                  # Esta documentação
```

## Funcionalidades

- **API RESTful** com FastAPI seguindo padrão MVC
- **Busca vetorial** usando ChromaDB e embeddings
- **Processamento de documentos** com LangChain
- **Chunking inteligente** de textos longos
- **Filtros por categoria** nas buscas
- **Interface de administração** para carregar documentos

## Endpoints da API

### Documentos
- `POST /api/v1/documents` - Adicionar documento
- `GET /api/v1/documents/{doc_id}` - Buscar documento por ID

### Busca
- `POST /api/v1/search` - Buscar documentos similares

### Administração
- `POST /api/v1/admin/load-documents` - Carregar documentos do diretório
- `GET /api/v1/admin/collection-info` - Informações da coleção

## Instalação e Execução

### Opção 1: Com Docker (Recomendado)

#### Pré-requisitos
- Docker
- Docker Compose

#### Executar
```bash
# Clone/navegue até o diretório do projeto
cd analista_dados_teste

# Execute com docker-compose
docker-compose up --build

# A API estará disponível em http://localhost:8000
# ChromaDB estará em http://localhost:8001
```

### Opção 2: Sem Docker

#### Pré-requisitos
- Python 3.11+
- ChromaDB standalone

#### 1. Instalar ChromaDB standalone
```bash
# Instalar ChromaDB
pip install chromadb

# Executar ChromaDB server
chroma run --host localhost --port 8001 --path ./chroma_data
```

#### 2. Configurar ambiente Python
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

#### 3. Configurar variáveis de ambiente
```bash
# Copiar arquivo de exemplo
copy .env.example .env

# Editar .env com suas configurações
```

#### 4. Executar a aplicação
```bash
python main.py

# A API estará disponível em http://localhost:8000
```
```bash
  # python -m spacy download pt_core_news_lg ou pt_core_news_sm mais leve 
```
## Uso da API

### 1. Carregar documentos iniciais
```bash
curl -X POST "http://localhost:8000/api/v1/admin/load-documents"
```




### 2. Buscar documentos
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "política de crédito",
    "limit": 5,
    "category_filter": "Política de Crédito"
  }'
```

### 3. Adicionar documento manualmente
```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Novo Documento",
    "category": "Categoria Teste",
    "content": "Conteúdo do documento...",
    "metadata": {"author": "Sistema"}
  }'
```

## Documentação Interativa

Após executar a aplicação, acesse:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configurações

As configurações podem ser alteradas através de variáveis de ambiente ou arquivo `.env`:

- `CHROMA_HOST`: Host do ChromaDB (padrão: localhost)
- `CHROMA_PORT`: Porta do ChromaDB (padrão: 8001)
- `CHROMA_COLLECTION_NAME`: Nome da coleção (padrão: documents)
- `EMBEDDING_MODEL`: Modelo de embeddings (padrão: all-MiniLM-L6-v2)

## Processamento de Documentos

O sistema processa automaticamente os documentos em `conteudo_ficticio/`:

1. **Extração de metadados** do cabeçalho (Título, Categoria)
2. **Limpeza de conteúdo** removendo ruído textual
3. **Chunking inteligente** para textos longos
4. **Geração de embeddings** com Sentence Transformers
5. **Armazenamento vetorial** no ChromaDB

## Estrutura dos Dados

Os documentos seguem o formato:
```
Título: Nome do Documento
Categoria: Categoria Principal

Conteúdo do documento...
```

## Troubleshooting

### Erro de conexão com ChromaDB
- Verifique se o ChromaDB está executando na porta correta
- Confirme as variáveis de ambiente `CHROMA_HOST` e `CHROMA_PORT`

### Modelo de embeddings não encontrado
- O primeiro uso baixará o modelo automaticamente
- Certifique-se de ter conexão com internet

### Permissões no Docker
- No Windows, certifique-se de que o Docker Desktop está executando
- No Linux, adicione seu usuário ao grupo docker: `sudo usermod -aG docker $USER`

## Tecnologias Utilizadas

- **FastAPI**: Framework web moderno para Python
- **ChromaDB**: Base de dados vetorial
- **LangChain**: Framework para aplicações com LLM
- **Sentence Transformers**: Modelos de embeddings
- **Pydantic**: Validação de dados
- **Docker**: Containerização