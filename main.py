import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from app.routes import admin, chat, evaluation
from app.core.config import settings
from app.services.database_service import database_service
from app.services.phoenix_service import phoenix_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Inicializando aplicação RAG...")
    
    try:
        print("Criando tabelas do banco de dados...")
        await database_service.create_tables()
        print("Banco de dados inicializado")
        
        # Phoenix pode falhar sem quebrar a aplicação
        if phoenix_service.is_enabled:
            print(f"Phoenix dashboard disponível em: {phoenix_service.get_phoenix_url()}")
        else:
            print("Phoenix não foi inicializado - continuando sem observabilidade")
        
        print("Aplicação RAG inicializada com sucesso!")
        
    except Exception as e:
        print(f"Erro durante inicialização: {str(e)}")
        print("Continuando mesmo com erros de inicialização...")
    
    yield
    
    print("Finalizando aplicação...")
    try:
        if phoenix_service.is_enabled:
            phoenix_service.shutdown()
            print("Phoenix finalizado")
    except Exception as e:
        print(f"Erro durante finalização: {str(e)}")
    
    print("Aplicação finalizada")

app = FastAPI(
    title="RAG System with Phoenix Observability + RAGAS Evaluation",
    description="Advanced RAG system with Phoenix tracing, RAGAS quality metrics, and comprehensive observability",
    version="2.0.0",
    lifespan=lifespan
)

app.include_router(admin.router, prefix="/api/v1", tags=["admin"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(evaluation.router, prefix="/api/v1", tags=["evaluation"])

@app.get("/")
async def root():
    return {"message": "RAG Document API is running"}


if __name__ == "__main__":
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)
    
    if not is_docker:
        print("Rodando localmente - iniciando servidor Uvicorn")
        import uvicorn
        import subprocess
        import spacy

        try:
            spacy.load("pt_core_news_lg")
            print("Modelo spaCy já está instalado")
        except OSError:
            print("Baixando modelo spaCy pt_core_news_lg")
            subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_lg"], check=True)
        
        #inicia o chroma em paralelo
        subprocess.Popen([
            "chroma",
            "run",
            "--host", "localhost",
            "--port", "8001",
            "--path", "./chroma_data"
        ])
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, loop="asyncio")
    else:
        print("Rodando dentro do Docker - inicie com docker-compose up --build")