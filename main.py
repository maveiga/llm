from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import documents, search, admin, chat, evaluation
from app.core.config import settings
from app.services.database_service import database_service
from app.services.phoenix_service import phoenix_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Criar tabelas do banco de dados na inicialização
    await database_service.create_tables()
    
    # Phoenix já é inicializado automaticamente no import
    if phoenix_service.is_enabled:
        print(f"🔥 Phoenix dashboard disponível em: {phoenix_service.get_phoenix_url()}")
    else:
        print("⚠️  Phoenix não foi inicializado - continuando sem observabilidade")
    
    yield
    
    # Cleanup
    if phoenix_service.is_enabled:
        phoenix_service.shutdown()

app = FastAPI(
    title="RAG System with Phoenix Observability + RAGAS Evaluation",
    description="Advanced RAG system with Phoenix tracing, RAGAS quality metrics, and comprehensive observability",
    version="2.0.0",
    lifespan=lifespan
)

app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(admin.router, prefix="/api/v1", tags=["admin"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(evaluation.router, prefix="/api/v1", tags=["evaluation"])

@app.get("/")
async def root():
    return {"message": "RAG Document API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)