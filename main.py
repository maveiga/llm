from fastapi import FastAPI
from app.routes import documents, search, admin
from app.core.config import settings

app = FastAPI(
    title="RAG Document API",
    description="API for document retrieval and search using vector embeddings",
    version="1.0.0"
)

app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(admin.router, prefix="/api/v1", tags=["admin"])

@app.get("/")
async def root():
    return {"message": "RAG Document API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)