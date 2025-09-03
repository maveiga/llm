from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_directory: str = "./chroma_data"
    
    class Config:
        env_file = ".env"

settings = Settings()