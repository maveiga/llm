from pydantic_settings import BaseSettings
from pydantic import Field
import os
class Settings(BaseSettings):
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "documents"
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    chroma_persist_directory: str = "./chroma_data"
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    
    class Config:
        env_file = ".env"

settings = Settings()