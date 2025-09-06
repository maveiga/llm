import aiosqlite
import warnings
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.models.rag_interaction import Base
from app.core.config import settings
import os
import logging

# Suprimir warnings específicos do SQLAlchemy
warnings.filterwarnings("ignore", message=".*Skipped unsupported reflection.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sqlalchemy.*")

# Configurar logging SQLAlchemy para reduzir ruído
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# SQLite database path
DATABASE_PATH = os.path.join(settings.chroma_persist_directory, "rag_interactions.db")
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_PATH}"

# Async engine and session com configurações otimizadas
engine = create_async_engine(
    DATABASE_URL, 
    echo=False,  # Sem logs SQL
    future=True,  # Usar SQLAlchemy 2.0 style
    connect_args={
        "check_same_thread": False,  # Permitir uso em múltiplas threads
    }
)
AsyncSessionLocal = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

class DatabaseService:
    def __init__(self):
        self.database_path = DATABASE_PATH
        self.database_url = DATABASE_URL

    async def create_tables(self):
        """Criar todas as tabelas do banco de dados"""
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_session(self) -> AsyncSession:
        """Obter uma sessão assíncrona do banco de dados"""
        async with AsyncSessionLocal() as session:
            yield session


# Instância global do serviço de banco de dados
database_service = DatabaseService()