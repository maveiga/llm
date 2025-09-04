import aiosqlite
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.models.rag_interaction import Base
from app.core.config import settings
import os

# SQLite database path
DATABASE_PATH = os.path.join(settings.chroma_persist_directory, "rag_interactions.db")
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_PATH}"

# Async engine and session
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
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

    async def health_check(self) -> bool:
        """Verificar se o banco de dados está acessível"""
        try:
            async with AsyncSessionLocal() as session:
                await session.execute("SELECT 1")
                return True
        except Exception:
            return False

# Instância global do serviço de banco de dados
database_service = DatabaseService()