from sqlalchemy import Column, String, DateTime, Text, JSON, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel

Base = declarative_base()

class RAGInteractionDB(Base):
    __tablename__ = "rag_interactions"
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=func.now())
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    contexts = Column(JSON)
    sources = Column(JSON)
    user_feedback = Column(Integer)
    ragas_scores = Column(JSON)
    model_version = Column(String, default="gpt-3.5-turbo")
    embedding_model = Column(String, default="all-MiniLM-L6-v2")
    response_time = Column(Float)

class RAGInteractionCreate(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    sources: List[Dict]
    response_time: Optional[float] = None

class RAGInteractionResponse(BaseModel):
    id: str
    timestamp: datetime
    question: str
    answer: str
    contexts: List[str]
    sources: List[Dict]
    user_feedback: Optional[int]
    ragas_scores: Optional[Dict]
    model_version: str
    embedding_model: str
    response_time: Optional[float]

    class Config:
        from_attributes = True

class UserFeedback(BaseModel):
    rating: int

class RAGASEvaluation(BaseModel):
    interaction_ids: Optional[List[str]] = None
    include_ground_truth: bool = False