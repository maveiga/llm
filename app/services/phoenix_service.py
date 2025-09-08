import os
import phoenix as px
from phoenix.trace import using_project
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np

class PhoenixService:
    """
    Serviço de observabilidade Phoenix para sistemas RAG
    
    O Phoenix é uma ferramenta que:
    1. Monitora todas as operações RAG em tempo real
    2. Visualiza embeddings e clusters de documentos
    3. Analisa performance e latência
    4. Detecta anomalias e drift nos dados
    5. Integra com ferramentas de avaliação como RAGAS
    """
    
    def __init__(self):
        self.session = None 
        self.is_enabled = False
        self.project_name = "rag-evaluation-system"
        self.setup_phoenix()
    
    def setup_phoenix(self):
        """Configurar e iniciar Phoenix"""
        try:
            is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)
            
            if is_docker:
                print("Docker detectado - configurando Phoenix para Docker")
                self.session = px.launch_app(
                    port=6006,
                    host="0.0.0.0"
                )
                print("Phoenix configurado para Docker!")
                print("Acesse via: http://localhost:6006")
            else:
                print("Ambiente local detectado - configurando Phoenix")
                self.session = px.launch_app(
                    port=6006,
                    host="127.0.0.1"
                )
            
            self._setup_opentelemetry()
            
            self._setup_instrumentation()
            
            self.is_enabled = True
            print(f"Phoenix iniciado em: {self.session.url}")
            
        except Exception as e:
            print(f"Phoenix não pôde ser iniciado: {str(e)}")
            print("Sistema continuará funcionando sem observabilidade Phoenix")
            self.is_enabled = False
    
    def _setup_opentelemetry(self):
        """Configurar OpenTelemetry para Phoenix"""
        try:
            tracer_provider = trace_sdk.TracerProvider()
            trace.set_tracer_provider(tracer_provider)
            
            is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)
            
            if is_docker:
                phoenix_endpoint = f"http://localhost:6006/v1/traces"
            else:
                phoenix_endpoint = f"http://127.0.0.1:6006/v1/traces"
            
            print(f"OpenTelemetry endpoint: {phoenix_endpoint}")
            exporter = OTLPSpanExporter(endpoint=phoenix_endpoint)
            
            span_processor = BatchSpanProcessor(exporter)
            tracer_provider.add_span_processor(span_processor)
            
            print("OpenTelemetry configurado com sucesso")
            
        except Exception as e:
            print(f"Erro na configuração OpenTelemetry: {str(e)}")
    
    def _setup_instrumentation(self):
        """Configurar instrumentação automática"""
        try:
            if not hasattr(self, '_langchain_instrumented'):
                LangChainInstrumentor().instrument()
                self._langchain_instrumented = True
            
            if not hasattr(self, '_openai_instrumented'):
                OpenAIInstrumentor().instrument()
                self._openai_instrumented = True
            
        except Exception as e:
            print(f"Erro na instrumentação: {str(e)}")
    
    def get_phoenix_url(self) -> Optional[str]:
        """Retorna URL do Phoenix dashboard"""
        if self.session and self.is_enabled:
            return self.session.url
        return None
    
    def shutdown(self):
        """Finalizar Phoenix session"""
        try:
            if self.session:
                print("Phoenix session finalizada")
                self.is_enabled = False
        except Exception as e:
            print(f"Erro ao finalizar Phoenix: {str(e)}")
phoenix_service = PhoenixService()