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
        self.session = None  # Sessão ativa do Phoenix (interface web)
        self.is_enabled = False  # Status: Phoenix está funcionando?
        self.project_name = "rag-evaluation-system"  # Nome do projeto no Phoenix
        self.setup_phoenix()  # Inicializa tudo automaticamente
    
    def setup_phoenix(self):
        """Configurar e iniciar Phoenix"""
        try:
            logging.getLogger("phoenix").setLevel(logging.INFO)
            
            self.session = px.launch_app(
                port=6006,
                host="127.0.0.1"
            )
            
            self._setup_opentelemetry()
            
            self._setup_instrumentation()
            
            self.is_enabled = True
            print(f"Phoenix iniciado em: {self.session.url}")
            
        except Exception as e:
            print(f"Erro ao iniciar Phoenix: {str(e)}")
            self.is_enabled = False
    
    def _setup_opentelemetry(self):
        """Configurar OpenTelemetry para Phoenix"""
        try:
            # Configurar tracer provider
            tracer_provider = trace_sdk.TracerProvider()
            trace.set_tracer_provider(tracer_provider)
            
            # Configurar exporter para Phoenix
            phoenix_endpoint = f"http://127.0.0.1:6006/v1/traces"
            exporter = OTLPSpanExporter(endpoint=phoenix_endpoint)
            
            # Configurar span processor
            span_processor = BatchSpanProcessor(exporter)
            tracer_provider.add_span_processor(span_processor)
            
        except Exception as e:
            print(f"Erro na configuração OpenTelemetry: {str(e)}")
    
    def _setup_instrumentation(self):
        """Configurar instrumentação automática"""
        try:
            # Instrumentar LangChain
            if not hasattr(self, '_langchain_instrumented'):
                LangChainInstrumentor().instrument()
                self._langchain_instrumented = True
            
            # Instrumentar OpenAI  
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
    
    def get_traces_data(self, limit: int = 100) -> Dict[str, Any]:
        """Recupera dados de traces do Phoenix"""
        if not self.is_enabled:
            return {"error": "Phoenix não está habilitado"}
        
        try:
            # Phoenix oferece APIs para acessar dados de trace
            # Por enquanto, retornamos informações básicas
            return {
                "phoenix_enabled": True,
                "dashboard_url": self.get_phoenix_url(),
                "project_name": self.project_name,
                "traces_available": True,
                "message": "Acesse o dashboard Phoenix para ver traces detalhados"
            }
            
        except Exception as e:
            return {"error": f"Erro ao acessar dados de traces: {str(e)}"}
    
    def shutdown(self):
        """Finalizar Phoenix session"""
        try:
            if self.session:
                # Phoenix geralmente não precisa shutdown manual
                # mas podemos limpar recursos se necessário
                print("🔥 Phoenix session finalizada")
                self.is_enabled = False
        except Exception as e:
            print(f"Erro ao finalizar Phoenix: {str(e)}")

# Instância global do serviço Phoenix
phoenix_service = PhoenixService()