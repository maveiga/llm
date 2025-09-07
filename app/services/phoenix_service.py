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
            
            # Detectar se está rodando no Docker
            is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)
            
            if is_docker:
                print("🐳 Docker detectado - configurando Phoenix para Docker")
                # No Docker, precisa usar 0.0.0.0 para ser acessível externamente
                self.session = px.launch_app(
                    port=6006,
                    host="0.0.0.0"  # Permite acesso externo no Docker
                )
                print("🔥 Phoenix configurado para Docker!")
                print("📍 Acesse via: http://localhost:6006 (se port forwarding configurado)")
                print("📍 Ou via: http://host.docker.internal:6006 (no Windows/Mac)")
            else:
                print("🖥️ Ambiente local detectado - configurando Phoenix")
                # Configuração para ambiente local
                self.session = px.launch_app(
                    port=6006,
                    host="127.0.0.1"
                )
            
            self._setup_opentelemetry()
            
            self._setup_instrumentation()
            
            self.is_enabled = True
            print(f"🔥 Phoenix iniciado em: {self.session.url}")
            
        except Exception as e:
            print(f"⚠️ Phoenix não pôde ser iniciado: {str(e)}")
            print("💡 Sistema continuará funcionando sem observabilidade Phoenix")
            self.is_enabled = False
    
    def _setup_opentelemetry(self):
        """Configurar OpenTelemetry para Phoenix"""
        try:
            # Configurar tracer provider
            tracer_provider = trace_sdk.TracerProvider()
            trace.set_tracer_provider(tracer_provider)
            
            # Detectar ambiente para configurar endpoint correto
            is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)
            
            if is_docker:
                # No Docker, usar localhost interno
                phoenix_endpoint = f"http://localhost:6006/v1/traces"
            else:
                # Local, usar 127.0.0.1
                phoenix_endpoint = f"http://127.0.0.1:6006/v1/traces"
            
            print(f"🔧 OpenTelemetry endpoint: {phoenix_endpoint}")
            exporter = OTLPSpanExporter(endpoint=phoenix_endpoint)
            
            # Configurar span processor
            span_processor = BatchSpanProcessor(exporter)
            tracer_provider.add_span_processor(span_processor)
            
            print("✅ OpenTelemetry configurado com sucesso")
            
        except Exception as e:
            print(f"❌ Erro na configuração OpenTelemetry: {str(e)}")
    
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