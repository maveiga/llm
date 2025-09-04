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
    def __init__(self):
        self.session = None
        self.is_enabled = False
        self.project_name = "rag-evaluation-system"
        self.setup_phoenix()
    
    def setup_phoenix(self):
        """Configurar e iniciar Phoenix"""
        try:
            # Configurar logging
            logging.getLogger("phoenix").setLevel(logging.INFO)
            
            # Inicializar Phoenix session
            self.session = px.launch_app(
                port=6006,  # Porta padrão do Phoenix
                host="127.0.0.1"
            )
            
            # Configurar OpenTelemetry para enviar dados para Phoenix
            self._setup_opentelemetry()
            
            # Configurar instrumentação
            self._setup_instrumentation()
            
            self.is_enabled = True
            print(f"🔥 Phoenix iniciado em: {self.session.url}")
            
        except Exception as e:
            print(f"⚠️  Erro ao iniciar Phoenix: {str(e)}")
            print("Continuando sem Phoenix...")
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
            
            print("✅ OpenTelemetry configurado para Phoenix")
            
        except Exception as e:
            print(f"⚠️  Erro na configuração OpenTelemetry: {str(e)}")
    
    def _setup_instrumentation(self):
        """Configurar instrumentação automática"""
        try:
            # Instrumentar LangChain
            if not hasattr(self, '_langchain_instrumented'):
                LangChainInstrumentor().instrument()
                self._langchain_instrumented = True
                print("✅ LangChain instrumentado com Phoenix")
            
            # Instrumentar OpenAI  
            if not hasattr(self, '_openai_instrumented'):
                OpenAIInstrumentor().instrument()
                self._openai_instrumented = True
                print("✅ OpenAI instrumentado com Phoenix")
            
        except Exception as e:
            print(f"⚠️  Erro na instrumentação: {str(e)}")
    
    def start_trace_session(self, session_name: str = None):
        """Iniciar uma sessão de trace com nome específico"""
        if not self.is_enabled:
            return None
        
        try:
            session_name = session_name or f"rag-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            return using_project(session_name)
        except Exception as e:
            print(f"Erro ao iniciar trace session: {str(e)}")
            return None
    
    def log_rag_interaction(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        sources: List[Dict],
        response_time: float,
        metadata: Optional[Dict] = None
    ):
        """Log manual de interação RAG para Phoenix"""
        if not self.is_enabled:
            return
        
        try:
            # Preparar dados para Phoenix (sem usar SpanAttributes por enquanto)
            interaction_data = {
                "input_value": question,
                "output_value": answer,
                "retrieval_documents": [
                    {
                        "document_content": context,
                        "document_id": f"doc_{i}",
                        "document_metadata": sources[i] if i < len(sources) else {}
                    }
                    for i, context in enumerate(contexts)
                ],
                "response_time_seconds": response_time,
                "num_documents_retrieved": len(contexts),
                "custom_metadata": metadata or {}
            }
            
            # Phoenix automaticamente captura através da instrumentação
            # Este método serve para logs adicionais se necessário
            
        except Exception as e:
            print(f"Erro ao logar interação no Phoenix: {str(e)}")
    
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
    
    def generate_embeddings_analysis(self) -> Dict[str, Any]:
        """Preparar dados para análise de embeddings no Phoenix"""
        if not self.is_enabled:
            return {"error": "Phoenix não está habilitado"}
        
        try:
            # Phoenix pode analisar embeddings automaticamente
            # quando instrumentado corretamente
            return {
                "analysis_available": True,
                "dashboard_url": self.get_phoenix_url(),
                "instructions": [
                    "1. Acesse o Phoenix dashboard",
                    "2. Vá para a aba 'Embeddings'",  
                    "3. Visualize clusters e similaridades",
                    "4. Analise retrieval patterns"
                ]
            }
            
        except Exception as e:
            return {"error": f"Erro na análise de embeddings: {str(e)}"}
    
    def export_traces_for_ragas(self, interaction_ids: List[str] = None) -> Dict[str, Any]:
        """Exportar dados de traces para integrar com RAGAS"""
        if not self.is_enabled:
            return {"traces": [], "message": "Phoenix não habilitado"}
        
        try:
            # Implementar exportação de dados Phoenix para RAGAS
            # Por enquanto, estrutura básica
            return {
                "traces_exported": len(interaction_ids) if interaction_ids else 0,
                "phoenix_dashboard": self.get_phoenix_url(),
                "integration_status": "available",
                "message": "Traces disponíveis no dashboard Phoenix"
            }
            
        except Exception as e:
            return {"error": f"Erro ao exportar traces: {str(e)}"}
    
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