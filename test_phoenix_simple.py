"""
Script simples para testar Phoenix com traces manuais
"""

import asyncio
import phoenix as px
from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
import time

def setup_simple_phoenix():
    """Setup Phoenix simples"""
    try:
        # Inicializar Phoenix
        session = px.launch_app(port=6006)
        print(f"🔥 Phoenix dashboard: {session.url}")
        
        # Configurar OpenTelemetry
        tracer_provider = trace_sdk.TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        
        # Configurar exporter para Phoenix
        otlp_exporter = OTLPSpanExporter(
            endpoint="http://127.0.0.1:6006/v1/traces",
        )
        
        # Adicionar processors
        tracer_provider.add_span_processor(
            BatchSpanProcessor(otlp_exporter)
        )
        
        # Também mostrar no console para debug
        console_exporter = ConsoleSpanExporter()
        tracer_provider.add_span_processor(
            BatchSpanProcessor(console_exporter)
        )
        
        return session
        
    except Exception as e:
        print(f"Erro no setup Phoenix: {e}")
        return None

def create_manual_traces():
    """Criar traces manuais para testar"""
    tracer = trace.get_tracer(__name__)
    
    print("📝 Criando traces de teste...")
    
    for i in range(3):
        with tracer.start_as_current_span(f"rag_query_{i+1}") as span:
            # Simular consulta RAG
            span.set_attribute("query", f"Pergunta de teste {i+1}")
            span.set_attribute("query_type", "test")
            
            # Simular busca vetorial
            with tracer.start_as_current_span("vector_search") as search_span:
                search_span.set_attribute("documents_found", 5)
                search_span.set_attribute("search_time_ms", 150)
                time.sleep(0.1)  # Simular tempo de busca
            
            # Simular chamada LLM
            with tracer.start_as_current_span("llm_generation") as llm_span:
                llm_span.set_attribute("model", "gpt-3.5-turbo")
                llm_span.set_attribute("tokens_used", 200)
                llm_span.set_attribute("response_time_ms", 800)
                time.sleep(0.2)  # Simular tempo de LLM
            
            span.set_attribute("total_time_ms", 950)
            span.set_attribute("success", True)
            
        print(f"✅ Trace {i+1} enviado")
        time.sleep(0.5)  # Pausa entre traces

async def test_with_rag_service():
    """Testar com o RAG service real"""
    try:
        from app.services.rag_service import RAGService
        from app.services.database_service import database_service
        
        print("\n🔄 Testando com RAG Service real...")
        
        await database_service.create_tables()
        rag_service = RAGService()
        
        test_questions = [
            "Como funciona a política de crédito?",
            "Quais documentos são necessários?",
            "Como é calculado o limite de crédito?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"🔍 Pergunta {i}: {question}")
            
            response = await rag_service.ask_question(
                question=question,
                max_documents=3,
                save_interaction=True
            )
            
            if "interaction_id" in response:
                print(f"   ✅ Interaction ID: {response['interaction_id']}")
            else:
                print(f"   ⚠️  Sem interaction ID")
                
            time.sleep(1)  # Pausa entre consultas
            
    except Exception as e:
        print(f"❌ Erro no teste RAG: {e}")

async def main():
    print("🧪 TESTE PHOENIX SIMPLIFICADO")
    print("=" * 50)
    
    # Setup Phoenix
    session = setup_simple_phoenix()
    if not session:
        print("❌ Falha no setup Phoenix")
        return
    
    # Aguardar Phoenix inicializar
    print("⏳ Aguardando Phoenix inicializar...")
    time.sleep(3)
    
    # Criar traces manuais
    create_manual_traces()
    
    # Aguardar traces serem processados
    print("\n⏳ Aguardando traces serem enviados...")
    time.sleep(5)
    
    # Testar com RAG service
    await test_with_rag_service()
    
    print(f"\n🎯 ACESSE O DASHBOARD: {session.url}")
    print("\n📊 O que verificar no Phoenix:")
    print("   1. Aba 'Traces' - deve mostrar os traces criados")
    print("   2. Cada trace deve ter sub-spans (vector_search, llm_generation)")
    print("   3. Atributos devem estar visíveis nos spans")
    
    print(f"\n✨ Teste concluído! Dashboard: {session.url}")

if __name__ == "__main__":
    asyncio.run(main())