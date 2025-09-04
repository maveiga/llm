"""
Script de teste para verificar a integração Phoenix + RAGAS
Execute este script após instalar as dependências para testar o sistema completo
"""

import asyncio
import time
from app.services.rag_service import RAGService
from app.services.ragas_service import ragas_service
from app.services.phoenix_service import phoenix_service
from app.services.database_service import database_service

# Perguntas de teste específicas para Phoenix
PHOENIX_TEST_QUESTIONS = [
    "Como funciona a política de crédito?",
    "Quais são os critérios de aprovação?",
    "Como é feita a verificação cadastral?",
    "Qual documentação é necessária?",
    "Como calcular o limite de crédito?"
]

async def test_phoenix_initialization():
    """Testa se Phoenix foi inicializado corretamente"""
    print("🔥 Testando inicialização do Phoenix...")
    
    if phoenix_service.is_enabled:
        print(f"✅ Phoenix ativo: {phoenix_service.get_phoenix_url()}")
        return True
    else:
        print("❌ Phoenix não foi inicializado")
        print("💡 Dicas:")
        print("   - Verifique se as dependências foram instaladas: pip install arize-phoenix")
        print("   - Verifique se a porta 6006 está disponível")
        print("   - Reinicie o servidor após instalar dependências")
        return False

async def test_instrumentation():
    """Testa se a instrumentação está funcionando"""
    print("\n🔧 Testando instrumentação...")
    
    try:
        # Inicializar serviços
        await database_service.create_tables()
        rag_service = RAGService()
        
        # Fazer uma consulta de teste
        print("📝 Executando consulta de teste...")
        response = await rag_service.ask_question(
            question="Teste de instrumentação Phoenix",
            max_documents=2,
            save_interaction=True
        )
        
        if "interaction_id" in response:
            print(f"✅ Consulta executada com ID: {response['interaction_id']}")
            print("✅ Instrumentação funcionando (dados enviados para Phoenix)")
            return response['interaction_id']
        else:
            print("⚠️  Consulta executada mas sem ID de interação")
            return None
            
    except Exception as e:
        print(f"❌ Erro na instrumentação: {str(e)}")
        return None

async def test_rag_with_phoenix_traces():
    """Testa múltiplas consultas RAG gerando traces no Phoenix"""
    print("\n📊 Gerando traces no Phoenix...")
    
    rag_service = RAGService()
    interaction_ids = []
    
    for i, question in enumerate(PHOENIX_TEST_QUESTIONS, 1):
        print(f"🔍 Pergunta {i}/5: {question[:40]}...")
        
        try:
            start_time = time.time()
            response = await rag_service.ask_question(
                question=question,
                max_documents=3,
                save_interaction=True
            )
            
            response_time = time.time() - start_time
            
            if "interaction_id" in response:
                interaction_ids.append(response["interaction_id"])
                print(f"   ✅ Trace gerado: {response_time:.2f}s")
            else:
                print(f"   ⚠️  Sem trace: {question[:30]}...")
                
        except Exception as e:
            print(f"   ❌ Erro: {str(e)}")
    
    print(f"\n📈 Total de traces gerados: {len(interaction_ids)}")
    return interaction_ids

async def test_ragas_evaluation():
    """Testa avaliação RAGAS das interações"""
    print("\n🧮 Testando avaliação RAGAS...")
    
    try:
        # Executar avaliação nas últimas interações
        results = await ragas_service.evaluate_interactions(limit=3)
        
        if "error" in results:
            print(f"❌ Erro RAGAS: {results['error']}")
            return False
        
        print("✅ Avaliação RAGAS concluída:")
        print(f"   📊 Interações avaliadas: {results.get('total_interactions', 0)}")
        
        if 'average_scores' in results:
            scores = results['average_scores']
            print(f"   🎯 Faithfulness: {scores.get('faithfulness', 0):.3f}")
            print(f"   📝 Answer Relevancy: {scores.get('answer_relevancy', 0):.3f}")
            print(f"   🔍 Context Precision: {scores.get('context_precision', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na avaliação RAGAS: {str(e)}")
        return False

async def test_combined_report():
    """Testa relatório combinado Phoenix + RAGAS"""
    print("\n📋 Testando relatório combinado...")
    
    try:
        combined_report = await ragas_service.generate_phoenix_ragas_report()
        
        if "error" in combined_report:
            print(f"❌ Erro no relatório: {combined_report['error']}")
            return False
        
        print("✅ Relatório combinado gerado:")
        
        # Status da integração
        integration = combined_report.get('integration_status', {})
        print(f"   🔥 Phoenix ativo: {integration.get('phoenix_active', False)}")
        print(f"   📊 RAGAS disponível: {integration.get('ragas_data_available', False)}")
        print(f"   🔗 Análise combinada: {integration.get('combined_analysis', 'N/A')}")
        
        # Recomendações
        recommendations = combined_report.get('recommendations', [])
        if recommendations:
            print("   💡 Recomendações:")
            for rec in recommendations[:3]:  # Primeiras 3 recomendações
                print(f"      {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no relatório combinado: {str(e)}")
        return False

async def display_dashboard_info():
    """Exibe informações para acessar os dashboards"""
    print("\n🎯 Informações dos Dashboards:")
    print("=" * 50)
    
    if phoenix_service.is_enabled:
        print(f"🔥 Phoenix Dashboard: {phoenix_service.get_phoenix_url()}")
        print("   Funcionalidades:")
        print("   • Traces em tempo real das consultas RAG")
        print("   • Visualização de embeddings (UMAP/t-SNE)")
        print("   • Análise de performance e latência")
        print("   • Detecção de anomalias")
    else:
        print("❌ Phoenix Dashboard: Não disponível")
    
    print(f"\n📊 API Swagger: http://localhost:8000/docs")
    print("   Endpoints de avaliação:")
    print("   • /api/v1/evaluation/phoenix/status")
    print("   • /api/v1/evaluation/ragas/evaluate")
    print("   • /api/v1/evaluation/combined-report")

async def main():
    """Função principal do teste"""
    print("🚀 TESTE DE INTEGRAÇÃO PHOENIX + RAGAS")
    print("=" * 60)
    
    try:
        # Teste 1: Phoenix
        phoenix_ok = await test_phoenix_initialization()
        
        # Teste 2: Instrumentação
        test_interaction_id = await test_instrumentation()
        
        # Teste 3: Gerar traces
        interaction_ids = await test_rag_with_phoenix_traces()
        
        # Aguardar um momento para os traces serem processados
        if interaction_ids:
            print("\n⏳ Aguardando processamento dos traces...")
            await asyncio.sleep(2)
        
        # Teste 4: RAGAS
        ragas_ok = await test_ragas_evaluation()
        
        # Teste 5: Relatório combinado
        report_ok = await test_combined_report()
        
        # Resumo
        print("\n" + "=" * 60)
        print("📋 RESUMO DOS TESTES:")
        print(f"   🔥 Phoenix:         {'✅' if phoenix_ok else '❌'}")
        print(f"   🔧 Instrumentação:  {'✅' if test_interaction_id else '❌'}")
        print(f"   📈 Traces gerados:  {len(interaction_ids)} consultas")
        print(f"   🧮 RAGAS:          {'✅' if ragas_ok else '❌'}")
        print(f"   📋 Relatório:       {'✅' if report_ok else '❌'}")
        
        # Informações dos dashboards
        await display_dashboard_info()
        
        if phoenix_ok and ragas_ok:
            print("\n🎉 INTEGRAÇÃO PHOENIX + RAGAS FUNCIONANDO!")
            print("\n🚀 Próximos passos:")
            print("   1. Execute: python main.py")
            print("   2. Faça consultas RAG em /api/v1/chat/")
            print("   3. Monitore no Phoenix dashboard")
            print("   4. Avalie qualidade com RAGAS")
        else:
            print("\n⚠️  Alguns componentes apresentaram problemas")
            print("   Verifique os logs acima para detalhes")
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())