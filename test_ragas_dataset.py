"""
Script de teste para criar um dataset de exemplo e testar o RAGAS
Execute este script após instalar as dependências para testar o sistema
"""

import asyncio
import json
from app.services.rag_service import RAGService
from app.services.ragas_service import ragas_service
from app.services.database_service import database_service

# Dataset de perguntas de teste
TEST_QUESTIONS = [
    "Qual é a política de crédito da empresa?",
    "Como funciona o processo de verificação cadastral?",
    "Quais documentos são necessários para solicitar um empréstimo?",
    "Qual é o limite de crédito baseado na renda?",
    "Como é feita a análise de risco?",
    "Quais são os critérios para aprovação de crédito?",
    "Como funciona a integração com bureaus externos?",
    "Que tipo de comprovantes são aceitos?",
    "Qual é o processo de aprovação de empréstimos?",
    "Como é calculado o score de crédito?"
]

async def create_test_dataset():
    """Cria um dataset de teste executando as perguntas no sistema RAG"""
    print("🚀 Criando dataset de teste...")
    
    # Inicializar serviços
    await database_service.create_tables()
    rag_service = RAGService()
    
    interactions_created = []
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"📝 Processando pergunta {i}/{len(TEST_QUESTIONS)}: {question[:50]}...")
        
        try:
            # Executar pergunta no sistema RAG
            response = await rag_service.ask_question(
                question=question,
                max_documents=3,
                save_interaction=True
            )
            
            if "interaction_id" in response:
                interactions_created.append(response["interaction_id"])
                print(f"✅ Interação criada: {response['interaction_id']}")
            else:
                print(f"⚠️  Sem ID de interação para: {question[:30]}...")
                
        except Exception as e:
            print(f"❌ Erro ao processar '{question[:30]}...': {str(e)}")
    
    print(f"\n✨ Dataset criado com {len(interactions_created)} interações!")
    return interactions_created

async def run_ragas_evaluation(interaction_ids):
    """Executa avaliação RAGAS no dataset criado"""
    print("\n🔍 Executando avaliação RAGAS...")
    
    try:
        results = await ragas_service.evaluate_interactions(
            interaction_ids=interaction_ids[:5]  # Avaliar apenas as primeiras 5 para teste
        )
        
        if "error" in results:
            print(f"❌ Erro na avaliação: {results['error']}")
            return
        
        print("📊 Resultados da Avaliação RAGAS:")
        print(f"   Total de interações avaliadas: {results['total_interactions']}")
        print("\n📈 Scores Médios:")
        
        avg_scores = results['average_scores']
        for metric, score in avg_scores.items():
            print(f"   {metric}: {score:.3f}")
        
        # Mostrar alguns exemplos individuais
        if results['individual_scores']:
            print("\n🔍 Exemplos de Scores Individuais:")
            for i, score in enumerate(results['individual_scores'][:3]):
                print(f"   Pergunta {i+1}: {score['question']}")
                print(f"   - Faithfulness: {score.get('faithfulness', 'N/A')}")
                print(f"   - Answer Relevancy: {score.get('answer_relevancy', 'N/A')}")
                print(f"   - Context Precision: {score.get('context_precision', 'N/A')}")
                print()
        
    except Exception as e:
        print(f"❌ Erro durante avaliação: {str(e)}")

async def get_quality_report():
    """Gera e exibe relatório de qualidade"""
    print("\n📋 Gerando Relatório de Qualidade...")
    
    try:
        report = await ragas_service.get_quality_report(days=1)  # Último dia
        
        if "error" in report:
            print(f"⚠️  {report['error']}")
            return
        
        print(f"📊 Relatório: {report['period']}")
        print(f"   Total de interações: {report['total_interactions']}")
        
        print("\n📈 Métricas por Categoria:")
        for metric, stats in report['metrics'].items():
            if stats['count'] > 0:
                print(f"   {metric}:")
                print(f"   - Média: {stats['mean']:.3f}")
                print(f"   - Min/Max: {stats['min']:.3f} / {stats['max']:.3f}")
                print(f"   - Amostras: {stats['count']}")
        
        overall = report['overall_quality']['average_score']
        print(f"\n🎯 Score Geral de Qualidade: {overall:.3f}")
        
    except Exception as e:
        print(f"❌ Erro ao gerar relatório: {str(e)}")

async def main():
    """Função principal do teste"""
    print("🧪 Teste do Sistema RAGAS")
    print("=" * 50)
    
    try:
        # Etapa 1: Criar dataset de teste
        interaction_ids = await create_test_dataset()
        
        if not interaction_ids:
            print("❌ Nenhuma interação foi criada. Verifique o sistema RAG.")
            return
        
        # Etapa 2: Executar avaliação RAGAS
        await run_ragas_evaluation(interaction_ids)
        
        # Etapa 3: Gerar relatório de qualidade
        await get_quality_report()
        
        print("\n✅ Teste concluído com sucesso!")
        print("\n💡 Próximos passos:")
        print("   1. Execute o servidor: python main.py")
        print("   2. Acesse os endpoints de avaliação em /api/v1/evaluation/")
        print("   3. Use /docs para explorar a API")
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())