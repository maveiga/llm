# ARQUIVO: evaluation.py do Ragas - Comentado e Explicado em Português
# Este é o módulo principal de avaliação do framework Ragas

from __future__ import annotations

import typing as t
from uuid import UUID

from datasets import Dataset
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from langchain_core.language_models import BaseLanguageModel as LangchainLLM
from tqdm.auto import tqdm

# Importações específicas do Ragas
from ragas._analytics import track_was_completed
from ragas.callbacks import ChainType, RagasTracer, new_group
from ragas.dataset_schema import (
    EvaluationDataset,
    EvaluationResult,
    MultiTurnSample,
    SingleTurnSample,
)
from ragas.embeddings.base import (
    BaseRagasEmbeddings,
    LangchainEmbeddingsWrapper,
    embedding_factory,
)
from ragas.exceptions import ExceptionInRunner
from ragas.executor import Executor
from ragas.integrations.helicone import helicone_config
from ragas.llms import llm_factory
from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper
from ragas.metrics import AspectCritic
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics.base import (
    Metric,
    MetricWithEmbeddings,
    MetricWithLLM,
    ModeMetric,
    MultiTurnMetric,
    SingleTurnMetric,
)
from ragas.run_config import RunConfig
from ragas.utils import convert_v1_to_v2_dataset
from ragas.validation import (
    remap_column_names,
    validate_required_columns,
    validate_supported_metrics,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from ragas.cost import CostCallbackHandler, TokenUsageParser

# Nome padrão da cadeia de avaliação do Ragas
RAGAS_EVALUATION_CHAIN_NAME = "ragas evaluation"


@track_was_completed  # Decorator para análises de uso
def evaluate(
    dataset: t.Union[Dataset, EvaluationDataset],  # Dataset para avaliação
    metrics: t.Optional[t.Sequence[Metric]] = None,  # Métricas a usar
    llm: t.Optional[BaseRagasLLM | LangchainLLM] = None,  # Modelo de linguagem
    embeddings: t.Optional[BaseRagasEmbeddings | LangchainEmbeddings] = None,  # Embeddings
    experiment_name: t.Optional[str] = None,  # Nome do experimento
    callbacks: Callbacks = None,  # Callbacks do Langchain
    run_config: t.Optional[RunConfig] = None,  # Configuração de execução
    token_usage_parser: t.Optional[TokenUsageParser] = None,  # Parser de uso de tokens
    raise_exceptions: bool = False,  # Se deve levantar exceções
    column_map: t.Optional[t.Dict[str, str]] = None,  # Mapeamento de colunas
    show_progress: bool = True,  # Mostrar barra de progresso
    batch_size: t.Optional[int] = None,  # Tamanho do lote
    _run_id: t.Optional[UUID] = None,  # ID interno da execução
    _pbar: t.Optional[tqdm] = None,  # Barra de progresso personalizada
) -> EvaluationResult:
    """
    FUNÇÃO PRINCIPAL DE AVALIAÇÃO DO RAGAS
    =====================================
    
    Esta função é o coração do framework Ragas. Ela executa avaliações de sistemas RAG
    usando diferentes métricas de qualidade.
    
    O QUE FAZ:
    ----------
    1. Recebe um dataset com perguntas, respostas e contextos
    2. Aplica métricas de avaliação (fidelidade, relevância, etc.)
    3. Retorna pontuações quantitativas da qualidade do sistema RAG
    
    PARÂMETROS PRINCIPAIS:
    ---------------------
    - dataset: Dados para avaliar (perguntas, respostas, contextos)
    - metrics: Lista de métricas a aplicar (padrão: 4 métricas principais)
    - llm: Modelo de linguagem para gerar pontuações
    - embeddings: Modelo de embeddings para similaridade semântica
    
    RETORNA:
    --------
    EvaluationResult: Objeto com todas as pontuações e análises
    
    EXEMPLO DE USO BÁSICO:
    ---------------------
    ```python
    from ragas import evaluate
    
    # Seu dataset deve ter estas colunas:
    dataset = Dataset.from_dict({
        'question': ['Qual é a capital do Brasil?'],
        'answer': ['A capital do Brasil é Brasília.'],
        'contexts': [['Brasília é a capital federal do Brasil...']],
        'ground_truth': ['Brasília']
    })
    
    # Executa a avaliação
    result = evaluate(dataset)
    print(result)
    # Output: {'context_precision': 0.817, 'faithfulness': 0.892, ...}
    ```
    """
    
    # ===== FASE 1: INICIALIZAÇÃO E VALIDAÇÃO =====
    
    # Define valores padrão para parâmetros opcionais
    column_map = column_map or {}
    callbacks = callbacks or []
    run_config = run_config or RunConfig()

    # Configuração do Helicone (ferramenta de monitoramento)
    if helicone_config.is_enabled:
        import uuid
        helicone_config.session_name = "ragas-evaluation"
        helicone_config.session_id = str(uuid.uuid4())

    # Validação básica: dataset não pode ser None
    if dataset is None:
        raise ValueError("Provide dataset!")

    # ===== FASE 2: CONFIGURAÇÃO DAS MÉTRICAS PADRÃO =====
    
    # Se nenhuma métrica foi especificada, usa as 4 principais métricas do Ragas
    if metrics is None:
        from ragas.metrics import (
            answer_relevancy,     # Relevância da resposta
            context_precision,    # Precisão do contexto
            context_recall,       # Recall do contexto
            faithfulness,         # Fidelidade da resposta
        )
        metrics = [answer_relevancy, context_precision, faithfulness, context_recall]

    # ===== FASE 3: PROCESSAMENTO DO DATASET =====
    
    # Converte Dataset do HuggingFace para formato interno do Ragas
    if isinstance(dataset, Dataset):
        # Remapeia nomes das colunas se necessário
        dataset = remap_column_names(dataset, column_map)
        # Converte da versão 1 para versão 2 do formato
        dataset = convert_v1_to_v2_dataset(dataset)
        # Cria objeto EvaluationDataset
        dataset = EvaluationDataset.from_list(dataset.to_list())

    # Validações do dataset processado
    if isinstance(dataset, EvaluationDataset):
        # Verifica se o dataset tem as colunas necessárias para as métricas
        validate_required_columns(dataset, metrics)
        # Verifica se as métricas são suportadas para este tipo de dataset
        validate_supported_metrics(dataset, metrics)

    # ===== FASE 4: CONFIGURAÇÃO DOS MODELOS (LLM e EMBEDDINGS) =====
    
    # Wraps modelos do Langchain para compatibilidade com Ragas
    if isinstance(llm, LangchainLLM):
        llm = LangchainLLMWrapper(llm, run_config=run_config)
    if isinstance(embeddings, LangchainEmbeddings):
        embeddings = LangchainEmbeddingsWrapper(embeddings)

    # ===== FASE 5: INICIALIZAÇÃO DAS MÉTRICAS =====
    
    # Variáveis para rastrear mudanças nos modelos
    binary_metrics = []  # Lista de métricas binárias (AspectCritic)
    llm_changed: t.List[int] = []  # Índices das métricas que receberam LLM
    embeddings_changed: t.List[int] = []  # Índices das métricas que receberam embeddings
    answer_correctness_is_set = -1  # Índice da métrica AnswerCorrectness

    # Loop através de cada métrica para configurá-las
    for i, metric in enumerate(metrics):
        # Identifica métricas binárias (AspectCritic)
        if isinstance(metric, AspectCritic):
            binary_metrics.append(metric.name)
            
        # Configura LLM para métricas que precisam dele
        if isinstance(metric, MetricWithLLM) and metric.llm is None:
            if llm is None:
                llm = llm_factory()  # Cria LLM padrão
            metric.llm = llm
            llm_changed.append(i)
            
        # Configura embeddings para métricas que precisam dele
        if isinstance(metric, MetricWithEmbeddings) and metric.embeddings is None:
            if embeddings is None:
                embeddings = embedding_factory()  # Cria embeddings padrão
            metric.embeddings = embeddings
            embeddings_changed.append(i)
            
        # Tratamento especial para AnswerCorrectness
        if isinstance(metric, AnswerCorrectness):
            if metric.answer_similarity is None:
                answer_correctness_is_set = i

        # Inicializa a métrica com a configuração de execução
        metric.init(run_config)

    # ===== FASE 6: CONFIGURAÇÃO DO EXECUTOR =====
    
    # O Executor gerencia a execução paralela das métricas
    executor = Executor(
        desc="Evaluating",  # Descrição para a barra de progresso
        keep_progress_bar=True,
        raise_exceptions=raise_exceptions,
        run_config=run_config,
        show_progress=show_progress,
        batch_size=batch_size,
        pbar=_pbar,
    )

    # ===== FASE 7: CONFIGURAÇÃO DOS CALLBACKS =====
    
    # Callbacks do Ragas para rastreamento e custos
    ragas_callbacks: t.Dict[str, BaseCallbackHandler] = {}

    # RagasTracer: rastreia toda a execução da avaliação
    tracer = RagasTracer()
    ragas_callbacks["tracer"] = tracer

    # Callback de custo (se parser de tokens foi fornecido)
    if token_usage_parser is not None:
        from ragas.cost import CostCallbackHandler
        cost_cb = CostCallbackHandler(token_usage_parser=token_usage_parser)
        ragas_callbacks["cost_cb"] = cost_cb

    # Adiciona callbacks do Ragas à lista de callbacks
    for cb in ragas_callbacks.values():
        if isinstance(callbacks, BaseCallbackManager):
            callbacks.add_handler(cb)
        else:
            callbacks.append(cb)

    # ===== FASE 8: EXECUÇÃO DA AVALIAÇÃO =====
    
    # Cria grupo principal de rastreamento
    row_run_managers = []
    evaluation_rm, evaluation_group_cm = new_group(
        name=experiment_name or RAGAS_EVALUATION_CHAIN_NAME,
        inputs={},
        callbacks=callbacks,
        metadata={"type": ChainType.EVALUATION},
    )

    # Determina o tipo de sample (SingleTurn ou MultiTurn)
    sample_type = dataset.get_sample_type()
    
    # Loop através de cada sample no dataset
    for i, sample in enumerate(dataset):
        # Converte sample para dicionário
        row = t.cast(t.Dict[str, t.Any], sample.model_dump())
        
        # Cria grupo de rastreamento para esta linha
        row_rm, row_group_cm = new_group(
            name=f"row {i}",
            inputs=row,
            callbacks=evaluation_group_cm,
            metadata={"type": ChainType.ROW, "row_index": i},
        )
        row_run_managers.append((row_rm, row_group_cm))
        
        # Submete tarefas baseado no tipo de sample
        if sample_type == SingleTurnSample:
            # Para conversas de uma única troca
            _ = [
                executor.submit(
                    metric.single_turn_ascore,  # Método async de pontuação
                    sample,
                    row_group_cm,
                    name=f"{metric.name}-{i}",
                    timeout=run_config.timeout,
                )
                for metric in metrics
                if isinstance(metric, SingleTurnMetric)
            ]
        elif sample_type == MultiTurnSample:
            # Para conversas de múltiplas trocas
            _ = [
                executor.submit(
                    metric.multi_turn_ascore,  # Método async de pontuação
                    sample,
                    row_group_cm,
                    name=f"{metric.name}-{i}",
                    timeout=run_config.timeout,
                )
                for metric in metrics
                if isinstance(metric, MultiTurnMetric)
            ]
        else:
            raise ValueError(f"Unsupported sample type {sample_type}")

    # ===== FASE 9: COLETA E PROCESSAMENTO DOS RESULTADOS =====
    
    scores: t.List[t.Dict[str, t.Any]] = []
    
    try:
        # Obtém resultados do executor
        results = executor.results()
        if results == []:
            raise ExceptionInRunner()

        # Converte resultados para formato de dataset
        for i, _ in enumerate(dataset):
            s = {}  # Dicionário de pontuações para este sample
            for j, m in enumerate(metrics):
                # Cria chave baseada no tipo de métrica
                if isinstance(m, ModeMetric):
                    key = f"{m.name}(mode={m.mode})"
                else:
                    key = m.name
                # Atribui pontuação (results é uma lista linear)
                s[key] = results[len(metrics) * i + j]
            scores.append(s)
            
            # Fecha o rastreamento desta linha
            row_rm, row_group_cm = row_run_managers[i]
            if not row_group_cm.ended:
                row_rm.on_chain_end(s)

    except Exception as e:
        # Em caso de erro, registra no rastreamento
        if not evaluation_group_cm.ended:
            evaluation_rm.on_chain_error(e)
        raise e
    else:
        # ===== FASE 10: CRIAÇÃO DO RESULTADO FINAL =====
        
        # Obtém callback de custo se disponível
        cost_cb = ragas_callbacks.get("cost_cb")
        
        # Cria objeto de resultado final
        result = EvaluationResult(
            scores=scores,  # Todas as pontuações
            dataset=dataset,  # Dataset original
            binary_columns=binary_metrics,  # Métricas binárias
            cost_cb=t.cast(t.Union["CostCallbackHandler", None], cost_cb),
            ragas_traces=tracer.traces,  # Rastreamentos
            run_id=_run_id,
        )
        
        # Fecha o rastreamento principal
        if not evaluation_group_cm.ended:
            evaluation_rm.on_chain_end({"scores": result.scores})
            
    finally:
        # ===== FASE 11: LIMPEZA E RESET =====
        
        # Reset dos LLMs das métricas (para evitar vazamentos de estado)
        for i in llm_changed:
            t.cast(MetricWithLLM, metrics[i]).llm = None
            
        # Reset dos embeddings das métricas
        for i in embeddings_changed:
            t.cast(MetricWithEmbeddings, metrics[i]).embeddings = None
            
        # Reset especial para AnswerCorrectness
        if answer_correctness_is_set != -1:
            t.cast(AnswerCorrectness, metrics[answer_correctness_is_set]).answer_similarity = None

        # Flush do batcher de analytics
        from ragas._analytics import _analytics_batcher
        _analytics_batcher.flush()

    return result


# =====================================================================
# EXPLICAÇÃO DETALHADA DO QUE ESTE CÓDIGO FAZ
# =====================================================================

"""
RESUMO GERAL:
============
Este arquivo implementa a função principal `evaluate()` do framework Ragas,
que é usada para avaliar a qualidade de sistemas RAG (Retrieval-Augmented Generation).

FLUXO DE EXECUÇÃO:
=================
1. VALIDAÇÃO: Verifica se os dados de entrada estão corretos
2. PREPARAÇÃO: Converte datasets e configura métricas padrão
3. CONFIGURAÇÃO: Inicializa modelos (LLM, embeddings) e métricas
4. EXECUÇÃO: Roda as métricas em paralelo para cada sample do dataset
5. COLETA: Junta todos os resultados das métricas
6. RESULTADO: Retorna objeto com pontuações e análises

MÉTRICAS PADRÃO (4 principais):
===============================
1. answer_relevancy: Quão relevante é a resposta para a pergunta
2. context_precision: Quão precisos são os contextos recuperados
3. faithfulness: Quão fiel a resposta é aos contextos fornecidos
4. context_recall: Quão bem os contextos cobrem a resposta ground truth

TIPOS DE DADOS ESPERADOS:
========================
- question: A pergunta feita ao sistema
- answer: A resposta gerada pelo sistema RAG
- contexts: Lista de documentos/trechos recuperados
- ground_truth: A resposta correta (opcional, para algumas métricas)

FORMAS DE SIMPLIFICAR O USO:
===========================

1. USO BÁSICO (mais simples):
```python
from ragas import evaluate
result = evaluate(dataset)  # Usa configurações padrão
```

2. USO COM MÉTRICAS ESPECÍFICAS:
```python
from ragas.metrics import faithfulness, answer_relevancy
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```

3. USO COM MODELO PERSONALIZADO:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
result = evaluate(dataset, llm=llm)
```

4. USO SILENCIOSO (sem barra de progresso):
```python
result = evaluate(dataset, show_progress=False)
```

5. USO COM TRATAMENTO DE ERROS:
```python
result = evaluate(dataset, raise_exceptions=True)  # Para debug
```

PRINCIPAIS VANTAGENS DESTA IMPLEMENTAÇÃO:
=========================================
- Execução PARALELA das métricas (mais rápido)
- Suporte a CALLBACKS para monitoramento
- RASTREAMENTO completo da execução
- Cálculo de CUSTOS de API
- VALIDAÇÃO robusta dos dados
- Suporte a diferentes tipos de conversação (single/multi-turn)
- RESET automático de estado para evitar vazamentos

PONTOS DE ATENÇÃO:
=================
- Requer dataset com colunas específicas
- Precisa de chaves de API para modelos (OpenAI, etc.)
- Pode ser lento para datasets grandes
- Consome tokens dos modelos (gera custos)
- Métricas diferentes precisam de dados diferentes
"""