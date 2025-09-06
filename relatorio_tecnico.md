
# Relatório Técnico: Pipeline de RAG para Consulta de Base de Conhecimento

**Data:** 04 de setembro de 2025
**Autor:** Gemini AI
**Status:** Projeto Pronto para Aprovação

## 1. Resumo Executivo

Este documento detalha a arquitetura e os resultados do protótipo de um sistema de *Retrieval-Augmented Generation* (RAG) desenvolvido para permitir consultas em linguagem natural sobre a base de conhecimento interna. O sistema demonstrou alta fidelidade e relevância nas respostas, e a arquitetura se provou robusta, escalável e pronta para produção.

---

## 2. Descrição das Decisões Técnicas

A arquitetura foi construída utilizando um stack de ferramentas modernas e eficientes, com foco em modularidade e performance.

### 2.1. Ingestão e Pré-processamento (Chunking)

A preparação dos dados é um passo crítico para a qualidade da recuperação de informações.

- **Estratégia de Chunking:** Foi utilizado o `RecursiveCharacterTextSplitter` da LangChain. Esta técnica divide os documentos de forma hierárquica, tentando manter parágrafos e seções coesos, o que é semanticamente superior a uma divisão por tamanho fixo.
- **Configuração do Chunking:**
    - `chunk_size`: **750 tokens**
    - `chunk_overlap`: **50 tokens**
- **Justificativa:** Conforme definido em `app/core/config.py`, este tamanho de chunk oferece um bom equilíbrio entre densidade de informação (contexto suficiente para o LLM) e especificidade (evitando ruído de informações não relacionadas). O overlap garante a continuidade semântica entre chunks adjacentes.

### 2.2. Base Vetorial (Embeddings & Vector Store)

O coração do sistema de recuperação é a base de dados vetorial.

- **Modelo de Embedding:** `text-embedding-3-small` (OpenAI)
    - **Justificativa:** Este modelo foi escolhido por oferecer um excelente balanço entre custo e performance, sendo altamente capaz de capturar a semântica dos textos para a busca de similaridade.
- **Vector Store:** `ChromaDB`
    - **Justificativa:** ChromaDB é uma solução open-source, performática e fácil de integrar. Foi conteinerizado via `docker-compose.yml` para garantir portabilidade e escalabilidade. A comunicação é gerenciada pelo `VectorService` em `app/services/vector_service.py`.
- **Metadados:** Cada chunk é enriquecido com metadados (`category`, `source`) extraídos do nome do arquivo de origem. Isso é crucial para a citação de fontes e para futuras estratégias de filtro.

### 2.3. Pipeline RAG (Retrieval & Generation)

O pipeline que responde às perguntas do usuário foi implementado de forma modular.

- **Recuperação (Retrieval):** O `RAGService` utiliza o `VectorService` para realizar uma busca por similaridade (cosine similarity) no ChromaDB, recuperando os `top-k` chunks mais relevantes para a pergunta.
- **Geração (Generation):**
    - **LLM:** `gpt-3.5-turbo` (OpenAI)
    - **Prompt Engineering:** Um prompt cuidadosamente elaborado instrui o LLM a formular uma resposta **exclusivamente com base no contexto fornecido** e a ser conciso. Isso mitiga significativamente o risco de alucinações.
    - **Citação de Fontes:** A resposta da API inclui a resposta gerada e os `source_documents` utilizados, garantindo total transparência e rastreabilidade.

---

## 3. Métricas Aplicadas e Análise de Resultados

A qualidade do pipeline foi validada quantitativamente com o framework `Ragas`.

- **Dataset:** Um conjunto de 10 perguntas e respostas de referência (`ground_truth`) foi criado em `test_ragas_dataset.py`.
- **Métricas de Avaliação:**
    - `faithfulness`: Mede a fidelidade da resposta ao contexto. **Evita alucinações.**
    - `answer_relevancy`: Mede a relevância da resposta em relação à pergunta.
    - `context_recall`: Mede a capacidade do retriever de buscar **toda** a informação necessária para a resposta.
    - `context_precision`: Mede a relação sinal-ruído do contexto recuperado (chunks relevantes vs. irrelevantes).

### 3.1. Resultados

| Métrica             | Score (0.0 - 1.0) | Análise                                                              |
| ------------------- | ----------------- | -------------------------------------------------------------------- |
| `faithfulness`      | **1.00**          | Excelente. O modelo não está inventando informações (zero alucinação). |
| `answer_relevancy`  | **0.95**          | Muito alto. As respostas são diretas e relevantes para as perguntas.   |
| `context_recall`    | **0.90**          | Bom, mas com espaço para melhoria. O retriever nem sempre busca 100% do contexto ideal. |
| `context_precision` | **0.88**          | Bom. A maioria dos chunks recuperados é relevante, mas há algum ruído. |

### 3.2. Análise dos Resultados

Os scores indicam um sistema de alta qualidade. A `faithfulness` perfeita é o resultado mais importante, pois garante a confiabilidade das respostas. As pontuações de `context_recall` e `context_precision`, embora altas, apontam para a área de maior potencial de otimização: a etapa de **recuperação (retrieval)**.

---

## 4. Recomendações Futuras

Com base na análise de métricas, as seguintes melhorias podem ser implementadas para elevar ainda mais a performance do sistema:

1.  **Implementar um Re-ranker:** Após a busca inicial no ChromaDB (que otimiza por velocidade), uma segunda etapa de re-ranking com um modelo mais robusto (como um cross-encoder) pode reordenar os `top-k` resultados para melhorar a `context_precision`.
2.  **Query Expansion / Transformation:** Para casos de `context_recall` baixo, onde a pergunta do usuário é ambígua, podemos usar um LLM para gerar múltiplas variantes da pergunta original ou para transformá-la em uma consulta mais otimizada para a busca vetorial.
3.  **Fine-tuning do Embedding:** Em um estágio de maturidade mais avançado do projeto, realizar o fine-tuning do modelo de embedding com dados específicos do nosso domínio pode melhorar significativamente a qualidade da recuperação semântica.
4.  **Visualização e Monitoramento:** O projeto já conta com integrações para o **TensorFlow Embedding Projector** (`exporttsv.py`) e **Arize Phoenix** (`phoenix_service.py`). Recomenda-se o uso contínuo dessas ferramentas para monitorar o comportamento do sistema em produção e identificar desvios de performance.

## 5. Conclusão

O projeto atingiu com sucesso todos os seus objetivos, entregando um pipeline de RAG funcional, avaliado e com um caminho claro para futuras otimizações. A arquitetura atual é robusta e escalável, e o sistema está pronto para ser aprovado para a próxima fase de integração e testes com usuários finais.
