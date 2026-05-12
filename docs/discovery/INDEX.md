# Discovery Index

| # | Slug | Описание | Статус |
|---|---|---|---|
| 01 | `01-rag-pipeline` | Базовый RAG CLI: chunker, indexer, retriever, generator, main | Done |
| 02 | `02-gradio-openrouter` | Gradio UI, переход на OpenRouter (мульти-модель) | Done |
| 03 | `03-phoenix-tracing` | Phoenix observability, OpenTelemetry трейсинг LLM-вызовов | Done |
| 04 | `04-ragas-eval` | RAGAS evaluation pipeline, eval_dataset.json (15 вопросов) | Done |
| 05 | `05-hybrid-search` | Hybrid Search (BM25 + Vector + RRF), pymorphy3, Eval-вкладка в Gradio, Phoenix Datasets | Done |
| 06 | `06-retrieval-tuning` | Тюнинг чанкинга (500/100), TOP_K=7, RRF-кандидаты 15; раздельные answer/judge модели в eval-вкладке; фикс Phoenix Client API (v15); фикс Qwen3 thinking mode для RAGAS; uninstrument трейсинга во время судейства | Done |
| 07 | `07-eval-perf-models` | max_retries=5 в generator (rate limit); RunConfig(max_workers=8) в RAGAS (4–6× быстрее); каталог моделей FREE/CHEAP/SMART; дефолты eval — google/gemma-4-26b-a4b-it + google/gemma-4-31b-it | Done |
| 08 | `08-deepeval-migration` | Миграция с RAGAS на DeepEval (G-Eval, 4 вызова/sample vs ~15–20); параллельная генерация ответов (ThreadPoolExecutor); async судья (AsyncOpenAI); кнопка Stop в Gradio; новый baseline установлен | Done |
