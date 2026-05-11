# System Context — rag-by-book

## Overview

RAG-приложение для вопросно-ответного взаимодействия с текстовой книгой. Пользователь индексирует `.txt` файл, после чего задаёт вопросы через Gradio web UI или CLI — система находит релевантные фрагменты и генерирует ответ через LLM. Поддерживает любую модель через OpenRouter. Включает Phoenix-трейсинг для наблюдаемости и RAGAS-пайплайн для оценки качества. Retrieval реализован как **Hybrid Search (BM25 + Vector + RRF)** с морфологической нормализацией для русского текста.

---

## Components

### `config.py`
Централизованная конфигурация.

| Параметр | Значение | Описание |
|---|---|---|
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Базовый URL OpenRouter |
| `DEFAULT_MODEL` | `openai/gpt-4o-mini` | Модель по умолчанию |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformers (локально, поддерживает RU) |
| `CHUNK_SIZE` | 1000 | Максимальный размер чанка в символах |
| `CHUNK_OVERLAP` | 200 | Перекрытие между чанками |
| `TOP_K` | 5 | Количество чанков для контекста |
| `CHROMA_DB_PATH` | `./chroma_db` | Путь к локальной ChromaDB |
| `COLLECTION_NAME` | `book_chunks` | Имя коллекции в ChromaDB |

---

### `chunker.py`
Загрузка и разбиение текста на чанки.

- Читает `.txt` файл целиком
- Разбивает по двойным переносам строк (абзацы)
- Собирает чанки с перекрытием: если абзац не помещается — сохраняет чанк, берёт последние `overlap` символов как начало следующего
- Возвращает `list[dict]` с полями `id`, `text`, `metadata` (`index`, `char_start`)

---

### `indexer.py`
Векторизация и сохранение чанков в ChromaDB.

- Lazy-инициализация `SentenceTransformer` (singleton `_model`)
- `get_embeddings(texts)` — батчевая векторизация
- `index_document(chunks)` — пересоздаёт коллекцию, добавляет чанки батчами по 64
- Персистентный клиент ChromaDB: данные сохраняются в `./chroma_db`

---

### `retriever.py`
Поиск релевантных чанков по запросу. Два режима.

**`retrieve(query, top_k)`** — чистый векторный поиск (fallback/baseline):
- Векторизует запрос через `get_embeddings` из `indexer`
- Запрашивает ChromaDB: `n_results=top_k`

**`hybrid_retrieve(query, top_k)`** — Hybrid Search (используется по умолчанию):
- **BM25** (`rank_bm25`): строит индекс всего корпуса в памяти, токенизирует с морфологической нормализацией (`pymorphy3` для русского, fallback на `lower().split()`)
- **Vector**: ChromaDB косинусный поиск
- **RRF (Reciprocal Rank Fusion)**: объединяет ранги, `score = Σ 1/(k + rank_i)`, `k=60`; отсутствующий документ получает штраф `corpus_size` (а не `top_n`) — это ключевой момент корректности
- Параметры: `BM25_TOP_N=10`, `VECTOR_TOP_N=10`, финальный `top_k=5`

Возвращает `dict` с ключами `documents`, `distances`, `ids`.

---

### `generator.py`
Генерация ответа через OpenRouter (OpenAI-совместимый API).

- `SYSTEM_PROMPT` — инструкция отвечать только по контексту, на русском
- `build_user_message(query, context_chunks)` — собирает промпт: контекст через `---`, затем вопрос
- `generate_answer(query, context_chunks, api_key, model)` — вызывает выбранную модель через `openai.OpenAI` с `base_url=OPENROUTER_BASE_URL`, `max_tokens=5000`

---

### `app.py`
Gradio web UI. Точка входа для пользователей.

**Вкладки:**
- **Чат** — диалог с книгой: Top-K слайдер, переключатель Retriever (hybrid/vector), Debug-режим (показывает чанки и промпт), history
- **Индексация** — загрузка `.txt` файла и индексация в ChromaDB
- **Оценка** — запуск RAGAS-оценки без консоли: выбор судьи, режима retriever, опциональная загрузка датасета JSON; результаты стримятся построчно и автоматически сохраняются в Phoenix как датасет (`eval-{timestamp}-{model}-{retriever}`)

**Параметры в UI:**
- OpenRouter API Key (password field, shared across tabs)
- Выбор модели из дропдауна: OpenAI, Anthropic, Gemini 2.x, DeepSeek, Qwen, Kimi, Llama

Запуск: `python app.py` → открывает браузер, заголовок содержит ссылку на Phoenix UI.

---

### `tracing.py`
Observability через Arize Phoenix.

- `setup_tracing()` — запускает Phoenix UI локально (`px.launch_app()`)
- Настраивает OpenTelemetry OTLP-экспортер → Phoenix
- Инструментирует OpenAI SDK (`OpenAIInstrumentor`) — все LLM-вызовы автоматически трейсируются
- Возвращает URL дашборда (отображается в заголовке Gradio)

**Что видно в Phoenix:** latency, токены, стоимость, входные/выходные данные каждого LLM-вызова.

---

### `eval.py`
RAGAS evaluation pipeline. CLI-инструмент для оценки качества RAG.

**Использование:**
```bash
python eval.py --api-key sk-or-v1-... [--model ...] [--judge-model ...] [--dataset ...] [--output ...]
```

**Метрики:**
| Метрика | Описание |
|---|---|
| `faithfulness` | Ответ основан только на найденных чанках (не галлюцинирует) |
| `answer_relevancy` | Ответ по существу вопроса |
| `context_precision` | Найденные чанки действительно релевантны вопросу |
| `context_recall` | Все нужные факты присутствуют в найденных чанках |

- Судья по умолчанию: `deepseek/deepseek-v4-pro` через OpenRouter
- Флаг `--retriever hybrid|vector` для сравнения режимов
- Использует `LangchainLLMWrapper` + `LangchainEmbeddingsWrapper` для совместимости с RAGAS
- Результаты выводятся в консоль и сохраняются в CSV
- Та же логика доступна через вкладку **Оценка** в `app.py` (без консоли)

---

### `eval_dataset.json`
15 вопросов по книге с эталонными ответами (`reference`) для RAGAS-оценки.

---

### `main.py`
CLI-интерфейс (REPL), устаревший вариант интерфейса.

**Команды:** `/index`, `/debug`, `/topk <n>`, `/help`, `/exit`

---

## Data Flow

```
Текстовый файл
    → chunker.load_and_chunk()        # list[dict] чанков
    → indexer.index_document()        # ChromaDB (./chroma_db)

Вопрос пользователя (app.py или main.py)
    → retriever.retrieve()            # top-K чанков из ChromaDB
    → generator.generate_answer()     # ответ от LLM через OpenRouter
    → вывод в Gradio UI / консоль
    → tracing.py → Phoenix UI         # трейс каждого вызова

Оценка качества (eval.py или вкладка Оценка в app.py)
    → retriever.hybrid_retrieve()     # hybrid: BM25 + Vector + RRF
    → generator.generate_answer()     # генерация ответов
    → RAGAS evaluate()                # 4 метрики через judge LLM
    → CSV + консоль / Phoenix Dataset
```

---

## Infrastructure

| Компонент | Технология |
|---|---|
| Vector DB | ChromaDB (локальная, персистентная) |
| Embeddings | sentence-transformers (локально) |
| LLM | Любая модель через OpenRouter (OpenAI-совместимый API) |
| Web UI | Gradio |
| Observability | Arize Phoenix + OpenTelemetry |
| Evaluation | RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) |
| Runtime | Python 3.11+ |

---

## Dependencies

```
openai
gradio
arize-phoenix
opentelemetry-sdk
openinference-instrumentation-openai
ragas
langchain-openai
sentence-transformers
chromadb
rank_bm25
pymorphy3
pymorphy3-dicts-ru
```

---

## Iterations

| # | Slug | Описание | Статус |
|---|---|---|---|
| 01 | `01-rag-pipeline` | Базовый RAG CLI: chunker, indexer, retriever, generator, main | Done |
| 02 | `02-gradio-openrouter` | Gradio UI, переход на OpenRouter (мульти-модель) | Done |
| 03 | `03-phoenix-tracing` | Phoenix observability, OpenTelemetry трейсинг LLM-вызовов | Done |
| 04 | `04-ragas-eval` | RAGAS evaluation pipeline, eval_dataset.json (15 вопросов) | Done |
| 05 | `05-hybrid-search` | Hybrid Search (BM25 + Vector + RRF), pymorphy3, Eval-вкладка в Gradio, Phoenix Datasets | Done |
