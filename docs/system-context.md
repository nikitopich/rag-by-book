# System Context — rag-by-book

## Overview

RAG-приложение для вопросно-ответного взаимодействия с текстовой книгой. Пользователь индексирует `.txt` файл, после чего задаёт вопросы через Gradio web UI или CLI — система находит релевантные фрагменты и генерирует ответ через LLM. Поддерживает любую модель через OpenRouter. Включает Phoenix-трейсинг для наблюдаемости и **DeepEval**-пайплайн для оценки качества. Retrieval реализован как **Hybrid Search (BM25 + Vector + RRF)** с морфологической нормализацией для русского текста.

---

## Components

### `config.py`
Централизованная конфигурация.

| Параметр | Значение | Описание |
|---|---|---|
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Базовый URL OpenRouter |
| `DEFAULT_MODEL` | `deepseek/deepseek-v4-flash` | Модель по умолчанию |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformers (локально, поддерживает RU) |
| `CHUNK_SIZE` | 500 | Максимальный размер чанка в символах (уменьшен с 1000 для повышения precision) |
| `CHUNK_OVERLAP` | 100 | Перекрытие между чанками |
| `TOP_K` | 7 | Количество чанков для контекста (увеличен с 5 для повышения recall) |
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
- Параметры: `BM25_TOP_N=15`, `VECTOR_TOP_N=15`, финальный `top_k=7`

Возвращает `dict` с ключами `documents`, `distances`, `ids`.

---

### `generator.py`
Генерация ответа через OpenRouter (OpenAI-совместимый API).

- `SYSTEM_PROMPT` — инструкция отвечать только по контексту, на русском
- `build_user_message(query, context_chunks)` — собирает промпт: контекст через `---`, затем вопрос
- `generate_answer(query, context_chunks, api_key, model)` — вызывает выбранную модель через `openai.OpenAI` с `base_url=OPENROUTER_BASE_URL`, `max_tokens=5000`, `max_retries=5` (автоматический exponential backoff при 429 rate limit)

---

### `app.py`
Gradio web UI. Точка входа для пользователей.

**Вкладки:**
- **Чат** — диалог с книгой: дропдаун модели, Top-K слайдер, переключатель Retriever (hybrid/vector), Debug-режим (показывает чанки и промпт), history
- **Индексация** — загрузка `.txt` файла и индексация в ChromaDB
- **Оценка** — запуск DeepEval-оценки без консоли:
  - **Модель-ответчик** — отдельный дропдаун для модели, генерирующей ответы
  - **Судья (judge model)** — отдельный дропдаун для DeepEval-судьи
  - Переключатель Retriever (hybrid/vector), опциональная загрузка датасета JSON
  - Кнопка **⏹ Остановить** — прерывает eval через Gradio `cancels` механизм
  - Результаты стримятся построчно, сохраняются в Phoenix через `phoenix.client.Client` (`client.datasets.create_dataset`) с именем `eval-{timestamp}-{model}-{retriever}`
  - Во время судейства трейсинг временно отключается (`OpenAIInstrumentor().uninstrument()`) — иначе служебные вызовы засоряют Phoenix

**Layout:** OpenRouter API Key — единственный глобальный элемент вверху; дропдаун модели перенесён внутрь таба «Чат».

**Доступные модели** (сгруппированы в UI по секциям FREE / CHEAP / SMART):
- **FREE** (лимит 200 req/day): `google/gemma-4-26b-a4b-it:free`, `google/gemma-4-31b-it:free`, `qwen/qwen3-next-80b-a3b-instruct:free`, `deepseek/deepseek-r1:free`, `meta-llama/llama-3.3-70b-instruct:free`
- **CHEAP**: `google/gemma-4-26b-a4b-it` ($0.06/M), `google/gemma-4-31b-it` ($0.13/M), `deepseek/deepseek-v4-flash`, `openai/gpt-4o-mini`, `qwen/qwen3.6-flash`
- **SMART**: `openai/gpt-4o`, `anthropic/claude-sonnet-4-5`, `google/gemini-2.5-flash/pro-preview`, `deepseek/deepseek-r1/r1-0528`, `qwen/qwen3.6-plus`, `moonshotai/kimi-k2`

**Дефолты eval-вкладки:** answer=`google/gemma-4-26b-a4b-it`, judge=`google/gemma-4-31b-it` — оптимальный баланс цены и качества для RU-текста.

**Особенность Qwen3:** модели серии `qwen3.*` используют режим `<think>...</think>` по умолчанию, что может мешать парсингу ответов судьи. При необходимости передавай `extra_body={"thinking": False}` вручную или выбирай другую модель-судью.

Запуск: `python app.py` → открывает браузер, заголовок содержит ссылку на Phoenix UI.

---

### `tracing.py`
Observability через Arize Phoenix.

- `setup_tracing()` — запускает Phoenix UI локально (`px.launch_app()`) если не запущен, иначе переиспользует
- Настраивает OpenTelemetry OTLP-экспортер → Phoenix
- Инструментирует OpenAI SDK (`OpenAIInstrumentor`) — все LLM-вызовы автоматически трейсируются
- Возвращает URL дашборда (отображается в заголовке Gradio)

**Что видно в Phoenix:** latency, токены, стоимость, входные/выходные данные каждого LLM-вызова.

---

### `eval.py`
DeepEval evaluation pipeline. CLI-инструмент для оценки качества RAG.

**Использование:**
```bash
python eval.py --api-key sk-or-v1-... [--model ...] [--judge-model ...] [--dataset ...] [--output ...] [--retriever hybrid|vector]
```

**Метрики (G-Eval, 1 LLM-вызов на метрику на sample):**
| Метрика | Описание |
|---|---|
| `Faithfulness` | Ответ основан только на найденных чанках (не галлюцинирует) |
| `Answer Relevancy` | Ответ по существу вопроса |
| `Contextual Precision` | Найденные чанки действительно релевантны вопросу |
| `Contextual Recall` | Все нужные факты присутствуют в найденных чанках |

- Судья: `OpenRouterLLM(DeepEvalBaseLLM)` — обёртка над OpenAI-compatible API; `a_generate` реализован через `AsyncOpenAI` для настоящего параллелизма
- **Генерация ответов:** `ThreadPoolExecutor(max_workers=5)` — параллельные HTTP-запросы; retrieval последовательно (ChromaDB не thread-safe)
- **Eval:** `async_mode=True` (дефолт DeepEval) — все метрики по всем sample параллельно
- Итого вызовов судьи: 4 на sample (vs ~15–20 у RAGAS)
- Флаг `--retriever hybrid|vector` для A/B сравнения режимов
- Результаты выводятся в консоль и сохраняются в CSV
- Та же логика доступна через вкладку **Оценка** в `app.py` (без консоли)

**Новый baseline (судья `google/gemma-4-31b-it`, answer `google/gemma-4-26b-a4b-it`, hybrid):**
| Метрика | Значение |
|---|---|
| Faithfulness | 0.878 |
| Answer Relevancy | 0.741 |
| Contextual Precision | 0.422 |
| Contextual Recall | 0.444 |

---

### `eval_dataset.json`
15 вопросов по книге с эталонными ответами (`reference`) для RAGAS-оценки.
Содержит как фактические вопросы, так и «негативные» (о несуществующих персонажах/событиях) — последние структурно занижают `context_recall`, т.к. ответ «такого нет» не встречается в чанках.

---

### `main.py`
CLI-интерфейс (REPL), устаревший вариант интерфейса.

**Команды:** `/index`, `/debug`, `/topk <n>`, `/help`, `/exit`

---

## Data Flow

```
Текстовый файл
    → chunker.load_and_chunk()        # list[dict] чанков (CHUNK_SIZE=500, OVERLAP=100)
    → indexer.index_document()        # ChromaDB (./chroma_db)

Вопрос пользователя (app.py или main.py)
    → retriever.hybrid_retrieve()     # BM25(top15) + Vector(top15) → RRF → top-7
    → generator.generate_answer()     # ответ от LLM через OpenRouter
    → вывод в Gradio UI / консоль
    → tracing.py → Phoenix UI         # трейс каждого вызова

Оценка качества (eval.py или вкладка Оценка в app.py)
    → retriever.hybrid_retrieve()     # hybrid: BM25 + Vector + RRF
    → generator.generate_answer()     # генерация ответов (answer model)
    → [uninstrument трейсинга]
    → DeepEval evaluate()             # 4 метрики, async_mode=True, 4 вызова/sample
    → [reinstrument трейсинга]
    → CSV + консоль / Phoenix Dataset (phoenix.client.Client)
```

---

## Infrastructure

| Компонент | Технология |
|---|---|
| Vector DB | ChromaDB (локальная, персистентная) |
| Embeddings | sentence-transformers (локально) |
| LLM | Любая модель через OpenRouter (OpenAI-совместимый API) |
| Web UI | Gradio |
| Observability | Arize Phoenix 15.x + OpenTelemetry |
| Evaluation | DeepEval (Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall) |
| Runtime | Python 3.11+ |

---

## Dependencies

```
openai
gradio
arize-phoenix>=15.0
opentelemetry-sdk
openinference-instrumentation-openai
deepeval
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
| 06 | `06-retrieval-tuning` | Тюнинг чанкинга (500/100), TOP_K=7, RRF-кандидаты 15; раздельные answer/judge модели в eval-вкладке; фикс Phoenix Client API (v15); фикс Qwen3 thinking mode для RAGAS; uninstrument трейсинга во время судейства | Done |
| 07 | `07-eval-perf-models` | max_retries=5 в generator (rate limit); RunConfig(max_workers=8) в RAGAS (4–6× быстрее); каталог моделей FREE/CHEAP/SMART; дефолты eval — google/gemma-4-26b-a4b-it + google/gemma-4-31b-it | Done |
| 08 | `08-deepeval-migration` | Миграция с RAGAS на DeepEval (G-Eval, 4 вызова/sample vs ~15–20); параллельная генерация ответов (ThreadPoolExecutor); async судья (AsyncOpenAI); кнопка Stop в Gradio; новый baseline установлен | Done |
