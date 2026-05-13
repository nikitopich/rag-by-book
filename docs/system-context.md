# System Context — rag-by-book

## Overview

RAG-приложение для вопросно-ответного взаимодействия с текстовой книгой. Пользователь индексирует `.txt` файл, после чего задаёт вопросы через Gradio web UI или CLI — система находит релевантные фрагменты и генерирует ответ через LLM. Поддерживает любую модель через OpenRouter. Включает Phoenix-трейсинг для наблюдаемости и **DeepEval**-пайплайн для оценки качества. Retrieval реализован как **Hybrid Search (BM25 + Vector + RRF) + Cross-Encoder Reranker** с морфологической нормализацией для русского текста.

---

## Components

### `config.py`
Централизованная конфигурация.

| Параметр | Значение | Описание |
|---|---|---|
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Базовый URL OpenRouter |
| `DEFAULT_MODEL` | `deepseek/deepseek-v4-flash` | Модель по умолчанию |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Sentence-transformers, многоязычная; индексация с префиксом `passage: `, поиск — `query: ` |
| `CHUNK_SIZE` | 500 | Максимальный размер чанка в символах |
| `CHUNK_OVERLAP` | 100 | Перекрытие между чанками |
| `TOP_K` | 7 | Финальное количество чанков для контекста |
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
- `get_embeddings(texts, prefix="")` — батчевая векторизация с опциональным префиксом E5 (`passage: ` при индексации, `query: ` при поиске — это требование модели E5)
- `index_document(chunks)` — пересоздаёт коллекцию, добавляет чанки батчами по 64
- Персистентный клиент ChromaDB: данные сохраняются в `./chroma_db`

---

### `retriever.py`
Поиск релевантных чанков по запросу. **Три режима.**

**`retrieve(query, top_k)`** — чистый векторный поиск (fallback/baseline):
- Векторизует запрос с префиксом `query: `, запрашивает ChromaDB

**`hybrid_retrieve(query, top_k)`** — Hybrid Search:
- **BM25** (`rank_bm25`): индекс всего корпуса в памяти, токенизация с морфологической нормализацией (`pymorphy3` для русского, fallback `lower().split()`)
- **Vector**: ChromaDB косинусный поиск с префиксом `query: `
- **RRF (Reciprocal Rank Fusion)**: `score = Σ 1/(k + rank_i)`, `k=60`; отсутствующий документ получает штраф `corpus_size` — ключевой момент корректности
- Параметры: `BM25_TOP_N=15`, `VECTOR_TOP_N=15`, финальный `top_k=7`

**`hybrid_retrieve_reranked(query, top_k)`** — Hybrid + Cross-Encoder Rerank (default, см. [ADR-004](adr/ADR-004-cross-encoder-reranker.md)):
- BM25 top-30 + Vector top-30 → RRF → top-30 кандидатов
- **Cross-encoder** (`BAAI/bge-reranker-v2-m3`, lazy-singleton `_reranker`) переоценивает все 30 пар `(query, doc)` через cross-attention
- Возвращает финальный top-K по reranker-скору + поле `reranker_scores` в результате

`RERANKER_CANDIDATES = 30`. Кеш BM25-индекса (`_bm25_cache`) — переиспользуется между запросами пока корпус не меняется.

Возвращает `dict` с ключами `documents`, `distances`, `ids` (+ `reranker_scores` для rerank-режима).

---

### `generator.py`
Генерация ответа через OpenRouter (OpenAI-совместимый API).

- `SYSTEM_PROMPT` — инструкция отвечать только по контексту, на русском
- `build_user_message(query, context_chunks)` — собирает промпт: контекст через `---`, затем вопрос
- `stream_answer(...)` / `generate_answer(...)` — стриминговая и блокирующая генерация, `max_tokens=2000`, `max_retries=5` (exponential backoff при 429)

---

### `app.py`
Gradio web UI. Точка входа для пользователей.

**Вкладки:**
- **Чат** — диалог с книгой: дропдаун модели, Top-K слайдер, переключатель Retriever (vector / hybrid / hybrid+rerank), Debug-режим (показывает чанки и промпт), history
- **Индексация** — загрузка `.txt` файла и индексация в ChromaDB
- **Оценка** — запуск DeepEval-оценки без консоли:
  - **Модель-ответчик** — отдельный дропдаун
  - **Судья (judge model)** — отдельный дропдаун
  - Переключатель Retriever, опциональная загрузка датасета JSON
  - Кнопка **⏹ Остановить** — прерывает eval через Gradio `cancels`
  - Результаты стримятся построчно, сохраняются в Phoenix через `phoenix.client.Client`
  - Во время судейства трейсинг временно отключается (`OpenAIInstrumentor().uninstrument()`)

**Судья (`OpenRouterLLM(DeepEvalBaseLLM)`) — устойчивый wrapper:**
- 4 попытки с exponential backoff
- Первые 2 попытки с `response_format={"type": "json_object"}`, последние 2 без — некоторые промпты ломают DeepSeek в JSON-mode (модель возвращает `len=0`)
- `_clean_llm_output` снимает `<think>...</think>`, markdown-фенсы ` ```json...``` ` и извлекает первый сбалансированный `{...}` через скобочный баланс с учётом строк/экранирования
- `_is_valid_json` — валидация перед возвратом; невалидный JSON → новый retry
- Защита от `resp.choices=None` (OpenRouter иногда отдаёт `{"error": ...}` без `choices`)
- `try/except` вокруг API-вызовов — сетевые сбои тоже идут в retry
- Финальный fallback: `"{}"` чтобы `ErrorConfig(ignore_errors=True)` смог корректно пропустить тест

**Глобальные настройки DeepEval (выставляются ДО импорта):**
- `os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "600"` — стандартный per-task timeout 180s слишком мал для 4 метрик × ~11 judge-вызовов на медленных провайдерах
- В вызов `deepeval_evaluate(...)` передаётся `ErrorConfig(ignore_errors=True)` — один упавший тест не убивает весь прогон

**Доступные модели** (сгруппированы в UI по секциям FREE / CHEAP / SMART):
- **FREE** (лимит 200 req/day): `google/gemma-4-26b-a4b-it:free`, `google/gemma-4-31b-it:free`, `qwen/qwen3-next-80b-a3b-instruct:free`, `deepseek/deepseek-r1:free`, `meta-llama/llama-3.3-70b-instruct:free`
- **CHEAP**: `google/gemma-4-26b-a4b-it`, `google/gemma-4-31b-it`, `deepseek/deepseek-v4-flash`, `openai/gpt-4o-mini`, `qwen/qwen3.6-flash`
- **SMART**: `openai/gpt-4o`, `anthropic/claude-sonnet-4-5`, `google/gemini-2.5-pro-preview`, `deepseek/deepseek-r1`, `qwen/qwen3.6-plus`, `moonshotai/kimi-k2`

**Дефолты eval-вкладки:** answer=`google/gemma-4-26b-a4b-it`, judge=`deepseek/deepseek-v4-flash`.

**Особенность Qwen3:** модели серии `qwen3.*` используют `<think>...</think>` по умолчанию. `_clean_llm_output` сам срезает thinking-блоки.

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
python eval.py --api-key sk-or-v1-... [--model ...] [--judge-model ...] [--dataset ...] [--output ...] [--retriever vector|hybrid|hybrid+rerank]
```

**Метрики (декомпозированные, см. [ADR-003](adr/ADR-003-deepeval-migration.md)):**

| Метрика | Описание | Вызовов судьи |
|---|---|---|
| `Faithfulness` | Ответ основан только на найденных чанках (не галлюцинирует) | 4 (truths → claims → verdicts → reason) |
| `Answer Relevancy` | Ответ по существу вопроса | 3 (statements → verdicts → reason) |
| `Contextual Precision` | Найденные чанки действительно релевантны вопросу | 2 (verdicts → reason) |
| `Contextual Recall` | Все нужные факты присутствуют в найденных чанках | 2 (verdicts → reason) |

**Итого: ~11 вызовов судьи на 1 sample** (а не 4, как ошибочно указано в [ADR-003](adr/ADR-003-deepeval-migration.md)). Реальная экономия vs RAGAS (~15–20 вызовов) — примерно в 1.5×, а не в 4×. Чтобы получить настоящий «1 вызов на метрику» — нужно перейти на `GEval`-метрики (с явной chain-of-thought и criteria), но они менее интерпретируемы.

**Параллелизм:**
- Генерация ответов: `ThreadPoolExecutor(max_workers=5)`; retrieval последовательно (ChromaDB не thread-safe)
- Eval: `async_mode=True` — все метрики по всем sample параллельно через `AsyncOpenAI`

---

### Датасеты

- **`eval_dataset.json`** — 15 вопросов, изначальный сет
- **`eval/data_set_2.json`** — 30 вопросов (текущий основной), сгенерирован из `eval/data_set_2.csv`. Поля: `question`, `reference`, `type` (`factual` / `thematic` / `character` / `plot` / `multi-hop` / `negative`), `difficulty` (`easy` / `medium` / `hard`). Конвертация в JSON: `python -c "import csv, json; ..."` (см. историю в eval-reports)

---

### `main.py`
CLI-интерфейс (REPL), устаревший вариант.

**Команды:** `/index`, `/debug`, `/topk <n>`, `/help`, `/exit`

---

## Data Flow

```
Текстовый файл
    → chunker.load_and_chunk()        # list[dict] чанков (500/100)
    → indexer.index_document()        # E5 prefix "passage: " → ChromaDB

Вопрос пользователя (app.py или main.py)
    → retriever.hybrid_retrieve_reranked()    # BM25(30) + Vector(30) → RRF(30) → CrossEncoder → top-7
    → generator.stream_answer()                # ответ через OpenRouter
    → вывод в Gradio UI / консоль
    → tracing.py → Phoenix UI                  # трейс каждого вызова

Оценка качества (eval.py или вкладка Оценка в app.py)
    → retriever.<mode>()                       # vector / hybrid / hybrid+rerank
    → generator.generate_answer() × N          # ThreadPoolExecutor(max_workers=5)
    → [uninstrument трейсинга]
    → DeepEval evaluate()                      # 4 метрики, async, ~11 вызовов/sample
    → [reinstrument трейсинга]
    → CSV + консоль / Phoenix Dataset
```

---

## Infrastructure

| Компонент | Технология |
|---|---|
| Vector DB | ChromaDB (локальная, персистентная) |
| Embeddings | `intfloat/multilingual-e5-base` (sentence-transformers, локально) |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder, локально, ~570MB) |
| LLM | Любая модель через OpenRouter (OpenAI-совместимый API) |
| Web UI | Gradio |
| Observability | Arize Phoenix 15.x + OpenTelemetry |
| Evaluation | DeepEval (декомпозированные метрики: Faithfulness, Answer Relevancy, Contextual Precision/Recall) |
| Runtime | Python 3.11+ (тестируется на 3.14) |

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

## Eval Reports

Каждый значимый прогон оценки сохраняется в [`docs/eval-reports/`](eval-reports/) с описанием конфигурации, диаграммой и вердиктом.

**Текущий baseline** (2026-05-13, [eval-reports/2026-05-13-gemma-deepseek.md](eval-reports/2026-05-13-gemma-deepseek.md)):
- Answer: `google/gemma-4-26b-a4b-it`, Judge: `deepseek/deepseek-v4-flash`, Retriever: `hybrid+rerank`
- Датасет: `eval/data_set_2.json` (30 вопросов)

| Метрика | Значение |
|---|---|
| Faithfulness | 0.795 |
| Answer Relevancy | 0.935 |
| Contextual Precision | 0.674 |
| Contextual Recall | 0.753 |
| **Pass Rate** | **63.3%** (19/30) |

---

## Iterations

| # | Slug | Описание | Статус |
|---|---|---|---|
| 01 | `01-rag-pipeline` | Базовый RAG CLI: chunker, indexer, retriever, generator, main | Done |
| 02 | `02-gradio-openrouter` | Gradio UI, переход на OpenRouter (мульти-модель) | Done |
| 03 | `03-phoenix-tracing` | Phoenix observability, OpenTelemetry трейсинг LLM-вызовов | Done |
| 04 | `04-ragas-eval` | RAGAS evaluation pipeline, eval_dataset.json (15 вопросов) | Done |
| 05 | `05-hybrid-search` | Hybrid Search (BM25 + Vector + RRF), pymorphy3, Eval-вкладка в Gradio, Phoenix Datasets | Done |
| 06 | `06-retrieval-tuning` | Тюнинг чанкинга (500/100), TOP_K=7, RRF-кандидаты 15; раздельные answer/judge модели в eval-вкладке; фикс Phoenix Client API (v15); фикс Qwen3 thinking mode; uninstrument трейсинга во время судейства | Done |
| 07 | `07-eval-perf-models` | max_retries=5 в generator (rate limit); RunConfig(max_workers=8) в RAGAS; каталог моделей FREE/CHEAP/SMART; дефолты eval — gemma-4-26b + gemma-4-31b | Done |
| 08 | `08-deepeval-migration` | Миграция с RAGAS на DeepEval; параллельная генерация ответов (ThreadPoolExecutor); async судья (AsyncOpenAI); кнопка Stop в Gradio; первый DeepEval-baseline | Done |
| 09 | `09-cross-encoder-reranker` | Cross-encoder reranker `BAAI/bge-reranker-v2-m3`; режим `hybrid+rerank` в UI; ADR-004; смена embedding-модели на `intfloat/multilingual-e5-base` с префиксами `passage:`/`query:` | Done |
| 10 | `10-eval-reliability` | Hardening судьи: `response_format=json_object` + fallback, retry с backoff, защита от `choices=None`, очистка markdown-фенсов, балансовое извлечение JSON; `DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=600`; `ErrorConfig(ignore_errors=True)`; датасет `data_set_2.json` (30 вопросов); каталог `eval-reports/`; зафиксирована ошибка в ADR-003 о ~11 вызовах/sample вместо 4 | Done |
