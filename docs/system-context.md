# System Context — rag-by-book

## Overview

CLI-приложение для вопросно-ответного взаимодействия с текстовой книгой через RAG (Retrieval-Augmented Generation). Пользователь индексирует текстовый файл, после чего задаёт вопросы — система находит релевантные фрагменты и генерирует ответ с помощью LLM.

---

## Components

### `config.py`
Централизованная конфигурация. Читает `.env` через `python-dotenv`.

| Параметр | Значение | Описание |
|---|---|---|
| `ANTHROPIC_API_KEY` | из `.env` | Ключ Anthropic API |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformers модель (поддерживает RU) |
| `LLM_MODEL` | `claude-haiku-4-5` | LLM для генерации ответов |
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
- Собирает чанки с перекрытием: если абзац не помещается в текущий чанк — сохраняет чанк, берёт последние `overlap` символов как начало следующего
- Возвращает `list[dict]` с полями `id`, `text`, `metadata` (`index`, `char_start`)

---

### `indexer.py`
Векторизация и сохранение чанков в ChromaDB.

- Lazy-инициализация `SentenceTransformer` (singleton `_model`)
- `get_embeddings(texts)` — батчевая векторизация
- `index_document(chunks)` — пересоздаёт коллекцию (удаляет старую), добавляет чанки батчами по 64
- Персистентный клиент ChromaDB: данные сохраняются в `./chroma_db`

---

### `retriever.py`
Поиск релевантных чанков по запросу.

- Векторизует запрос через тот же `get_embeddings` из `indexer`
- Запрашивает ChromaDB: `n_results=top_k`, возвращает `documents` + `distances`
- Возвращает `dict` с ключами `documents`, `distances`, `ids`

---

### `generator.py`
Генерация ответа через Anthropic API.

- `SYSTEM_PROMPT` — инструкция отвечать только по контексту, на русском
- `build_user_message(query, context_chunks)` — собирает промпт: контекст через `---`, затем вопрос
- `generate_answer(query, context_chunks)` — вызывает `claude-haiku-4-5`, `max_tokens=5000`

---

### `main.py`
CLI-интерфейс (REPL).

**Команды:**
- `/index [файл]` — загрузить и проиндексировать книгу (default: `aa.txt`)
- `/debug` — toggle: показывать найденные чанки со scores и финальный промпт
- `/topk <n>` — изменить количество чанков для контекста
- `/help` — справка
- `/exit` / `/quit` / `/q` — выход

Всё остальное — вопрос, передаётся в `ask_question()`.

---

## Data Flow

```
Текстовый файл
    → chunker.load_and_chunk()        # list[dict] чанков
    → indexer.index_document()        # ChromaDB (./chroma_db)

Вопрос пользователя
    → retriever.retrieve()            # top-K чанков из ChromaDB
    → generator.generate_answer()     # ответ от LLM
    → вывод в консоль
```

---

## Infrastructure

| Компонент | Технология |
|---|---|
| Vector DB | ChromaDB (локальная, персистентная) |
| Embeddings | sentence-transformers (локально) |
| LLM | Anthropic Claude Haiku 4.5 (API) |
| Runtime | Python 3.11+ |
| Config | python-dotenv (`.env`) |

---

## Dependencies

```
anthropic
sentence-transformers
chromadb
click
python-dotenv
```

---

## Iterations

| # | Slug | Описание | Статус |
|---|---|---|---|
| 01 | `01-rag-pipeline` | Базовый RAG CLI: chunker, indexer, retriever, generator, main | Done |
