# ADR-002: Hybrid Search (BM25 + Vector) для улучшения retrieval

**Статус:** Принято  
**Дата:** 2026-05-12  
**Автор:** Nikita Babchenko

---

## Контекст

После добавления RAGAS-оценки (ADR-001) получены следующие метрики текущего пайплайна:

| Метрика | Значение | Интерпретация |
|---|---|---|
| faithfulness | 0.844 | хорошо — модель честна к контексту |
| answer_relevancy | 0.089 | критично низко |
| context_precision | 0.243 | ретривер тащит много нерелевантных чанков |
| context_recall | 0.367 | ретривер пропускает нужные чанки |

Текущий retriever (`retriever.py`) использует **только векторный поиск** через ChromaDB:
- Запрос преобразуется в embedding
- Ищутся top-K чанков по косинусному расстоянию
- Результат подаётся в LLM

**Проблема:** Векторный поиск хорошо ловит семантическую близость, но плохо работает с:
- Точными терминами, именами, датами
- Редкими словами, которые "размазываются" в embedding-пространстве
- Специфичными фразами из книги (термины, названия глав)

Следствие: context_precision = 0.243 и context_recall = 0.367 → неверный контекст → answer_relevancy = 0.089.

---

## Рассмотренные варианты

### Вариант 1: Тюнинг чанкинга
Уменьшить размер чанков, добавить overlap — может помочь recall, но не решает проблему нерелевантных результатов при точных запросах.

### Вариант 2: Reranker (cross-encoder)
Добавить cross-encoder поверх retrieved чанков для переранжирования. Эффективно, но:
- Требует отдельной тяжёлой модели
- Решает precision, но не recall

### Вариант 3: Hybrid Search (BM25 + Vector) ✓
Комбинировать два принципиально разных подхода:
- **BM25** — keyword-based, точен на терминах и редких словах
- **Vector** — semantic, хорош для смысловой близости
- **RRF (Reciprocal Rank Fusion)** — простая функция слияния без дополнительного обучения

Закрывает оба провала: precision (BM25 фильтрует мусор) и recall (два сигнала вместо одного).

---

## Решение

Реализовать **Hybrid Search с Reciprocal Rank Fusion**:

```
query
  ├─→ BM25 (rank_bm25) → top-N с BM25-скорами
  ├─→ Vector (ChromaDB) → top-N с косинусными скорами
  └─→ RRF merge → финальный top-K для LLM
```

**Формула RRF:**
```
RRF_score(doc) = Σ 1 / (k + rank_i)
```
где `k=60` (стандартное значение), `rank_i` — позиция документа в i-м ранкере.

**Реализация:**
1. Добавить `rank_bm25` в `requirements.txt`
2. Обновить `retriever.py` — добавить функцию `hybrid_retrieve`
3. BM25-индекс строить из корпуса чанков в памяти (не персистировать — корпус небольшой)
4. Оставить `retrieve` как fallback для сравнения метрик

**Параметры:**
- `TOP_K = 5` — финальное количество чанков для LLM (без изменений)
- `bm25_top_n = 10`, `vector_top_n = 10` — кандидаты перед RRF
- `rrf_k = 60` — стандартный параметр

---

## Отклонённые альтернативы

- **Qdrant с встроенным hybrid search** — смена векторного хранилища избыточна, ChromaDB достаточен для учебного масштаба
- **LangChain EnsembleRetriever** — скрывает детали реализации; для учёбы важно понимать RRF напрямую
- **Только reranker** — решает precision, но не recall; hybrid + reranker — следующий шаг после верификации улучшений

---

## Последствия

**Положительные:**
- Ожидаемый рост context_precision и context_recall → за ними вырастет answer_relevancy
- BM25 строится на лету — нет дополнительной персистентной инфраструктуры
- `rank_bm25` — лёгкая зависимость (~20KB), без тяжёлых моделей
- Можно A/B сравнивать старый и новый retriever через RAGAS

**Риски:**
- BM25 работает на токенах без учёта морфологии — для русскоязычных текстов может потребоваться лемматизация
- Если книга на русском — добавить `pymorphy3` для нормализации перед BM25-индексацией

---

## Метрики успеха

После реализации запустить `eval.py` и сравнить с baseline:

| Метрика | Baseline | Цель | Результат | Статус |
|---|---|---|---|---|
| context_precision | 0.243 | > 0.400 | 0.192 | ❌ |
| context_recall | 0.367 | > 0.500 | 0.542 | ✅ |
| answer_relevancy | 0.089 | > 0.300 | 0.487 | ✅ |
| faithfulness | 0.844 | ≥ 0.800 | 1.000 | ✅ |

**Конфигурация замера:** answer model `google/gemma-4-31b-it`, judge `deepseek/deepseek-v4-pro`, embedding `intfloat/multilingual-e5-base`, retriever `hybrid`.

> ⚠️ Baseline мерялся с судьёй `openai/gpt-4o`, результаты — с `deepseek/deepseek-v4-pro`. Прямое сравнение приблизительно.

**Вывод:** Замена embedding-модели на `multilingual-e5-base` дала основной прирост — особенно по `context_recall` (+48%) и `answer_relevancy` (+447%). `context_precision` не достиг цели; следующий шаг — reranker (cross-encoder) поверх hybrid retrieval.

---

## Ссылки

- [rank_bm25 (PyPI)](https://pypi.org/project/rank-bm25/)
- [Reciprocal Rank Fusion — оригинальная статья (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [ChromaDB docs — query](https://docs.trychroma.com/reference/py-collection#query)
