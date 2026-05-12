# ADR-003: Миграция с RAGAS на DeepEval для снижения стоимости оценки

**Статус:** Принято  
**Дата:** 2026-05-12  
**Автор:** Nikita Babchenko

---

## Контекст

Текущий пайплайн оценки (`eval.py`) использует **RAGAS** для измерения четырёх метрик:

| Метрика | Как RAGAS считает | LLM-вызовов на 1 sample |
|---|---|---|
| faithfulness | Извлечение claims → NLI для каждого claim отдельно | N claims + 1 |
| answer_relevancy | Генерирует несколько вопросов из ответа → embedding similarity | 3–5 вызовов + 1 embedding |
| context_precision | Оценивает каждый чанк из context по отдельности | K (= TOP_K = 5) |
| context_recall | Разбивает reference на предложения → проверяет каждое | M sentences |

**Итого: ~15–20 LLM-вызовов на один тестовый вопрос.** На датасете из 10 вопросов = 150–200 вызовов к судье (gpt-4o или deepseek-v4-pro).

Дополнительные боли RAGAS:
- Для `answer_relevancy` требует отдельный embedding-провайдер (текущее решение: `LangchainEmbeddingsWrapper` + `text-embedding-3-small` через OpenRouter)
- Зависит от `langchain-openai` как прослойки, которая не нужна нигде кроме eval
- Частые ошибки при работе через OpenRouter (несовместимость форматов streaming/tool-use)

---

## Рассмотренные варианты

### Вариант 1: Оставить RAGAS, оптимизировать промпты
Уменьшить `max_workers`, снизить `timeout` — не решает фундаментальную проблему числа вызовов.

### Вариант 2: Написать кастомные метрики на чистом Python + LLM
Полный контроль над промптами и количеством вызовов, но значительные затраты на написание и поддержку.

### Вариант 3: Перейти на DeepEval ✓

**DeepEval** использует архитектуру **G-Eval** (LLM-as-judge с chain-of-thought) — **один вызов судьи на метрику на sample**, независимо от числа чанков или длины ответа.

Сравнение вызовов для 1 sample × 4 метрики:

| Фреймворк | Вызовов судьи | Embedding-вызовов |
|---|---|---|
| RAGAS | 15–20 | 3–5 |
| DeepEval | 4 (1 на метрику) | 0 |

**Соответствие метрик:**

| RAGAS | DeepEval | Примечание |
|---|---|---|
| faithfulness | `FaithfulnessMetric` | аналогично |
| answer_relevancy | `AnswerRelevancyMetric` | без embedding, только LLM |
| context_precision | `ContextualPrecisionMetric` | аналогично |
| context_recall | `ContextualRecallMetric` | аналогично |

Дополнительные преимущества:
- Не требует `langchain-openai` и `datasets` — можно удалить из зависимостей
- Нативная поддержка OpenAI-compatible API через `base_url`
- Встроенная интеграция с Phoenix через OpenTelemetry (те же трейсы что уже есть)
- CLI: `deepeval test run` — можно интегрировать в CI

---

## Решение

Переписать `eval.py` на **DeepEval**:

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
```

**Конфигурация судьи** через `DeepEvalBaseLLM` с OpenAI-compatible endpoint (OpenRouter):

```python
class OpenRouterLLM(DeepEvalBaseLLM):
    def generate(self, prompt: str) -> str:
        client = OpenAI(api_key=self._api_key, base_url=OPENROUTER_BASE_URL)
        ...
    async def a_generate(self, prompt: str) -> str:  # настоящий async через AsyncOpenAI
        client = AsyncOpenAI(api_key=self._api_key, base_url=OPENROUTER_BASE_URL)
        ...
```

**Параметры метрик:**
- `threshold=0.5` — стандартный порог pass/fail
- `model=judge` — экземпляр `OpenRouterLLM`
- `include_reason=True` — логировать объяснения для отладки
- `display_config=DisplayConfig(print_results=False)` — вывод управляется вручную

**Параллелизация генерации ответов:**

ChromaDB не thread-safe — `PersistentClient` нельзя создавать из нескольких потоков. Решение: retrieval последовательно, генерация параллельно через `ThreadPoolExecutor(max_workers=5)`.

```
1. Retrieval (последовательно, ~секунды)
2. generate_answer × N (ThreadPoolExecutor, 5 потоков)
3. DeepEval evaluate (async_mode=True, все метрики параллельно)
```

**Изменения в `requirements.txt`:**
```diff
- ragas
- langchain-openai
- datasets
+ deepeval
```

---

## Отклонённые альтернативы

- **TruLens** — хорошая визуализация, но аналогичная RAGAS декомпозированная структура вызовов; приоритет на LlamaIndex, поддержка ChromaDB вторична
- **Кастомные метрики** — избыточно на учебном этапе; DeepEval даёт достаточный контроль через параметр `model`

---

## Последствия

**Положительные:**
- Снижение числа LLM-вызовов в 4–5× → прямая экономия на API
- Удаление двух тяжёлых зависимостей (`langchain-openai`, `datasets`)
- Единообразная схема: 1 sample → 1 вызов на метрику → легко предсказывать стоимость прогона
- Числовые оценки совместимы по смыслу с текущими RAGAS метриками — можно сравнивать тренды

**Риски:**
- Абсолютные значения метрик изменятся (разные промпты); нельзя напрямую сравнивать с baseline из ADR-002
- G-Eval чувствителен к качеству модели-судьи; бесплатные/дешёвые модели могут давать нестабильные оценки
- DeepEval активно развивается, API между версиями менялось — зафиксировать версию в `requirements.txt`

---

## Метрики успеха

Первый прогон после миграции — новый baseline установлен:

| Метрика | Baseline (ADR-002) | Новый baseline | Статус |
|---|---|---|---|
| Faithfulness | 1.000 | 0.878 | ✅ |
| Answer Relevancy | 0.487 | 0.741 | ✅ |
| Contextual Precision | 0.192 | 0.422 | ✅ |
| Contextual Recall | 0.542 | 0.444 | ✅ |

**Конфигурация замера:** answer model `google/gemma-4-26b-a4b-it`, judge `google/gemma-4-31b-it`, retriever `hybrid`.

> ⚠️ Прямое сравнение с ADR-002 baseline некорректно: разные фреймворки (RAGAS vs DeepEval), разные промпты судьи, разная модель-ответчик. Эти цифры — новая точка отсчёта.

**Вывод:** Миграция успешна. Все метрики возвращают числовые значения. Answer Relevancy вырос (+52% vs ADR-002), что может объясняться как реальным улучшением, так и разницей в методологии оценки. Faithfulness снизился с 1.0 — вероятно, G-Eval строже к галлюцинациям, чем RAGAS NLI-подход. Contextual Precision вырос вдвое (+120% vs ADR-002) — следующий шаг: reranker (cross-encoder).

---

## Ссылки

- [DeepEval docs](https://docs.confident-ai.com)
- [G-Eval paper (Liu et al., 2023)](https://arxiv.org/abs/2303.16634)
- [DeepEval GitHub](https://github.com/confident-ai/deepeval)
