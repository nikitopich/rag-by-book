"""
Оценка качества RAG пайплайна с помощью DeepEval.

Использование:
    python eval.py --api-key sk-or-v1-...

Перед запуском:
    1. Убедись что книга проиндексирована (запусти /index через main.py или app.py)
    2. Заполни eval_dataset.json своими вопросами и правильными ответами

Метрики:
    - Faithfulness:          ответ основан только на найденных чанках (не галлюцинирует)
    - Answer Relevancy:      ответ по существу вопроса
    - Contextual Precision:  найденные чанки действительно релевантны вопросу
    - Contextual Recall:     все нужные факты присутствуют в найденных чанках
"""

import argparse
import json
import getpass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from openai import OpenAI, AsyncOpenAI
from deepeval import evaluate
from deepeval.evaluate.configs import DisplayConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from retriever import retrieve, hybrid_retrieve
from generator import generate_answer
from config import DEFAULT_MODEL, OPENROUTER_BASE_URL, TOP_K
from tracing import setup_tracing


class OpenRouterLLM(DeepEvalBaseLLM):
    """Обёртка над OpenRouter для использования как судья в DeepEval."""

    def __init__(self, model: str, api_key: str):
        self._model = model
        self._api_key = api_key

    def load_model(self) -> OpenAI:
        return OpenAI(api_key=self._api_key, base_url=OPENROUTER_BASE_URL)

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        resp = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        client = AsyncOpenAI(api_key=self._api_key, base_url=OPENROUTER_BASE_URL)
        resp = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    def get_model_name(self) -> str:
        return self._model


def build_test_cases(
    questions: list[dict], api_key: str, model: str, use_hybrid: bool = True, max_workers: int = 5
) -> list[LLMTestCase]:
    retriever_fn = hybrid_retrieve if use_hybrid else retrieve

    # Retrieval — последовательно (ChromaDB не thread-safe)
    retrieved = []
    for item in questions:
        results = retriever_fn(item["question"], top_k=TOP_K)
        retrieved.append({
            "question": item["question"],
            "reference": item.get("reference", ""),
            "contexts": results["documents"],
        })

    # Генерация — параллельно (HTTP-запросы независимы)
    def generate_one(entry: dict) -> LLMTestCase:
        answer = generate_answer(entry["question"], entry["contexts"], api_key, model)
        return LLMTestCase(
            input=entry["question"],
            actual_output=answer,
            expected_output=entry["reference"],
            retrieval_context=entry["contexts"],
        )

    test_cases = [None] * len(retrieved)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_one, entry): i for i, entry in enumerate(retrieved)}
        for future in as_completed(futures):
            i = futures[future]
            test_cases[i] = future.result()
            done = sum(tc is not None for tc in test_cases)
            print(f"[{done}/{len(questions)}] {retrieved[i]['question']}")
    return test_cases


def main():
    parser = argparse.ArgumentParser(description="DeepEval оценка для RAG пайплайна")
    parser.add_argument("--api-key", help="OpenRouter API key")
    parser.add_argument("--model", default="openai/gpt-4o", help="Модель для генерации ответов")
    parser.add_argument("--judge-model", default="openai/gpt-4o", help="Модель-судья для DeepEval")
    parser.add_argument("--dataset", default="eval_dataset.json", help="Путь к датасету")
    parser.add_argument("--output", default=None, help="Сохранить результаты в CSV (опционально)")
    parser.add_argument(
        "--retriever",
        choices=["hybrid", "vector"],
        default="hybrid",
        help="Режим retrieval: hybrid (BM25+Vector, по умолчанию) или vector",
    )
    args = parser.parse_args()

    phoenix_url = setup_tracing()
    print(f"Phoenix UI: {phoenix_url}\n")

    api_key = args.api_key or getpass.getpass("OpenRouter API Key: ").strip()

    with open(args.dataset, encoding="utf-8") as f:
        questions = json.load(f)

    use_hybrid = args.retriever == "hybrid"
    retriever_label = "hybrid (BM25 + Vector)" if use_hybrid else "vector only"
    print(f"\nГенерирую ответы для {len(questions)} вопросов (модель: {args.model}, retriever: {retriever_label})...")
    test_cases = build_test_cases(questions, api_key, args.model, use_hybrid=use_hybrid)

    judge = OpenRouterLLM(model=args.judge_model, api_key=api_key)
    metrics = [
        FaithfulnessMetric(threshold=0.5, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.5, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=0.5, model=judge, include_reason=True),
        ContextualRecallMetric(threshold=0.5, model=judge, include_reason=True),
    ]

    print(f"\nОцениваю с помощью DeepEval (судья: {args.judge_model})...")
    results = evaluate(test_cases, metrics, display_config=DisplayConfig(print_results=False))

    metric_scores: dict[str, list[float]] = {}
    rows = []
    for tr in results.test_results:
        row = {"question": tr.input}
        for md in tr.metrics_data:
            if md.score is not None:
                row[md.name] = md.score
                metric_scores.setdefault(md.name, []).append(md.score)
        rows.append(row)

    print("\n" + "=" * 54)
    print("РЕЗУЛЬТАТЫ DEEPEVAL")
    print("=" * 54)
    for metric_name, scores in metric_scores.items():
        avg = sum(scores) / len(scores)
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"{metric_name:<30} {bar}  {avg:.3f}")
    print("=" * 54)
    print("Значения от 0 до 1. Чем выше — тем лучше.\n")

    output = args.output or f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Детальные результаты сохранены в {output}")


if __name__ == "__main__":
    main()
