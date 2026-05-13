"""
Phoenix Evals для RAG пайплайна — аналог eval.py, но через phoenix.evals.

Использование:
    python eval_phoenix.py --api-key sk-or-v1-...

Метрики (аналог RAGAS):
    - faithfulness        → ответ основан только на контексте (не галлюцинирует)
    - answer_relevancy    → ответ по существу вопроса
    - context_relevance   → найденный контекст релевантен вопросу
    - answer_correctness  → ответ совпадает с эталонным ответом (reference)
"""

import argparse
import json
import getpass
from datetime import datetime

import pandas as pd
from phoenix.evals import create_classifier, evaluate_dataframe, LLM

from retriever import retrieve, hybrid_retrieve
from generator import generate_answer
from config import DEFAULT_MODEL, OPENROUTER_BASE_URL, TOP_K
from tracing import setup_tracing


FAITHFULNESS_TEMPLATE = """\
You are evaluating whether an AI assistant's answer is faithful to the provided context.
A faithful answer contains ONLY information that is explicitly supported by the context — it does not introduce new facts, names, or claims that are absent from the context.

Question: {question}
Context: {context}
Answer: {answer}

Is the answer faithful to the context?
"""

ANSWER_RELEVANCY_TEMPLATE = """\
You are evaluating whether an AI assistant's answer directly addresses the question asked.
A relevant answer speaks to what was asked. An irrelevant answer goes off-topic or ignores the question.

Question: {question}
Answer: {answer}

Is the answer relevant to the question?
"""

CONTEXT_RELEVANCE_TEMPLATE = """\
You are evaluating whether the retrieved context is useful for answering the question.
Relevant context contains information that helps answer the question.
Irrelevant context does not help answer the question.

Question: {question}
Context: {context}

Is the retrieved context relevant to the question?
"""

ANSWER_CORRECTNESS_TEMPLATE = """\
You are evaluating whether an AI assistant's answer matches the ground-truth reference answer.
A correct answer conveys the same key facts as the reference. An incorrect answer contradicts the reference or omits critical facts.

Question: {question}
Reference answer: {reference}
AI answer: {answer}

Is the AI answer correct compared to the reference?
"""


def build_dataframe(
    questions: list[dict],
    api_key: str,
    model: str,
    use_hybrid: bool = True,
) -> pd.DataFrame:
    retriever_fn = hybrid_retrieve if use_hybrid else retrieve
    rows = []
    for i, item in enumerate(questions, 1):
        question = item["question"]
        reference = item.get("reference", "")
        print(f"  [{i}/{len(questions)}] {question[:70]}")
        results = retriever_fn(question, top_k=TOP_K)
        contexts = results["documents"]
        answer = generate_answer(question, contexts, api_key, model)
        rows.append({
            "question": question,
            "answer": answer,
            "context": "\n\n---\n\n".join(contexts),
            "reference": reference,
        })
    return pd.DataFrame(rows)


def parse_avg_score(series: pd.Series) -> float | None:
    """Вытаскивает средний числовой score из колонки с JSON-сериализованными Score-объектами."""
    scores = []
    for val in series:
        if val is None:
            continue
        try:
            parsed = json.loads(val) if isinstance(val, str) else val
            if isinstance(parsed, list):
                for s in parsed:
                    if isinstance(s, dict) and s.get("score") is not None:
                        scores.append(float(s["score"]))
            elif isinstance(parsed, dict) and parsed.get("score") is not None:
                scores.append(float(parsed["score"]))
        except Exception:
            pass
    return sum(scores) / len(scores) if scores else None


def main():
    parser = argparse.ArgumentParser(description="Phoenix evals для RAG пайплайна")
    parser.add_argument("--api-key", help="OpenRouter API key")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Модель для генерации ответов")
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-4o-mini",
        help="Модель-судья для Phoenix evals (должна поддерживать tool calling)",
    )
    parser.add_argument("--dataset", default="eval_dataset.json", help="Путь к датасету")
    parser.add_argument("--output", default=None, help="Сохранить результаты в CSV (опционально)")
    parser.add_argument(
        "--retriever",
        choices=["hybrid", "vector"],
        default="hybrid",
        help="Режим retrieval: hybrid (BM25+Vector) или vector",
    )
    args = parser.parse_args()

    phoenix_url = setup_tracing()
    print(f"Phoenix UI: {phoenix_url}\n")

    api_key = args.api_key or getpass.getpass("OpenRouter API Key: ").strip()

    with open(args.dataset, encoding="utf-8") as f:
        questions = json.load(f)

    use_hybrid = args.retriever == "hybrid"
    retriever_label = "hybrid (BM25 + Vector)" if use_hybrid else "vector only"
    print(f"Генерирую ответы для {len(questions)} вопросов")
    print(f"  модель: {args.model}  |  retriever: {retriever_label}\n")
    df = build_dataframe(questions, api_key, args.model, use_hybrid=use_hybrid)

    # Phoenix LLM с OpenRouter (совместим с OpenAI API)
    judge_llm = LLM(
        provider="openai",
        model=args.judge_model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )

    evaluators = [
        create_classifier(
            name="faithfulness",
            prompt_template=FAITHFULNESS_TEMPLATE,
            llm=judge_llm,
            choices={"faithful": 1.0, "unfaithful": 0.0},
        ),
        create_classifier(
            name="answer_relevancy",
            prompt_template=ANSWER_RELEVANCY_TEMPLATE,
            llm=judge_llm,
            choices={"relevant": 1.0, "irrelevant": 0.0},
        ),
        create_classifier(
            name="context_relevance",
            prompt_template=CONTEXT_RELEVANCE_TEMPLATE,
            llm=judge_llm,
            choices={"relevant": 1.0, "irrelevant": 0.0},
        ),
        create_classifier(
            name="answer_correctness",
            prompt_template=ANSWER_CORRECTNESS_TEMPLATE,
            llm=judge_llm,
            choices={"correct": 1.0, "incorrect": 0.0},
        ),
    ]

    print(f"\nОцениваю с помощью Phoenix evals (судья: {args.judge_model})...")
    result_df = evaluate_dataframe(df, evaluators)

    score_cols = [c for c in result_df.columns if c.endswith("_score")]

    print("\n" + "=" * 52)
    print("РЕЗУЛЬТАТЫ PHOENIX EVALS")
    print("=" * 52)
    for col in score_cols:
        avg = parse_avg_score(result_df[col])
        if avg is not None:
            metric = col.replace("_score", "")
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            print(f"{metric:<22} {bar}  {avg:.3f}")
    print("=" * 52)
    print("Значения от 0 до 1. Чем выше — тем лучше.\n")

    output = args.output or f"eval_phoenix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result_df.to_csv(output, index=False)
    print(f"Детальные результаты сохранены в {output}")


if __name__ == "__main__":
    main()
