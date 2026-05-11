"""
Оценка качества RAG пайплайна с помощью RAGAS.

Использование:
    python eval.py --api-key sk-or-v1-...

Перед запуском:
    1. Убедись что книга проиндексирована (запусти /index через main.py или app.py)
    2. Заполни eval_dataset.json своими вопросами и правильными ответами

Метрики:
    - faithfulness:       ответ основан только на найденных чанках (не галлюцинирует)
    - answer_relevancy:   ответ по существу вопроса
    - context_precision:  найденные чанки действительно релевантны вопросу
    - context_recall:     все нужные факты присутствуют в найденных чанках
"""

import argparse
import json
import getpass
from datetime import datetime

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from retriever import retrieve, hybrid_retrieve
from generator import generate_answer
from config import DEFAULT_MODEL, OPENROUTER_BASE_URL, TOP_K
from tracing import setup_tracing


def build_dataset(questions: list[dict], api_key: str, model: str, use_hybrid: bool = True) -> EvaluationDataset:
    retriever_fn = hybrid_retrieve if use_hybrid else retrieve
    samples = []
    for i, item in enumerate(questions, 1):
        question = item["question"]
        reference = item.get("reference", "")

        print(f"[{i}/{len(questions)}] {question}")

        results = retriever_fn(question, top_k=TOP_K)
        contexts = results["documents"]
        answer = generate_answer(question, contexts, api_key, model)

        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference,
        ))

    return EvaluationDataset(samples=samples)


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation для RAG пайплайна")
    parser.add_argument("--api-key", help="OpenRouter API key")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Модель для генерации ответов")
    parser.add_argument("--judge-model", default="deepseek/deepseek-v4-pro", help="Модель-судья для RAGAS")
    parser.add_argument("--dataset", default="eval_dataset.json", help="Путь к датасету")
    parser.add_argument("--output", default=None, help="Сохранить результаты в CSV (опционально)")
    parser.add_argument("--retriever", choices=["hybrid", "vector"], default="hybrid",
                        help="Режим retrieval: hybrid (BM25+Vector, по умолчанию) или vector")
    args = parser.parse_args()

    phoenix_url = setup_tracing()
    print(f"Phoenix UI: {phoenix_url}\n")

    api_key = args.api_key or getpass.getpass("OpenRouter API Key: ").strip()

    with open(args.dataset, encoding="utf-8") as f:
        questions = json.load(f)

    use_hybrid = args.retriever == "hybrid"
    retriever_label = "hybrid (BM25 + Vector)" if use_hybrid else "vector only"
    print(f"\nГенерирую ответы для {len(questions)} вопросов (модель: {args.model}, retriever: {retriever_label})...")
    dataset = build_dataset(questions, api_key, args.model, use_hybrid=use_hybrid)

    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        model=args.judge_model,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
    ))

    print(f"\nОцениваю с помощью RAGAS (судья: {args.judge_model})...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        show_progress=True,
    )

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ RAGAS")
    print("=" * 50)
    for metric, score in result._repr_dict.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"{metric:<22} {bar}  {score:.3f}")
    print("=" * 50)
    print("Значения от 0 до 1. Чем выше — тем лучше.\n")

    output = args.output or f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result.to_pandas().to_csv(output, index=False)
    print(f"Детальные результаты сохранены в {output}")


if __name__ == "__main__":
    main()
