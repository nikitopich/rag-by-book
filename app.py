import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import gradio as gr
import pandas as pd
import phoenix as px
from openai import OpenAI, AsyncOpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.client import Client as PhoenixClient
from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.configs import DisplayConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from chunker import load_and_chunk
from config import DEFAULT_MODEL, OPENROUTER_BASE_URL, TOP_K
from generator import generate_answer, build_user_message, SYSTEM_PROMPT
from indexer import index_document
from retriever import retrieve, hybrid_retrieve
from tracing import setup_tracing

phoenix_url = setup_tracing()


def _clean_llm_output(content: str | None) -> str:
    """Убирает <think>...</think> блоки reasoning-моделей и возвращает непустую строку."""
    if content is None:
        return ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


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
        return _clean_llm_output(resp.choices[0].message.content)

    async def a_generate(self, prompt: str) -> str:
        client = AsyncOpenAI(api_key=self._api_key, base_url=OPENROUTER_BASE_URL)
        resp = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return _clean_llm_output(resp.choices[0].message.content)

    def get_model_name(self) -> str:
        return self._model


def chat(message: str, history: list, api_key: str, model: str, top_k: int, use_hybrid: bool, debug: bool):
    if not api_key or not api_key.strip():
        return "Введи OpenRouter API ключ в поле выше"

    try:
        retriever_fn = hybrid_retrieve if use_hybrid else retrieve
        results = retriever_fn(message, top_k=int(top_k))
        chunks = results["documents"]

        debug_info = ""
        if debug:
            lines = ["\n\n---\n**DEBUG: найденные чанки**\n"]
            for i, (chunk_id, doc, dist) in enumerate(
                zip(results["ids"], results["documents"], results["distances"]), 1
            ):
                lines.append(f"**[{i}] {chunk_id}** | distance: {dist:.4f}\n```\n{doc}\n```\n")
            lines.append(f"\n**Промпт:**\n```\n[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{build_user_message(message, chunks)}\n```")
            debug_info = "\n".join(lines)

        answer = generate_answer(message, chunks, api_key.strip(), model)
        return answer + debug_info

    except Exception as e:
        return f"Ошибка: {type(e).__name__}: {e}"


def index_file(file, api_key: str, chunk_size: int, chunk_overlap: int):
    if file is None:
        return "Выбери файл для индексации"
    chunks = load_and_chunk(file.name, chunk_size=chunk_size, overlap=chunk_overlap)
    total = index_document(chunks)
    return f"Готово! Проиндексировано {total} чанков (size={chunk_size}, overlap={chunk_overlap}) из файла {file.name}"


def run_eval(api_key, model, judge_model, retriever_mode, dataset_file):
    if not api_key or not api_key.strip():
        yield "Введи OpenRouter API ключ", gr.update(visible=False)
        return

    # Load dataset
    if dataset_file is not None:
        with open(dataset_file.name, encoding="utf-8") as f:
            questions = json.load(f)
    else:
        try:
            with open("eval_dataset.json", encoding="utf-8") as f:
                questions = json.load(f)
        except FileNotFoundError:
            yield "Файл eval_dataset.json не найден. Загрузи датасет через поле выше.", gr.update(visible=False)
            return

    use_hybrid = retriever_mode == "hybrid"
    retriever_fn = hybrid_retrieve if use_hybrid else retrieve
    retriever_label = "hybrid (BM25 + Vector)" if use_hybrid else "vector only"

    yield f"Загружено {len(questions)} вопросов | модель: {model} | retriever: {retriever_label}\n\nГенерирую ответы...\n", gr.update(visible=False)

    # Retrieval — последовательно (ChromaDB не thread-safe)
    retrieved = []
    for item in questions:
        results = retriever_fn(item["question"], top_k=TOP_K)
        retrieved.append({
            "question": item["question"],
            "reference": item.get("reference", ""),
            "contexts": results["documents"],
        })

    yield f"✓ Retrieval готов для {len(questions)} вопросов\n\nГенерирую ответы параллельно...", gr.update(visible=False)

    # Генерация — параллельно (HTTP-запросы независимы)
    def generate_one(entry: dict) -> LLMTestCase:
        answer = generate_answer(entry["question"], entry["contexts"], api_key.strip(), model)
        return LLMTestCase(
            input=entry["question"],
            actual_output=answer,
            expected_output=entry["reference"],
            retrieval_context=entry["contexts"],
        )

    test_cases = [None] * len(retrieved)
    done_count = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(generate_one, entry): i for i, entry in enumerate(retrieved)}
        for future in as_completed(futures):
            i = futures[future]
            test_cases[i] = future.result()
            done_count += 1
            yield f"Генерирую ответы... {done_count}/{len(questions)}\n✓ {retrieved[i]['question'][:70]}", gr.update(visible=False)

    log_lines = [f"✓ Сгенерировано {len(questions)} ответов"]

    log_lines.append(f"\nОцениваю с DeepEval (судья: {judge_model})...")
    yield "\n".join(log_lines), gr.update(visible=False)

    judge = OpenRouterLLM(model=judge_model, api_key=api_key.strip())
    metrics = [
        FaithfulnessMetric(threshold=0.5, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.5, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=0.5, model=judge, include_reason=True),
        ContextualRecallMetric(threshold=0.5, model=judge, include_reason=True),
    ]

    # Uninstrument to avoid flooding Phoenix with DeepEval internal LLM calls
    OpenAIInstrumentor().uninstrument()
    try:
        eval_result = deepeval_evaluate(test_cases, metrics, display_config=DisplayConfig(print_results=False))
    finally:
        OpenAIInstrumentor().instrument()

    metric_scores: dict[str, list[float]] = {}
    rows = []
    for tr in eval_result.test_results:
        row = {"user_input": tr.input, "response": tr.actual_output}
        for md in tr.metrics_data:
            if md.score is not None:
                row[md.name] = md.score
                metric_scores.setdefault(md.name, []).append(md.score)
        rows.append(row)

    scores_lines = ["\n" + "=" * 54, "РЕЗУЛЬТАТЫ DEEPEVAL", "=" * 54]
    for metric_name, scores in metric_scores.items():
        avg = sum(scores) / len(scores)
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        scores_lines.append(f"{metric_name:<30} {bar}  {avg:.3f}")
    scores_lines.append("=" * 54)

    df = pd.DataFrame(rows)

    # Save to Phoenix datasets
    try:
        client = PhoenixClient()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_slug = model.split("/")[-1]
        dataset_name = f"eval-{ts}-{model_slug}-{retriever_mode}"
        client.datasets.create_dataset(
            name=dataset_name,
            dataframe=df,
            input_keys=["user_input"],
            output_keys=["response"],
        )
        scores_lines.append(f"\n✅ Датасет сохранён в Phoenix: «{dataset_name}»")
        scores_lines.append(f"   Открыть: {phoenix_url}datasets")
    except Exception as e:
        scores_lines.append(f"\n⚠️  Не удалось сохранить в Phoenix: {e}")

    yield "\n".join(log_lines + scores_lines), gr.update(value=df, visible=True)


with gr.Blocks(title="RAG-чат с книгой") as demo:
    gr.Markdown(
        f"# RAG-чат с книгой\n"
        f"Задавай вопросы по проиндексированной книге. "
        f"Phoenix: [{phoenix_url}]({phoenix_url})"
    )

    api_key_input = gr.Textbox(
        label="OpenRouter API Key",
        type="password",
        placeholder="sk-or-v1-...",
    )

    with gr.Tabs():
        with gr.Tab("💬 Чат"):
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=[
                        "--- FREE ---",
                        "google/gemma-4-26b-a4b-it:free",
                        "google/gemma-4-31b-it:free",
                        "qwen/qwen3-next-80b-a3b-instruct:free",
                        "deepseek/deepseek-r1:free",
                        "meta-llama/llama-3.3-70b-instruct:free",
                        "--- CHEAP ---",
                        "google/gemma-4-26b-a4b-it",
                        "google/gemma-4-31b-it",
                        "deepseek/deepseek-v4-flash",
                        "openai/gpt-4o-mini",
                        "qwen/qwen3.6-flash",
                        "--- SMART ---",
                        "openai/gpt-4o",
                        "anthropic/claude-sonnet-4-5",
                        "google/gemini-2.5-flash-preview",
                        "google/gemini-2.5-pro-preview",
                        "deepseek/deepseek-r1",
                        "deepseek/deepseek-r1-0528",
                        "qwen/qwen3.6-plus",
                        "moonshotai/kimi-k2",
                    ],
                    value=DEFAULT_MODEL,
                    label="Модель",
                    scale=2,
                )
                top_k_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top-K фрагментов", scale=1)
                retriever_toggle = gr.Radio(
                    choices=["hybrid", "vector"],
                    value="hybrid",
                    label="Retriever",
                )
                debug_checkbox = gr.Checkbox(label="Debug-режим", value=False)

            gr.ChatInterface(
                fn=chat,
                additional_inputs=[api_key_input, model_selector, top_k_slider, retriever_toggle, debug_checkbox],
                chatbot=gr.Chatbot(height=500),
                textbox=gr.Textbox(
                    placeholder="Задай вопрос по книге...",
                    container=False,
                    submit_btn="Отправить",
                ),
            )

        with gr.Tab("📥 Индексация"):
            gr.Markdown("Загрузи текстовый файл книги для индексации в ChromaDB.")
            file_input = gr.File(label="Файл книги (.txt)", file_types=[".txt"])
            with gr.Row():
                chunk_size_slider = gr.Slider(minimum=100, maximum=2000, value=500, step=50, label="Chunk size (символы)")
                chunk_overlap_slider = gr.Slider(minimum=0, maximum=500, value=100, step=25, label="Chunk overlap (символы)")
            index_btn = gr.Button("Индексировать", variant="primary")
            index_output = gr.Textbox(label="Результат", interactive=False)
            index_btn.click(fn=index_file, inputs=[file_input, api_key_input, chunk_size_slider, chunk_overlap_slider], outputs=index_output)

        with gr.Tab("🔬 Оценка"):
            gr.Markdown(
                "Запускает DeepEval-оценку и сохраняет результаты в Phoenix как датасет.\n"
                f"Посмотреть датасеты: [{phoenix_url}datasets]({phoenix_url}datasets)"
            )

            with gr.Row():
                eval_answer_model_dropdown = gr.Dropdown(
                    choices=[
                        "--- FREE ---",
                        "google/gemma-4-26b-a4b-it:free",
                        "google/gemma-4-31b-it:free",
                        "qwen/qwen3-next-80b-a3b-instruct:free",
                        "deepseek/deepseek-r1:free",
                        "meta-llama/llama-3.3-70b-instruct:free",
                        "--- CHEAP ---",
                        "google/gemma-4-26b-a4b-it",
                        "google/gemma-4-31b-it",
                        "deepseek/deepseek-v4-flash",
                        "openai/gpt-4o-mini",
                        "qwen/qwen3.6-flash",
                        "--- SMART ---",
                        "openai/gpt-4o",
                        "anthropic/claude-sonnet-4-5",
                        "google/gemini-2.5-flash-preview",
                        "deepseek/deepseek-r1",
                        "qwen/qwen3.6-plus",
                    ],
                    value="google/gemma-4-26b-a4b-it",
                    label="Модель-ответчик (answer model)",
                )
                judge_model_dropdown = gr.Dropdown(
                    choices=[
                        "--- FREE ---",
                        "google/gemma-4-31b-it:free",
                        "deepseek/deepseek-r1:free",
                        "--- CHEAP ---",
                        "openai/gpt-4o-mini",
                        "google/gemma-4-31b-it",
                        "--- SMART ---",
                        "openai/gpt-4o",
                        "deepseek/deepseek-r1",
                        "deepseek/deepseek-v4-pro",
                        "anthropic/claude-sonnet-4-5",
                        "google/gemini-2.5-flash-preview",
                    ],
                    value="google/gemma-4-31b-it",
                    label="Судья (judge model)",
                )
                eval_retriever_radio = gr.Radio(
                    choices=["hybrid", "vector"],
                    value="hybrid",
                    label="Retriever",
                )

            eval_dataset_file = gr.File(
                label="Датасет JSON (опционально — по умолчанию eval_dataset.json)",
                file_types=[".json"],
            )

            with gr.Row():
                run_eval_btn = gr.Button("▶ Запустить оценку", variant="primary", size="lg")
                stop_eval_btn = gr.Button("⏹ Остановить", variant="stop", size="lg")

            eval_log = gr.Textbox(
                label="Прогресс",
                interactive=False,
                lines=15,
                max_lines=30,
            )
            eval_table = gr.Dataframe(
                label="Детальные результаты по вопросам",
                visible=False,
                wrap=True,
            )

            run_event = run_eval_btn.click(
                fn=run_eval,
                inputs=[api_key_input, eval_answer_model_dropdown, judge_model_dropdown, eval_retriever_radio, eval_dataset_file],
                outputs=[eval_log, eval_table],
            )
            stop_eval_btn.click(fn=None, cancels=[run_event])


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
