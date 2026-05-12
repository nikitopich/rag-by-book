import json
from datetime import datetime

import gradio as gr
import pandas as pd
import phoenix as px
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.client import Client as PhoenixClient
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.run_config import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

from chunker import load_and_chunk
from config import DEFAULT_MODEL, OPENROUTER_BASE_URL, TOP_K
from generator import generate_answer, build_user_message, SYSTEM_PROMPT
from indexer import index_document
from retriever import retrieve, hybrid_retrieve
from tracing import setup_tracing

phoenix_url = setup_tracing()


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

    samples = []
    log_lines = []
    for i, item in enumerate(questions, 1):
        question = item["question"]
        reference = item.get("reference", "")

        results = retriever_fn(question, top_k=TOP_K)
        contexts = results["documents"]
        answer = generate_answer(question, contexts, api_key.strip(), model)

        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference,
        ))

        log_lines.append(f"[{i}/{len(questions)}] ✓ {question[:70]}")
        yield "\n".join(log_lines) + "\n", gr.update(visible=False)

    log_lines.append(f"\nОцениваю с RAGAS (судья: {judge_model})...")
    yield "\n".join(log_lines), gr.update(visible=False)

    # Qwen3 models enable chain-of-thought thinking by default — RAGAS can't parse
    # <think>...</think> blocks, resulting in zero scores. Disable it for the judge.
    judge_extra = {}
    if "qwen3" in judge_model.lower():
        judge_extra = {"model_kwargs": {"extra_body": {"thinking": False}}}

    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        model=judge_model,
        openai_api_key=api_key.strip(),
        openai_api_base=OPENROUTER_BASE_URL,
        **judge_extra,
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key.strip(),
        openai_api_base=OPENROUTER_BASE_URL,
    ))

    # Uninstrument to avoid flooding Phoenix with RAGAS internal LLM calls (~25/question)
    OpenAIInstrumentor().uninstrument()
    try:
        result = evaluate(
            dataset=EvaluationDataset(samples),
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=RunConfig(max_workers=8, timeout=60),
        )
    finally:
        OpenAIInstrumentor().instrument()

    scores_lines = ["\n" + "=" * 50, "РЕЗУЛЬТАТЫ RAGAS", "=" * 50]
    for metric, score in result._repr_dict.items():
        if score != score:  # NaN check
            scores_lines.append(f"{metric:<22} {'?' * 20}  N/A")
        else:
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            scores_lines.append(f"{metric:<22} {bar}  {score:.3f}")
    scores_lines.append("=" * 50)

    # Save to Phoenix datasets
    try:
        df = result.to_pandas()
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
        df = result.to_pandas()

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
                "Запускает RAGAS-оценку и сохраняет результаты в Phoenix как датасет.\n"
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

            run_eval_btn = gr.Button("▶ Запустить оценку", variant="primary", size="lg")

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

            run_eval_btn.click(
                fn=run_eval,
                inputs=[api_key_input, eval_answer_model_dropdown, judge_model_dropdown, eval_retriever_radio, eval_dataset_file],
                outputs=[eval_log, eval_table],
            )


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
