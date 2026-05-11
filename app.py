import json
from datetime import datetime

import gradio as gr
import pandas as pd
import phoenix as px
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
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


def index_file(file, api_key: str):
    if file is None:
        return "Выбери файл для индексации"
    chunks = load_and_chunk(file.name)
    total = index_document(chunks)
    return f"Готово! Проиндексировано {total} чанков из файла {file.name}"


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

    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        model=judge_model,
        openai_api_key=api_key.strip(),
        openai_api_base=OPENROUTER_BASE_URL,
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key.strip(),
        openai_api_base=OPENROUTER_BASE_URL,
    ))

    result = evaluate(
        dataset=EvaluationDataset(samples),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    scores_lines = ["\n" + "=" * 50, "РЕЗУЛЬТАТЫ RAGAS", "=" * 50]
    for metric, score in result._repr_dict.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        scores_lines.append(f"{metric:<22} {bar}  {score:.3f}")
    scores_lines.append("=" * 50)

    # Save to Phoenix datasets
    try:
        df = result.to_pandas()
        client = px.Client()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_slug = model.split("/")[-1]
        dataset_name = f"eval-{ts}-{model_slug}-{retriever_mode}"
        client.upload_dataset(
            dataframe=df,
            dataset_name=dataset_name,
            input_keys=["user_input"],
            output_keys=["reference"],
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

    with gr.Row():
        api_key_input = gr.Textbox(
            label="OpenRouter API Key",
            type="password",
            placeholder="sk-or-v1-...",
            scale=3,
        )
        model_selector = gr.Dropdown(
            choices=[
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                "anthropic/claude-haiku-4-5",
                "anthropic/claude-sonnet-4-5",
                "google/gemini-2.0-flash",
                "google/gemini-2.5-flash-preview",
                "google/gemini-2.5-pro-preview",
                "deepseek/deepseek-chat",
                "deepseek/deepseek-v4-flash",
                "deepseek/deepseek-r1",
                "deepseek/deepseek-r1-0528",
                "qwen/qwen-2.5-72b-instruct",
                "moonshotai/kimi-k2",
                "meta-llama/llama-3.1-8b-instruct",
            ],
            value=DEFAULT_MODEL,
            label="Модель",
            scale=2,
        )

    with gr.Tabs():
        with gr.Tab("💬 Чат"):
            with gr.Row():
                top_k_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top-K фрагментов")
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
            index_btn = gr.Button("Индексировать", variant="primary")
            index_output = gr.Textbox(label="Результат", interactive=False)
            index_btn.click(fn=index_file, inputs=[file_input, api_key_input], outputs=index_output)

        with gr.Tab("🔬 Оценка"):
            gr.Markdown(
                "Запускает RAGAS-оценку и сохраняет результаты в Phoenix как датасет.\n"
                f"Посмотреть датасеты: [{phoenix_url}datasets]({phoenix_url}datasets)"
            )

            with gr.Row():
                judge_model_dropdown = gr.Dropdown(
                    choices=[
                        "deepseek/deepseek-v4-pro",
                        "openai/gpt-4o",
                        "openai/gpt-4o-mini",
                        "anthropic/claude-sonnet-4-5",
                        "google/gemini-2.5-flash-preview",
                    ],
                    value="deepseek/deepseek-v4-pro",
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
                inputs=[api_key_input, model_selector, judge_model_dropdown, eval_retriever_radio, eval_dataset_file],
                outputs=[eval_log, eval_table],
            )


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
