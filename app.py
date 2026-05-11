import gradio as gr
from retriever import retrieve
from generator import generate_answer, build_user_message, SYSTEM_PROMPT
from chunker import load_and_chunk
from indexer import index_document
from config import DEFAULT_MODEL
from tracing import setup_tracing

phoenix_url = setup_tracing()


def chat(message: str, history: list, api_key: str, model: str, top_k: int, debug: bool):
    if not api_key or not api_key.strip():
        return "Введи OpenRouter API ключ в поле выше"

    try:
        results = retrieve(message, top_k=int(top_k))
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


with gr.Blocks(title="RAG-чат с книгой") as demo:
    gr.Markdown(f"# 📚 RAG-чат с книгой\nЗадавай вопросы по проиндексированной книге. Phoenix: [{phoenix_url}]({phoenix_url})")

    with gr.Row():
        api_key_input = gr.Textbox(
            label="OpenRouter API Key",
            type="password",
            placeholder="sk-or-v1-...",
            scale=3,
        )
        model_selector = gr.Dropdown(
            choices=[
                # OpenAI
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                # Anthropic
                "anthropic/claude-haiku-4-5",
                "anthropic/claude-sonnet-4-5",
                # Google Gemini
                "google/gemini-2.0-flash",
                "google/gemini-2.5-flash-preview",
                "google/gemini-2.5-pro-preview",
                # DeepSeek
                "deepseek/deepseek-chat",
                "deepseek/deepseek-r1",
                "deepseek/deepseek-r1-0528",
                # Qwen
                "qwen/qwen-2.5-72b-instruct",
                "qwen/qwq-32b-preview",
                # Kimi
                "moonshotai/kimi-k2",
                # Meta
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
                debug_checkbox = gr.Checkbox(label="Debug-режим", value=False)

            gr.ChatInterface(
                fn=chat,
                additional_inputs=[api_key_input, model_selector, top_k_slider, debug_checkbox],
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


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
