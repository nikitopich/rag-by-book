from collections.abc import Iterator

from openai import OpenAI
from config import OPENROUTER_BASE_URL

SYSTEM_PROMPT = """Ты отвечаешь на вопросы по книге.
                Используй ТОЛЬКО предоставленный контекст для ответа.
                Если ответа нет в контексте — честно скажи об этом.
                Отвечай на русском языке."""


def build_user_message(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"""Контекст из книги:

                    {context}

                    ---

                    Вопрос: {query}"""


def _messages(query: str, context_chunks: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(query, context_chunks)},
    ]


def stream_answer(query: str, context_chunks: list[str], api_key: str, model: str) -> Iterator[str]:
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, max_retries=5)
    stream = client.chat.completions.create(
        model=model,
        max_tokens=2000,
        messages=_messages(query, context_chunks),
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def generate_answer(query: str, context_chunks: list[str], api_key: str, model: str) -> str:
    return "".join(stream_answer(query, context_chunks, api_key, model))
