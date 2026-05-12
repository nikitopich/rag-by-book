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


def generate_answer(query: str, context_chunks: list[str], api_key: str, model: str) -> str:
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, max_retries=5)

    user_message = build_user_message(query, context_chunks)

    response = client.chat.completions.create(
        model=model,
        max_tokens=5000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content
