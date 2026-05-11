import anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL

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


def generate_answer(query: str, context_chunks: list[str]) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_message = build_user_message(query, context_chunks)

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=5000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return next(block.text for block in response.content if block.type == "text")
