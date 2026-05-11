from chunker import load_and_chunk
from indexer import index_document
from retriever import retrieve
from generator import generate_answer, build_user_message, SYSTEM_PROMPT

DEBUG = False
TOP_K = 5

HELP_TEXT = """
Команды:
  /index [файл]   — проиндексировать книгу (по умолчанию aa.txt)
  /debug           — вкл/выкл debug-режим (показывает чанки, scores, промпт)
  /topk <число>    — изменить количество фрагментов (сейчас: {top_k})
  /help            — показать эту справку
  /exit            — выйти

Всё остальное — вопрос по книге.
"""


def handle_command(line: str) -> bool:
    global DEBUG, TOP_K

    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()

    if cmd in ("/exit", "/quit", "/q"):
        print("Пока!")
        return False

    elif cmd == "/help":
        print(HELP_TEXT.format(top_k=TOP_K))

    elif cmd == "/index":
        file_path = parts[1] if len(parts) > 1 else "aa.txt"
        print(f"Загружаю и разбиваю на чанки: {file_path}")
        chunks = load_and_chunk(file_path)
        print(f"Получилось {len(chunks)} чанков")
        print("Индексирую в ChromaDB...")
        total = index_document(chunks)
        print(f"Готово! Проиндексировано {total} чанков.\n")

    elif cmd == "/debug":
        DEBUG = not DEBUG
        state = "ВКЛ" if DEBUG else "ВЫКЛ"
        print(f"Debug-режим: {state}\n")

    elif cmd == "/topk":
        if len(parts) > 1 and parts[1].isdigit():
            TOP_K = int(parts[1])
            print(f"Top-K: {TOP_K}\n")
        else:
            print(f"Текущий top-k: {TOP_K}. Использование: /topk 3\n")

    else:
        print(f"Неизвестная команда: {cmd}. Введи /help\n")

    return True


def print_debug(question: str, results: dict, context_chunks: list[str]):
    """Выводит debug-информацию: найденные чанки со scores и итоговый промпт."""
    print("\n" + "=" * 60)
    print("DEBUG: НАЙДЕННЫЕ ЧАНКИ (отсортированы по релевантности)")
    print("=" * 60)

    for i, (chunk_id, doc, dist) in enumerate(
        zip(results["ids"], results["documents"], results["distances"]), 1
    ):
        print(f"\n--- [{i}] {chunk_id} | distance: {dist:.4f} ---")
        print(doc)

    print("\n" + "=" * 60)
    print("DEBUG: ПРОМПТ, ОТПРАВЛЕННЫЙ В LLM")
    print("=" * 60)
    print(f"\n[SYSTEM]\n{SYSTEM_PROMPT}")
    print(f"\n[USER]\n{build_user_message(question, context_chunks)}")
    print("=" * 60 + "\n")


def ask_question(question: str):
    results = retrieve(question, top_k=TOP_K)
    chunks = results["documents"]

    if DEBUG:
        print_debug(question, results, chunks)

    answer = generate_answer(question, chunks)
    print(f"\n{answer}\n")


def main():
    print("📚 RAG-чат с книгой")
    print("Введи /help для списка команд, или просто задай вопрос.\n")

    while True:
        try:
            line = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nПока!")
            break

        if not line:
            continue

        if line.startswith("/"):
            if not handle_command(line):
                break
        else:
            ask_question(line)


if __name__ == "__main__":
    main()
