from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_and_chunk(file_path: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""
    chunk_index = 0

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
        else:
            if current_chunk:
                chunks.append({
                    "id": f"chunk_{chunk_index}",
                    "text": current_chunk,
                    "metadata": {"index": chunk_index, "char_start": _find_position(text, current_chunk)},
                })
                chunk_index += 1

            if current_chunk and overlap > 0:
                overlap_text = current_chunk[-overlap:]
                current_chunk = f"{overlap_text}\n\n{paragraph}"
            else:
                current_chunk = paragraph

    if current_chunk:
        chunks.append({
            "id": f"chunk_{chunk_index}",
            "text": current_chunk,
            "metadata": {"index": chunk_index, "char_start": _find_position(text, current_chunk)},
        })

    return chunks


def _find_position(full_text: str, chunk_text: str) -> int:
    search = chunk_text[:50]
    pos = full_text.find(search)
    return pos if pos != -1 else 0
