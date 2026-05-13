# RAG over a Book

A Gradio web app that lets you upload any book (`.txt`) and ask questions about it in a chat. The language model answers using only the relevant passages retrieved from the book — that's RAG (Retrieval-Augmented Generation).

Beyond the chat, the app includes built-in RAG evaluation powered by **DeepEval** and full request tracing via **Phoenix**.

---

## Features

- **Chat with your book** — ask a question, the system finds relevant passages and generates an answer
- **Upload books via UI** — drop a `.txt` file, tune chunk settings, index with one click
- **Two retriever modes** — vector (semantic search) and hybrid (BM25 + vector)
- **Debug mode** — shows retrieved chunks and the full prompt directly in the chat
- **RAG evaluation** — runs DeepEval against your question dataset and reports metrics
- **Tracing in Phoenix** — every LLM call is visible in a local observability dashboard

---

## Stack

| Component | Technology |
|---|---|
| UI | Gradio |
| Vector store | ChromaDB |
| Embeddings | `intfloat/multilingual-e5-base` (local) |
| LLM | Any model via OpenRouter |
| Evaluation | DeepEval (Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall) |
| Tracing | Arize Phoenix |

---

## Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd rag-by-book
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get an OpenRouter API key

Sign up at [openrouter.ai](https://openrouter.ai) and grab a key that looks like `sk-or-v1-...`. You paste it directly in the UI — no `.env` file needed.

### 4. Run the app

```bash
python app.py
```

The browser opens automatically. If not, go to [http://127.0.0.1:7860](http://127.0.0.1:7860).

---

## How to Use

### Index a book

1. Open the **Индексация** tab
2. Upload a `.txt` file
3. Optionally adjust chunk size (default: 500 chars, overlap: 100)
4. Click **Индексировать**

### Ask questions

1. Open the **Чат** tab
2. Enter your OpenRouter API key at the top
3. Choose a model and retriever mode
4. Start chatting

### Evaluate RAG quality

1. Prepare a dataset as a JSON file — a list of objects with `question` and `reference` fields
2. Open the **Оценка** tab
3. Choose an answer model and a judge model (use instruction-following models like `google/gemma-4-31b-it` — reasoning models like DeepSeek R1 produce unstable JSON and may break evaluation)
4. Click **Запустить оценку**

Example dataset:

```json
[
  {
    "question": "What is this book about?",
    "reference": "The book is about ..."
  }
]
```

By default the app looks for `eval_dataset.json` in the project root.

---

## Phoenix — LLM Request Tracing

[Arize Phoenix](https://phoenix.arize.com) is a local observability tool for LLM applications. It starts automatically alongside the app.

**What you can see in Phoenix:**
- Every LLM call: prompt, response, latency, token counts
- Full request chains (retrieval → generation)
- DeepEval results saved as datasets

Phoenix is available at the URL shown in the app header (usually [http://localhost:6006](http://localhost:6006)).

---

## Project Structure

```
app.py              — Gradio UI and evaluation logic
chunker.py          — splits text into chunks
indexer.py          — loads chunks into ChromaDB
retriever.py        — vector and hybrid search
generator.py        — LLM answer generation
config.py           — constants (models, paths, parameters)
tracing.py          — Phoenix setup
eval_dataset.json   — evaluation dataset
```
