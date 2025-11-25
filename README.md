# CV QA Agent

A simple, production-friendly FastAPI backend that exposes a small QA service over a CV (PDF). The app loads a CV, chunks it into sections, builds an embeddings index (FAISS) and answers questions using a chat/completion API.


---

## Key features

* Loads a PDF CV and extracts text using `pypdf`.
* Optional LLM-based chunking of the CV (with a local fallback).
* Builds embeddings per section and a FAISS index for retrieval.
* Chat endpoint that answers questions using only CV context.
* Simple health check endpoint.
* Works in a development mode without external model calls (`skip_model_calls`).

---

## Quick start

### Requirements

* Python 3.10+ recommended
* `pip` and a virtual environment
* System packages: none special, but FAISS must be installed; use the appropriate wheel for your platform.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file next to `src/services.py` or set these in your shell:

```ini
GITHUB_TOKEN=ghp_...             # Required for model/embed API auth in current code
CHAT_COMPLETIONS_URL=...         # optional override (defaults provided in code)
EMBEDDINGS_URL=...               # optional override
CHAT_MODEL=openai/gpt-4.1-mini   # optional override
EMBED_MODEL=openai/text-embedding-3-small
```

> If you do not want to call remote models while developing the UI, set `skip_model_calls=True` when constructing `CVService` or set that behavior via code (the example app already supports it in the constructor).

### Place the CV file

Put a PDF file (the CV) in the project `static` directory. By default the app looks for a file named:

```
Fatma Alzahraa Alhabbash - CV.pdf
```

If that file is missing, the app will try any `*.pdf` in the static directory. If none are found, startup will fail with a clear error.

### Run the app (development)

Start the FastAPI app with uvicorn:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` to see the OpenAPI docs and test the endpoints.

---

## API

### `POST /api/chat`

Body (JSON):

```json
{
  "message": "Your question about the CV",
  "history": [{"role": "user", "content": "..."}]  
}
```

Response:

```json
{ "response": "Answer from the agent" }
```

**Behavior:** the service retrieves the top-k relevant CV sections, builds a context, and calls a chat completions API. When `skip_model_calls` is enabled the response is a placeholder useful for UI development.

### `GET /api/health`

Simple readiness check that returns `{"status": "ok"}` when the FAISS index is ready.

---

## Project layout (important files)

```
src/
├─ main.py           # creates and configures the FastAPI app
├─ routes/routes.py  # /api endpoints (chat, health)
├─ services.py       # CVService: loads CV, chunks, builds embeddings/index, answers
└─ static/           # static files and the CV PDF go here
```

---

## Configuration notes & troubleshooting

* **Missing GITHUB_TOKEN**: the service expects a token for the embedding/chat endpoints. If not set, requests to remote APIs will fail; the code raises a clear RuntimeError in `_check_token()`.

* **CV file not found**: startup raises `FileNotFoundError` with a helpful log message listing what was checked.

* **FAISS issues**: FAISS must be installed for your platform. If you have trouble installing, use `skip_model_calls=True` for UI testing.

* **Static files not served**: the app searches several candidate locations (`src/static`, `static`, `backend/static`) and logs which directory it uses. Ensure your static dir exists and contains the CV.

---

## Example curl usage

Health:

```bash
curl http://localhost:8000/api/health
```

Chat (replace QUESTION):

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "QUESTION", "history": []}'
```

---
