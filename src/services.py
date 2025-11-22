# 
# backend/services.py
import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import faiss
import requests
from dotenv import load_dotenv
from pypdf import PdfReader

log = logging.getLogger(__name__)

# Load .env relative to this file (safer)
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # fallback: load from cwd if present
    load_dotenv()

# Configuration (tweak these if using a different provider)
CHAT_COMPLETIONS_URL = os.getenv("CHAT_COMPLETIONS_URL", "https://models.github.ai/inference/chat/completions")
EMBEDDINGS_URL = os.getenv("EMBEDDINGS_URL", "https://models.github.ai/inference/embeddings")
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")

# Request defaults
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15.0"))  # seconds

def get_headers():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        # Do not raise here; let the calling code raise with context if needed
        return None
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

class CVService:
    def __init__(self, cv_path: str = "cv.pdf", skip_model_calls: bool = False):
        """
        skip_model_calls: if True, don't call LLM/embed endpoints (useful for UI dev).
        """
        self.cv_path = cv_path
        self.cv_text: str = ""
        self.sections: List[Dict[str, Any]] = []
        self.section_texts: List[str] = []
        self.section_meta: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.ready = False
        self.skip_model_calls = skip_model_calls

    def _check_token(self):
        headers = get_headers()
        if headers is None:
            raise RuntimeError(
                "GITHUB_TOKEN environment variable is not set. "
                "Set it in your shell or in backend/.env (GITHUB_TOKEN=...)."
            )
        return headers

    def load_cv_text(self) -> None:
        log.info("Loading CV from %s", self.cv_path)
        if not Path(self.cv_path).exists():
            raise FileNotFoundError(f"CV file not found: {self.cv_path}")
        reader = PdfReader(self.cv_path)
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
        self.cv_text = "\n\n".join(parts)
        log.info("Loaded CV text (length=%d)", len(self.cv_text))

    def llm_chunk_cv(self) -> None:
        log.info("Requesting LLM to chunk CV into sections...")
        if self.skip_model_calls:
            # fallback: naive chunking (split by headings heuristically)
            log.info("skip_model_calls=True, using heuristic chunking.")
            lines = [l.strip() for l in self.cv_text.splitlines() if l.strip()]
            # very naive: treat long blocks as a single "Section"
            self.sections = [{"section": "CV", "content": self.cv_text}]
            return

        headers = self._check_token()
        messages = [
            {
                "role": "system",
                "content": (
                    "Split the following CV into logical sections (e.g. Experience, Education, "
                    "Summary, Skills). Return a JSON array of objects: "
                    "[{\"section\": \"<name>\", \"content\": \"<text>\"}, ...]."
                ),
            },
            {"role": "user", "content": self.cv_text},
        ]
        payload = {"model": CHAT_MODEL, "messages": messages, "max_tokens": 1024, "temperature": 0.0}

        resp = requests.post(CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        try:
            resp.raise_for_status()
        except Exception:
            log.exception("Chunking API request failed: status=%s body=%s", resp.status_code, resp.text[:1000])
            raise

        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # remove triple-backticks if present
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("Chunking LLM returned unexpected format (expected list).")
        self.sections = parsed
        log.info("LLM returned %d sections", len(self.sections))

    def build_embeddings_and_index(self) -> None:
        log.info("Building embeddings for %d sections", len(self.sections))
        vectors = []
        self.section_texts = []
        self.section_meta = []

        if self.skip_model_calls:
            # Create random vectors so index can be built for UI exploration (dimension 1536 typical)
            dim = 1536
            for sec in self.sections:
                text_input = sec.get("content", "")
                vectors.append(np.random.rand(dim).astype("float32"))
                self.section_texts.append(text_input)
                self.section_meta.append(sec.get("section", "unknown"))
            self.embeddings = np.vstack(vectors).astype("float32")
            d = self.embeddings.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(self.embeddings)
            self.index = index
            self.ready = True
            log.info("Built fake index for dev (skip_model_calls=True)")
            return

        headers = self._check_token()

        for sec in self.sections:
            section_name = sec.get("section", "unknown")
            content = sec.get("content", "")
            if isinstance(content, list):
                text_input = "\n".join(c if isinstance(c, str) else json.dumps(c) for c in content)
            else:
                text_input = str(content)

            payload = {"model": EMBED_MODEL, "input": text_input}
            resp = requests.post(EMBEDDINGS_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            try:
                resp.raise_for_status()
            except Exception:
                log.exception("Embedding API request failed for section '%s': status=%s body=%s",
                              section_name, getattr(resp, "status_code", None), getattr(resp, "text", "")[:1000])
                raise

            data = resp.json()
            emb = np.array(data["data"][0]["embedding"], dtype=np.float32)
            vectors.append(emb)
            self.section_texts.append(text_input)
            self.section_meta.append(section_name)

        if not vectors:
            raise RuntimeError("No embeddings created - cannot build index.")

        self.embeddings = np.vstack(vectors).astype("float32")
        d = self.embeddings.shape[1]
        log.info("Embeddings shape: %s", self.embeddings.shape)
        index = faiss.IndexFlatL2(d)
        index.add(self.embeddings)
        self.index = index
        self.ready = True
        log.info("FAISS index built with %d vectors, dimension %d", index.ntotal, d)

    def initialize(self) -> None:
        """Run the full startup. Raises on error with helpful logs."""
        self.load_cv_text()
        self.llm_chunk_cv()
        self.build_embeddings_and_index()

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        if self.index is None:
            raise RuntimeError("Index not built. Call initialize() first.")

        if self.skip_model_calls:
            # returns top_k dummy sections
            results = []
            for i in range(min(top_k, len(self.section_texts))):
                results.append({"section": self.section_meta[i], "content": self.section_texts[i]})
            return results

        headers = self._check_token()
        payload = {"model": EMBED_MODEL, "input": query}
        resp = requests.post(EMBEDDINGS_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        try:
            resp.raise_for_status()
        except Exception:
            log.exception("Query embedding request failed: status=%s body=%s", getattr(resp, "status_code", None), getattr(resp, "text", "")[:1000])
            raise

        q_emb = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)[None, :]

        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            results.append({"section": self.section_meta[idx], "content": self.section_texts[idx]})
        return results

    def answer_question(self, message: str, history: List[Dict[str, str]] = None) -> str:
        if history is None:
            history = []

        chunks = self.retrieve(message, top_k=3)
        context = "\n\n".join(f"{c['section']}: {c['content']}" for c in chunks)
        system_prompt = (
            "You are acting as the person described in this CV (Fatma Alzahraa Alhabbash).\n"
            "Use ONLY the context below (extracted from the CV) to answer user questions. "
            "If the information is not in the CV, politely say you don't have that info.\n\n"
            f"Context:\n{context}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for item in history:
            messages.append(item)
        messages.append({"role": "user", "content": message})

        if self.skip_model_calls:
            # A naive canned reply for dev (so UI works while model calls are stubbed)
            return "This is a placeholder response (skip_model_calls=True). Backend retrieved: " + "; ".join([c["section"] for c in chunks])

        headers = self._check_token()
        payload = {"model": CHAT_MODEL, "messages": messages, "max_tokens": 256, "temperature": 0.7}
        resp = requests.post(CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        try:
            resp.raise_for_status()
        except Exception:
            log.exception("Chat completions request failed: status=%s body=%s", getattr(resp, "status_code", None), getattr(resp, "text", "")[:1000])
            raise

        return resp.json()["choices"][0]["message"]["content"]
