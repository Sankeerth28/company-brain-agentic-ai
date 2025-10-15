# Company Brain — Agentic AI (Hackathon MVP)

**What this is:**  
A 48-hour hackathon prototype that demonstrates an _agentic_ AI combining local RAG retrieval, reasoning, and concrete next actions.  
- Uses local sentence-transformer embeddings for retrieval and a simple query layer.
- `rag_agent.py` supports mock (offline) mode by default to avoid API quota issues.

## Project Structure

```
prosper-hack/
├─ app_streamlit.py         # Streamlit demo UI
├─ ingestion.py             # Ingest files → chunk → compute local embeddings → persist in db/
├─ rag_agent.py             # Retriever + (mock or live OpenAI) answer generator
├─ data/                    # Put demo docs here (txt, md, csv, pdf)
├─ db/                      # Persisted embeddings/texts after running ingestion.py
├─ .env                     # Optional (OPENAI_API_KEY, USE_MOCK_DEMO=false to enable live LLM)
├─ requirements.txt
└─ README.md
```

## Quick Start (Windows PowerShell)

1. **Open PowerShell** in project root (e.g. `D:\prosper-hack`).

2. **Create and activate virtual environment:**
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    # If PowerShell blocks scripts, run once:
    # Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
    ```

3. **Install dependencies:**
    ```powershell
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. *(Optional)* **Add demo docs:**
    - Place files in `D:\prosper-hack\data`  
      *(txt, md, csv, pdf)*  
    - Or use the Streamlit uploader later.

5. **Remove old DB (safe):**
    ```powershell
    Remove-Item -Recurse -Force .\db -ErrorAction SilentlyContinue
    ```

6. **Run ingestion (creates db/embeddings.npy, texts.json, metadatas.json):**
    ```powershell
    python ingestion.py
    ```
    You should see output like:  
    `Found X files in data. Total chunks: N. Computing embeddings with all-MiniLM-L6-v2 (local)... Ingested N chunks. Files saved in db/`

7. **Start Streamlit app:**
    ```powershell
    streamlit run app_streamlit.py
    ```
    - Open the URL printed (usually [http://localhost:8501](http://localhost:8501)).
    - Upload/index files via the UI or use files already in `data/`.
    - Then ask queries!

---

## Toggle Mock vs Live LLM

- **Default:** `rag_agent.py` uses **mock mode** (`USE_MOCK_DEMO=true`) — safe and deterministic for demos.
- **To enable live OpenAI calls:**
    1. Set these in `.env`:
        ```
        USE_MOCK_DEMO=false
        OPENAI_API_KEY=sk-...
        OPENAI_MODEL=gpt-3.5-turbo
        ```
    2. **Restart** Streamlit (or restart the app environment) so the env is reloaded.

> **Note:** If you run live calls, keep them minimal during the demo to avoid quota/latency risk.

---

## Re-run / Debug Checklist

- Remove old DB:
    ```powershell
    Remove-Item -Recurse -Force .\db
    python ingestion.py
    ```
- Start Streamlit:
    ```powershell
    streamlit run app_streamlit.py
    ```
- Reinstall requirements (if needed):
    ```powershell
    pip install -r requirements.txt
    ```
- If a model download stalls during ingestion, give it a minute — the sentence-transformers model downloads once.

---

## Common Issues & Fixes

- **PowerShell execution blocked:**  
  Run:  
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force`
- **API rate limit/quota (OpenAI):**  
  Use mock mode (default) or set `USE_MOCK_DEMO=true`.
- **Missing db/ or embeddings.npy:**  
  Run `python ingestion.py` to rebuild.
- **Slow first run:**  
  Sentence-transformers downloads the model on first embed; subsequent runs are fast.

---

## How to Demo (60–90s)

1. Open app. Show “Ingest documents” panel, mention local indexing.
2. Ask:  
   “What are the top three customer complaints in Q3 and what should we do?”  
   → Click Run Query.
3. Point at **Answer** (concise), **Proposed actions** (owner + est hours), and **Retrieved context** (grounding).
4. Closing line:  
   **“It doesn’t just answer — it acts: planning, prioritizing, and proposing concrete next steps.”**

---

## Notes for Judges / Tech Questions

- **Retrieval:** Local sentence-transformer embeddings + cosine similarity.
- **Reasoning:** Mock canned actions or live LLM (OpenAI) consuming retrieved context.
- **Scale path:** Swap in a cloud vector DB (Chroma/Weaviate/FAISS), scale embeddings generation to batch workers, or plug in hosted LLMs for production reasoning.
